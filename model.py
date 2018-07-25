import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import time
import numpy as np

'''this is the model object. it consists mostly of tensorflow variables and has a few functions for computing probabilities'''
'''it gets called in train.py and the poem generation script'''

class Model():
    def __init__(self, args, infer=False):
        '''these arguments appear in full in train.py'''
        self.args = args

        '''it seems this will never happen'''
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        '''the types of models at our disposal'''
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))
        
        '''this is a placeholder for dropout, defaults to 0 for computing f(x)'''
        self.dropout = tf.placeholder_with_default(0., shape=())

        '''the structure of the cell is formed here'''
        cells = []
        for _ in range(args.num_layers):            
            cell = cell_fn(args.rnn_size)
            cell = rnn.DropoutWrapper(cell, output_keep_prob= 1 - self.dropout)
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells)

        '''the model object includes train data, test data if specified, and some batch/epoch pointers'''
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        self.test_x = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
        self.test_y = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])           

        '''i never figured out what this does'''        
        tf.summary.scalar("time_batch", self.batch_time)
        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                #with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                #tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                #tf.summary.histogram('histogram', var)

        '''begin defining model variables'''        
        with tf.variable_scope('rnnlm'):
            '''the get_variable is an initializer: here we get weights, then biases'''
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size],
                                        initializer = tf.truncated_normal_initializer(mean = 0., stddev = .1, seed = 2018, dtype = tf.float32))
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size],
                                        initializer = tf.constant_initializer(np.repeat(0., args.vocab_size),  tf.float32, args.vocab_size))
            variable_summaries(softmax_b)
            with tf.device("/cpu:0"):

                '''W will be the word embeddings'''

                self.W = tf.Variable(tf.constant(0.0, shape=[args.vocab_size, args.embedding_dim]), name="W")

                self.embedding_placeholder = tf.placeholder(tf.float32, [args.vocab_size, args.embedding_dim])
                self.embedding_init = self.W.assign(self.embedding_placeholder)

                '''the data to input to the model for some computation'''
                inputs = tf.split(tf.nn.embedding_lookup(self.W, self.input_data), args.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
                
                test_inputs = tf.split(tf.nn.embedding_lookup(self.W, self.test_x), args.seq_length, 1)
                test_inputs = [tf.squeeze(test_input_, [1]) for test_input_ in test_inputs]
                
        '''im not 100% on this one, but it never gets used'''        
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            print(tf.argmax(prev,1))
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            print(prev_symbol)
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        '''the model output, logits, probability distbution, and loss'''
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.temp = tf.placeholder_with_default(1., shape=())
        self.temped_logits = self.logits / self.temp
        self.probs = tf.nn.softmax(self.temped_logits)
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        '''the test output, logits, and loss'''        
        test_outputs, test_last_state = legacy_seq2seq.rnn_decoder(test_inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        test_output = tf.reshape(tf.concat(test_outputs, 1), [-1, args.rnn_size])
        self.test_logits = tf.matmul(test_output, softmax_w) + softmax_b
        self.test_probs = tf.nn.softmax(self.test_logits)
        test_loss = legacy_seq2seq.sequence_loss_by_example([self.test_logits],
                [tf.reshape(self.test_y, [-1])],
                [tf.ones([self.test_y.shape[0]])],
                args.vocab_size)
        self.test_cost = tf.reduce_sum(test_loss) / args.batch_size / args.seq_length        
                
        tf.summary.scalar("cost", self.cost)

        '''for retrieval of the final states'''        
        self.final_state = last_state
        self.test_final_state = test_last_state

        '''the optimizer'''
        self.lr = tf.Variable(0.0, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)

        '''so this was the really hacky way i got the embeddings to be trainable on demand: two channels for optimization. turn whichever you wish'''
        self.tvars = tf.trainable_variables()
        self.tvars_no_W = [var for var in tf.trainable_variables() if "W:0" not in var.name]
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars),
                args.grad_clip)
        grads_no_W, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars_no_W),
                args.grad_clip)

        '''running this updates the parameters'''        
        self.train_op = optimizer.apply_gradients(zip(grads, self.tvars))
        self.train_op_no_W = optimizer.apply_gradients(zip(grads_no_W, self.tvars_no_W))
        
        
    def score_a_list(self, sess, vocab, seq, temp = 1):
        '''given a sequence, this computes the probability of the sequence conditions on the 0th word'''

        n_words = len(seq)

        tensor = np.array(list(map(vocab.get, seq)))
    
        xdata = tensor
        ydata = xdata[1:]
        xdata = xdata[:-1]

        prob = 0
        state = sess.run(self.initial_state)

        input = np.zeros((1,1))
        
        for i in range(n_words-1):
            input[0,0] = xdata[i]
            feed = {self.input_data: input, self.initial_state: state, self.temp : temp}
            [probs, state] = sess.run([self.probs, self.final_state],
                                        feed)
            prob += np.log(probs.squeeze()[ydata[i]])
        return prob

        
    def compute_fx(self, sess, vocab, p, x, state, temp):
        '''produce the (new) probability distribution given a session, vocab, previous prob. distribution, sequence, temperature, and state'''

        input = np.array([[vocab[x[0]]]])

        if state is None:
            state = sess.run(model.initial_state)
        
        feed = {self.input_data: input, self.initial_state: state, self.temp : temp}
        
        [probs, state] = sess.run([self.probs, self.final_state],
                                    feed)
        
        dist = p.squeeze() + np.log(probs.squeeze())        
        
        return dist.squeeze(), state
    
    
    def beamscore(self, sess, vocab, p, x, y, state, temp):
        """Returns log p(y+x), and state after prediction,"""
        """Given a target y and sequence up through x and its prob and its state"""
        
        input = np.array([[vocab[x[0]]]])
        y = np.array([[vocab[y]]])
        
        feed = {self.input_data: input, self.initial_state: state, self.temp : temp}
        
        [probs, state] = sess.run([self.probs, self.final_state],
                                    feed)
        
        prob = np.log(probs.squeeze()[y])
        
        ''' all penalties/boosts can be here '''
        ''' repetition. in wordpool. assonance or consonance '''
        
        return p + prob, state 
        