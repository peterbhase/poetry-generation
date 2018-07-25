from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.training.learning_rate_decay import cosine_decay_restarts
import matplotlib.pyplot as plt
import argparse
import time
import os
from six.moves import cPickle
from utils import TextLoader
from model import Model

'''this is the training script. starts with a whole bunch of arguments, then trains (and saves) a model'''                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/combined',
                       help='data directory containing input.txt')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=1000,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=1,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default= 5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9,
                       help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=0.7,
                       help='%% of gpu memory to be allocated to this process. Default is 70%')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--test_split', type=float, default = 0.,
                        help = "percent of data OR number of words to use as hold-out set")
    parser.add_argument('--embedding_dim', type=int, default = 300,
                        help = "dimensionality of word embeddings")
    parser.add_argument('--dropout', type = float, default = .3,
                        help = "dropout rate for cell output")
    parser.add_argument('--reverse', type = int, default = 1,
                        help = "bool for reversing training text")
    # 0 <-> never train embeddings, 1 <-> always train embeddings, c greater than 1 <-> train embeddings after c epochs
    parser.add_argument('--trainable_embeddings', type = int, default = 15,
                        help = "0 to never train, 1 to always train, c to train after c epochs")
    
    
    args = parser.parse_args()
    train(args)

def train(args):
    '''start by getting the data_loader object'''        
    data_loader = TextLoader(args.reverse, args.data_dir, args.test_split, args.batch_size, args.seq_length, args.input_encoding)

    '''some informative prints'''        
    args.vocab_size = data_loader.vocab_size 
    print("Train size: ", data_loader.num_batches * args.batch_size)
    if args.test_split > 0:
        print("Test size: ", data_loader.test_num_batches * args.batch_size)
    print("Vocab size: ", args.vocab_size)

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)
        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    '''idk what the pickle.dump does'''        
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)
  
    '''start up the model'''        
    model = Model(args)
    
    '''if a test split is requested, get it'''            
    if args.test_split > 0:
        test_x = data_loader.test_x
        test_y = data_loader.test_y

    '''not sure about this stuff'''        
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

    '''begin the session for training'''        
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # take a look at the learning rate schedule, if so desired (note parameters here should match the ones used)
        plot = False
        if plot:
            n = args.num_epochs * data_loader.num_batches
            n = 150000
            x = np.arange(n)
            y = np.zeros((n,1))
            y = cosine_decay_restarts(args.learning_rate,
                                           x, # shift down every epoch
                                           50000, # check out this sweet graph https://github.com/tensorflow/tensorflow/pull/11749
                                           .9, # doesn't hurt to look at the tf docs too
                                           .1,
                                           1e-12
                                           ).eval()
            plt.figure()
            plt.plot(x,y)
            plt.title("Learning rate schedule")
            plt.show()

        '''not sure what this does'''        
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        
        '''fun fact: you cant put comments inside if-else clauses'''        
        '''initialize from a previous model OR start from scratch, which means grabbing GloVe embeddings ''' 
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        else:
            print("Loading my knowledge of the English language...")
            embeddings = data_loader.get_embeddings()
            sess.run([model.embedding_init], {model.embedding_placeholder: embeddings})        
        
        '''iterate over the range of epochs specified'''        
        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            #sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e))) #this is the vanilla exponential deca

            '''learning rate decay is cosine annealing'''        
            sess.run(tf.assign(model.lr,
                cosine_decay_restarts(args.learning_rate,
                                               e * data_loader.num_batches, # shift down every epoch
                                               20000, # check out this sweet graph https://github.com/tensorflow/tensorflow/pull/11749
                                               1, # doesn't hurt to look at the tf docs too
                                               .1,
                                               1e-12
                                               )))

            '''reset the pointer to start from the beginning'''
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0
            if args.init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)
            if args.init_from is not None:
                data_loader.pointer = model.batch_pointer.eval()
                args.init_from = None
        
            '''iterative over the batches in the dataset'''        
            for b in range(data_loader.pointer, data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                
                '''the feed dictionary gets passed to the model when tensorflow variables are computed'''        
                feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                        model.batch_time: speed, model.dropout: args.dropout}

                '''variables to be trained, either with or without word embeddings'''        
                run_list_full = [merged, model.cost, model.final_state, model.train_op, model.inc_batch_pointer_op]
                run_list_no_W = [merged, model.cost, model.final_state, model.train_op_no_W, model.inc_batch_pointer_op]
                # YES, TRAIN THE EMBEDDINGS
                if args.trainable_embeddings == 1: 
                    summary, train_loss, state, _, _ = sess.run(run_list_full, feed)            
                # NO, DO NOT TRAIN THE EMBEDDINGS (train_op_no_W)
                elif args.trainable_embeddings == 0: 
                    summary, train_loss, state, _, _ = sess.run(run_list_no_W, feed)            
                # it's been e epochs, so start training the embeddings
                elif e > args.trainable_embeddings:
                    summary, train_loss, state, _, _ = sess.run(run_list_full, feed)
                # it hasn't been e epochs, don't train the embeddings 
                else:           
                    summary, train_loss, state, _, _ = sess.run(run_list_no_W, feed)

                '''some diagnostics to be printed, and the model gets saved here too'''        
                train_writer.add_summary(summary, e * data_loader.num_batches + b)
                speed = time.time() - start
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                    print("learning rate: ", model.lr.eval())
                    
                    #TEST LOSS EVAL - evaluates batch by batch with same batch size as for training
                    if(args.test_split > 0):
                        test_loss = 0
                        batches_in_test = len(test_x)
                        save_state = state
                        state = sess.run(model.initial_state)
                        for i in range(batches_in_test):
                            feed = {model.test_x: test_x[i], model.test_y: test_y[i], model.initial_state: state}
                            loss, state, _ = sess.run([model.test_cost, model.test_final_state,
                                                    model.inc_batch_pointer_op], feed)
                            test_loss += loss
                        test_loss = test_loss / batches_in_test
                        state = save_state
                        print("test_loss = {:.3f}".format(test_loss))
                        
        '''one final evaluation of the entire dataset to check the loss'''        

        data_loader.reset_batch_pointer()
        state = sess.run(model.initial_state)
        ovr_loss = 0
        start = time.time()
        for b in range(data_loader.pointer, data_loader.num_batches):
            x, y = data_loader.next_batch()
            feed = {model.input_data: x, model.targets: y, model.initial_state: state}
            train_loss, state, _ = sess.run([model.cost, model.final_state,
                                                    model.inc_batch_pointer_op], feed)        
            ovr_loss += train_loss

        speed = time.time() - start
        print("ovr_train_loss = {:.3f}, time_to_eval = {:.3f}".format(ovr_loss / data_loader.num_batches, speed))
        
        '''lets you initialize a model without training it'''        
        if args.num_epochs == 0: 
            saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
            print("model saved to {}".format(checkpoint_path))
        
        train_writer.close()

if __name__ == '__main__':
    '''set a seed for reproducible results!'''        
    np.random.seed(2018)
    main()
