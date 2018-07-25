# -*- coding: utf-8 -*-
import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools
import nltk
from gensim.models.keyedvectors import KeyedVectors

'''here be utilities. which really means the textloader object to help with training'''                

class TextLoader():
    '''constructor, these arguments will appear in train.py'''    
    '''mainly the work gets done by self.preprocess'''    
    def __init__(self, reverse, data_dir, test_split, batch_size, seq_length, encoding=None):
        self.reverse = reverse
        self.data_dir = data_dir
        self.test_split = test_split
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # Let's not read vocab and data from file. We may change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, encoding)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.create_test()
        self.reset_batch_pointer()

        
    def nltk_clean(self, string):
        '''nltk cleaning'''        

        string = nltk.word_tokenize(string.strip().lower())
        string = ' '.join(string)
        return string

    def simple_clean(self,string):
        '''this is the cleaning function which gets used. cleans out all sorts of things'''                     
        '''note the difference between deleting and adding space, e.g. with punctuation '''
        '''eventually the text gets .split() on'''

        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"-", "", string)
        string = re.sub(r":", "", string)

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'d", "ed", string) #for the old style
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"I\'ve", "I have", string)
        string = re.sub(r"\'ll", " will", string)

        string = re.sub(r"[0-9]+", "EOS", string) # EOS tag for numeric titling, but this messes up normal numeral use

        string = re.sub(r";", ",", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\.", " . ", string)

        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\s{2,}", " ", string)
        
        return string.strip().lower()
        

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file, encoding):
        '''reverses the text (if specified), calls build_vocab'''
        '''saves the vocab'''

        with codecs.open(input_file, "r", encoding=encoding) as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        # data = self.nltk_clean(data)
        data = self.simple_clean(data)
        x_text = data.split()
                
        if self.reverse:
            x_text = [w for w in reversed(x_text)]

        self.vocab, self.words = self.build_vocab(x_text)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        self.tensor = np.array(list(map(self.vocab.get, x_text)))
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    
    def get_embeddings(self):
        '''get GloVe embeddings, or uses a (single!) random vector if one can't be found'''
        glove_model = KeyedVectors.load_word2vec_format('./storyline_for_reference/glove.6B.300d.word2vec.txt', binary=False)
        embeddings = np.zeros((self.vocab_size, 300))

        rand = np.random.normal(0,.001, 300)

        for i in range(self.vocab_size):
            word = self.words[i]
            try:
                lookup = glove_model[word]
                embeddings[i,:] = lookup
            except:
                embeddings[i,:] = rand              
            

        del(glove_model)

        return embeddings
        

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        
    def create_test(self):
        '''creates the test data'''

        if self.test_split is None:
            assert False, "Test split was none?"
        elif self.test_split < 1:
            cut = int((1-self.test_split) * self.tensor.size)
        elif self.test_split > 1:
            cut = -int(self.test_split)


        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        if self.test_split > 0:
            xdata = self.tensor[cut:]        
            self.test_num_batches = int(xdata.size / (self.batch_size *
                                                       self.seq_length))
    
            xdata = xdata[:self.test_num_batches * self.batch_size * self.seq_length]
            ydata = np.copy(xdata)
    
            ydata[:-1] = xdata[1:]
            ydata[-1] = xdata[0]
            self.test_x = np.split(xdata.reshape(self.batch_size, -1), self.test_num_batches, 1)
            self.test_y = np.split(ydata.reshape(self.batch_size, -1), self.test_num_batches, 1)
        

    def create_batches(self):
        '''creates the training data'''

        if self.test_split is None:
            assert False, "Test split was none?"
        elif self.test_split < 1:
            cut = int((1-self.test_split) * self.tensor.size)
        elif self.test_split > 1:
            cut = -int(self.test_split)

        xdata = self.tensor[:cut]
        
        self.num_batches = int(xdata.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."
            
        
        xdata = xdata[:self.num_batches * self.batch_size * self.seq_length]
        ydata = np.copy(xdata)

        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
