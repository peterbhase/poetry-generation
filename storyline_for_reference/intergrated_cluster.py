import nltk
import gensim
import re
import numpy as np
from nltk.corpus import state_union
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from rake_nltk import Rake
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#import RAKE
import json
import smart_open
import random
import logging
import pickle
import os, glob
from gensim.models.keyedvectors import KeyedVectors

def generation(cluster_matrix):
    # Alternative distance metric:
    # for each storyline-word, find most similar words;
    # filter for words that are reasonably close to other storyline-words
    allpool = []
    for cluster in cluster_matrix:
        pool = []
        try:
            for word in cluster:
                # find candidate words
                try:
                    thisword_tk = nltk.tokenize(word)
                    thistag_tk = nltk.pos_tag(thisword_tk)
                    if (thisword_tk[1] not in tagdict):
                        continue
                except:
                    continue
                cand =[tup[0] for tup in glove_model.most_similar(word, topn=200)]
                # calculate distances from candidate words to other storyline-words
                cand2clust_dists = np.sum([glove_model.distances(x,cand) for x in cluster if x!=word], axis=0)
                # indexes of qualified words in cand (comparing among themselves)
                indexes = cand2clust_dists.argsort()[:200] # get top 25
                keep = set()
                tagdict = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
                print (cand)
                smallest = len(cand)
                if smallest > 200:
                    smallest = 200
                for i in range(0, smallest):
                    try: 
                        word_tk = nltk.tokenize(cand[i])
                        tag_tk = nltk.pos_tag(word_tk)
                        if (tag_tk[1] in tagdict):
                            keep.add(cand[i])
                    except:
                        continue
                    if len(keep) == 25:
                        break
                # OR, comparing with all vocab
                # indexes of words whose total distance to other storyline-words is among top 1% of all vocab
                #top_dist = np.percentile(np.sum([glove_model.distances(x) for x in cluster if x!=word], axis=0),1)
                #keep = [cand[i] for i in range(len(cand2clust_dists)) if cand2clust_dists[i] <= top_dist]
                pool = pool + keep
                print (pool)
            allpool = allpool + pool
        except:
            print ("Sad!!")
            return None
    return allpool


glove_model = KeyedVectors.load_word2vec_format('./glove.6B.300d.word2vec.txt', binary=False)
# load trained word2vec model
word2vec_model = gensim.models.word2vec.Word2Vec.load('./word2vec_poetic.bin')

generated_clusters = pickle.load(open("../output_plotsummaries.pkl", "rb"))
list_dict = generated_clusters

list_wordpools = []

a_list = list_dict[0]
cluster_matrix = [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
count = -1
for i in range(0, 4):
    for j in range(0, 4):
        count = count + 1
        cluster_matrix[i][j] = a_list[count]
print (cluster_matrix)
thispool = generation(cluster_matrix)
if thispool != None:
    list_wordpools.append(thispool)

#for a_list in list_dict:
#    cluster_matrix = [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
#    count = -1
#    for i in range(0, 4):
#        for j in range(0, 4):
#            count = count + 1
#            cluster_matrix[i][j] = a_list[count]
#    thispool = generation(cluster_matrix)
#    if thispool != None:
#        list_wordpools.append(thispool)

#cluster_matrix = [
#    ['detective', 'feeling', 'partially', 'responsible'], ['lost', 'mountain', 'climbers', 'starts'], ['starts',
#     'questioning', 'everyone', 'seemingly'], ['briefly', 'considers', 'returning', 'meet']

#]




with open ('all_wordpool.pkl', "wb") as ff:
    pickle.dump(list_wordpools, ff)


