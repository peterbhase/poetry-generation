import numpy as np
import tensorflow as tf
import argparse
import os
from six.moves import cPickle
import nltk
import time
import queue as Q
from operator import itemgetter
from model import Model
import itertools
from numpy.random import choice
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
import string
import re
import collections
import sys
import pickle
import getopt

def isFitPattern(pattern,stress_num):
    if(len(pattern)+stress_num>10):
        return False
    i = stress_num
    ind = 0
    while(ind<len(pattern)):
        if(stress_num%2 == 0):
            if(pattern[ind]!="0"):
                return False
        else:
            if(pattern[ind]!="1"):
                return False
        ind+=1
        stress_num+=1
    return True

def createMeterGroups(corpus,dictMeters):
    ret = {}
    for word in corpus:
        if(word not in dictMeters):
            continue #THIS IS KEY STEP
        for pattern in dictMeters[word]: #THIS GENERATES REPEAT INSTANCES OF WORDS ter
            if(pattern not in ret):
                ret[pattern] = set([word])
            else:
                ret[pattern].add(word)
    return ret

def createPartSpeechTags(corpus,dictMeters):
    dictPartSpeechTags = {}
    for word in corpus:
        if(word not in dictMeters):
            continue
        token = nltk.word_tokenize(word)
        tag = nltk.pos_tag(token)
        dictPartSpeechTags[word] = tag[0][1]
    return dictPartSpeechTags

#DICTIONARY IS OF FORM (KEY: post_word_pos);(VALUE: pre_word_pos)
#SO dict["verb"] == set(adverb, noun, ...) BUT NOT set(adjective, determiner, etc)
def possiblePartsSpeechPaths():
    #SO dict["verb"] == set(adverb, noun, ...) BUT NOT set(adjective, determiner, etc)
    pos_list = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS", "LS","MD","NN","NNS","NNP","NNPS", \
                "PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","TO","UH","VB","VBD","VBG","VBN","VBP", \
                "VBZ","WDT","WP","WP$","WRB"]
    dictTags = {}
    for tag in pos_list:
        s = set([])
        if("VB" in tag):
            s = set(["CC","RB","RBR","RBS","NN","NN","NNS","NNP","NNPS","MD","PRP"])
            sing_nouns = set(["NN","NNP"])
            plur_nouns = set(["NNS","NNPS"])
            if(tag in set(["VB","VBG","VBP","VBN"])):
                s.difference(sing_nouns)
            if(tag in set(["VBG","VBZ","VBN"])):
                s.difference(plur_nouns)
            if(tag in set(["VBG","VBN"])):
                s.union(set(["VB","VBD","VBP","VBZ"]))
        else:
            s=set(pos_list)
            if("IN"==tag):
                t = set(["IN","DT","CC"]) #maybe not CC
                s.difference(t)
            if("JJ" in tag):
                t = set(["NN","NNS","NNP","NNPS"])
                s.difference(t)
            if("TO"==tag):
                t = set(["DT","CC","IN"])
                s.difference(t)
            if("CC"==tag):
                t = set(["DT","JJ","JJR","JJS"])
                s.difference(t)
            if("NN" in tag):
                t = set(["NN","NNS","NNP","NNPS","PRP","CC"]) #maybe not CC
                s.difference(t)
            if("MD"==tag):
                t = set(["DT","VB","VBD","VBG","VBN","VBP","VBZ"])
                s.difference(t)
            if("PRP"==tag):
                t = set(["CC","JJ","JJR","JJS","NN","NNS","NNP","NNPS","DT"])
                s.difference(t)
            if("PRP$"==tag):
                t = set(["CC","DT","VB","VBD","VBG","VBN","VBP","VBZ","PRP"])
                s.difference(t)
            adv = set(["RB","RBR","RBS"])
            if(tag not in adv):
                s.remove(tag)
        dictTags[tag] = s
    return dictTags
class State:
    def __init__(self, tup):
        self.coord = tup #form: (line,stress)
        self.nexts = set()
        self.prevs = set()

def formLinkedTree(stress,dictMeters,corpus,fsaLine,dictWordTransitions,dictCorpusMeterGroups):
    test = 0
    if(stress==10):
        return "base_case"
    for meter in dictCorpusMeterGroups:
        if(isFitPattern(meter,stress)):
            new_stress = stress+len(meter)
            if(new_stress > 10):
                continue
            recursion = formLinkedTree(new_stress,dictMeters,corpus,fsaLine,dictWordTransitions,dictCorpusMeterGroups)
            if(recursion=="no_children"):
                continue
            if(recursion!="base_case"):
                fsaLine = recursion[0]
                dictWordTransitions = recursion[1]
            dictWordTransitions[(stress,new_stress)]=dictCorpusMeterGroups[meter]
            fsaLine[stress].nexts.add(new_stress)
            fsaLine[new_stress].prevs.add(stress)
            test += 1
    if(test==0):
        return "no_children"
    return fsaLine,dictWordTransitions

def simple_clean(string):
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"-", "", string)
        string = re.sub(r":", "", string)

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'d", "ed", string) #for the old style
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"I\'ve", "I have", string)
        string = re.sub(r"\'ll", " will", string)

        string = re.sub(r"[0-9]+", "EOS", string) # EOS tag for numeric titling

        string = re.sub(r";", ",", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\.", " . ", string)

        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

def dynamic_rhyme(prompt, words, word2rhyme, rhyme2words, vocab):
    common = pickle.load(open("CommonRhymes.pkl","rb"))
    wors = [word for word in words if word in glove_model.vocab]
    # topic-representation is avg. of its words' reps
    prompt_words = prompt.split()
    prompt_rep = glove_model[prompt_words[0]]
    for word in prompt_words[1:]:
        prompt_rep = prompt_rep + glove_model[word]
    prompt_rep = prompt_rep / len(prompt_words)

    def cos_sim(w1,w2):
        return np.dot(w1,w2) / np.linalg.norm(w1) / np.linalg.norm(w2)

    #word2sim = dict(zip(wors, [glove_model.similarity(prompt, word) for word in wors])) # dict of distances
    word2sim = dict(zip(wors, [cos_sim(prompt_rep, glove_model[word]) for word in wors]))

    rhyme_used = set()
    rhyme_used.add("NOT_IN")
    pairs_picked = []
    for i in range(5): #repeat 5 times (get 5 topical rhyme pairs)
        pairs = [] # candidate pairs for this round
        for rhyme, w in rhyme2words.items():
            if rhyme not in rhyme_used: # check rhyme not used already
                pairs += list(itertools.combinations(w, 2))
        prob = [max([word2sim[x] for x in pair])**7 if all(word2sim[x] >0 for x in pair) else 0
               for pair in pairs]
        s = sum(prob)
        w = [x/s for x in prob] #  weights
        draw = choice(range(len(pairs)), 1, p=w)[0] # index
        pairs_picked.append(pairs[draw])
        rhyme_used.add(word2rhyme[pairs[draw][0]])
    # common rhyme pairs
    alphabet = set("abcdefghijklmnopqrstuvwxyz")
    pairs = [x[0] for x in common.most_common(50) if all(word not in alphabet for word in x[0])]
    for pair in pairs:
        if(pair[0] not in word2rhyme):
            word2rhyme[pair[0]] = "NOT_IN"
    pairs = [pair for pair in pairs if all(word in vocab for word in pair) and #train_dict
             word2rhyme[pair[0]] not in rhyme_used]
    common_2 = choice(len(pairs), 2)
    pairs_picked += [pairs[index] for index in common_2]
    return pairs_picked

def search(model, vocab, prob_sequence, sequence, post_stress, state, session, \
                temp, dictMeters, fsaLine, dictWordTransitions, dictPartSpeechTags, breadth, wordPool):
    def beamSearchOneLevel(model, vocab, prob_sequence, sequence, post_stress, state, session, \
                    temp, dictMeters, fsaLine, dictWordTransitions, dictPartSpeechTags, breadth, wordPool):
        def decayRepeat(word,sequence, scale):
            safe_repeat_words = []
            #safe_repeat_words = set(["with,the,of,in,i"])
            score_adjust = 0
            decr = -scale
            for w in range(len(sequence)):
                if(word==sequence[w] and word not in safe_repeat_words):
                    score_adjust += decr
                decr += scale/10 #decreases penalty as the words keep getting further from the new word
            return score_adjust
        def partsOfSpeechFilter(word1,word2,dictPartSpeechTags,dictPossiblePartsSpeech):
            okay_tags = set(["RB","RBR","RBS"]) #THESE ARE THE ADVERBS
            tag1 = dictPartSpeechTags[word1]
            tag2 = dictPartSpeechTags[word2]
            #if(tag1==tag2 and tag1 not in okay_tags):
            #    return True
            if(tag1 not in dictPossiblePartsSpeech[tag2]):
                return True
            else:
                return False
        if(post_stress==0):
            return("begin_line")
        ret = []
        scale = .02 #scale is the significant magnitude required to affect the score of bad/good things
        dist, state = model.compute_fx(session, vocab, prob_sequence, sequence, state, temp)
        for pred_stress in list(fsaLine[post_stress].prevs):
            word_set = set([])
            for word in dictWordTransitions[(pred_stress,post_stress)]:
                #PREVENTS REPEAT ADJACENT WORDS OR PROBLEM-TAGGED WORDS
                if(word == sequence[0]):
                    continue
                if(partsOfSpeechFilter(word,sequence[0],dictPartSpeechTags,dictPossiblePartsSpeech)):
                    continue
                #FACTORS IN SCORE ADJUSTMENTS
                score_adjust = decayRepeat(word, sequence, 100*scale) #repeats
                score_adjust += scale*len(word)/50 #length word
                if(word in wordPool):
                    score_adjust += scale
                #CALCULATES ACTUAL SCORE
                key = np.array([[vocab[word]]])
                new_prob = dist[key]
                score_tuple = (new_prob, state)
                score_tup = (score_tuple[0]+score_adjust,score_tuple[1]) #NOTE SHOULD SCORE_ADJUST BE ADDED HERE OR JUST IN THE ITEM LINE?
                item = (score_tup[0],(score_tup, [word]+sequence, pred_stress))
                if(item[0]==[[-float("inf")]]):
                    continue
                ret+=[item]
        return ret
    masterPQ = Q.PriorityQueue()
    checkList = []
    checkSet = set([])
    score_tuple = (prob_sequence, state)
    first = (score_tuple[0],(score_tuple, sequence, post_stress))
    masterPQ.put(first)#initial case
    set_explored = set([])
    while(not masterPQ.empty()):
        depthPQ = Q.PriorityQueue()
        while(not masterPQ.empty()):
            try:
                next_search = masterPQ.get()
            except:
                continue
            possible_branches = beamSearchOneLevel(model, vocab, next_search[1][0][0], next_search[1][1], next_search[1][2],\
                                next_search[1][0][1], session, temp, dictMeters, fsaLine, dictWordTransitions,\
                                dictPartSpeechTags, breadth, wordPool)
            if(possible_branches == "begin_line"):
                checkList+=[next_search]
                continue
            for branch in possible_branches:
                if(branch == []):
                	continue
                test = tuple(branch[1][1]) #need to make sure each phrase is being checked uniquely (want it to be checked once in possible branches then never again)
                if(test in set_explored):
                    continue
                set_explored.add(test)
                depthPQ.put(branch)
                try:
                    if(depthPQ.qsize()>breadth):
                        depthPQ.get()
                except:
                    pass
        masterPQ = depthPQ
    return checkList

def genPoem(save_dir, topic, width, wordPools):
    start = time.time()
    def sampleLine(lst, cut):
        ''' samples from top "cut" lines, the distribution being the softmax of true sentence probabilities'''
        probs = list()
        for i in range(cut):
            probs.append(np.exp(lst[i][0][0][0]))
        probs = np.exp(probs) / sum(np.exp(probs))
        index = np.random.choice(cut,1,p=probs)[0]
        return lst[index][1][1]
    def postProcess(poem, sess, vocab, model):
        '''takes a list of lists formatted poem and turns it into nice text'''
        ret = ""
        '''find comma locations here, ith line'''
        flat_poem = []
        for line in poem:
            for word in line:
                flat_poem.append(word)
        rev_flat_poem = [x for x in reversed(flat_poem)]
        n_spots = len(rev_flat_poem) - 1
        comma_probs = np.zeros(n_spots)
        state = sess.run(model.initial_state)
        p = np.zeros(len(vocab))
        for i in range(n_spots):
                seq = [y for y in reversed(rev_flat_poem[:i+1])] #so we're iterating from the rhyme word, but need to feed in forward
                p = np.zeros(len(vocab))
                p, state = model.compute_fx(sess, vocab, p, seq, state, 1)
                comma_probs[i] = p[vocab[","]]
        comma_probs = np.array([y for y in reversed(comma_probs)])
        num_commas = int(np.random.normal(9,2))
        num_commas = min(num_commas,14)
        num_commas = max(3,num_commas)
        cut_prob = [i for i in reversed(sorted(comma_probs))][num_commas]
        spots = np.argwhere(comma_probs > cut_prob)
        comma_counter = 0
        '''put some commas in there'''
        for i in range(14):
            for j in range(len(poem[i])):
                if(comma_counter in spots.squeeze()):
                #if( comma_counter in spots.squeeze() and (not i%3 or j=):
                    poem[i][j] = poem[i][j] + ","
                comma_counter = comma_counter + 1
        ''' capitalize and print'''
        for i in range(14):
            line = poem[i]
            for j in range(len(line)):
                if line[j] == "i":
                    line[j] = "I"
            line[0] = str.upper(line[0][0]) + line[0][1:]
            if(i == 3 or i == 7 or i == 11 or i == 13):
                if("," in line[-1]):
                    line[-1] = line[-1][:-1]
                ret += ' '.join(line) + "." + '\n'
            else:
                ret += ' '.join(line) + '\n'
        return ret
    tf.reset_default_graph()
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        word_keys, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            rhymes = dynamic_rhyme(topic, corpi, word2rhyme, rhyme2words, vocab)
            index = np.array([[0,2],
                              [1,3],
                              [4,6],
                              [5,7],
                              [8,10],
                              [9,11],
                              [12,13]]).reshape(14,1).squeeze()
            rhymes = np.array(rhymes).reshape(14,1).squeeze()
            rhymes = rhymes[index]
            poem = []
            line_num = 0
            wordPool_ind = 0
            for rhyme in rhymes:
                num_syll = len(dictMeters[rhyme][0])
                post_stress = 10-num_syll
                state = sess.run(model.initial_state)
                init_score = np.array([[0]])
                lst = search(model, vocab, init_score,[rhyme],post_stress, state, sess, 1,\
                                  dictMeters,fsa_row,dictWordTransitions,dictPartSpeechTags,width,wordPools[wordPool_ind])
                lst.sort(key=itemgetter(0), reverse = True)
                # line diagnostics
                #for i in range(10)
                #    print(lst[i][0][0][0], lst[i][1][1])
                choice = sampleLine(lst, min(10,len(lst)))
                poem.append(choice)
                line_num+=1
                if(line_num>3):
                    line_num = 0
                    wordPool_ind+=1
        poem = postProcess(poem,sess,vocab,model)
    print("Generation took {:.3f} seconds".format(time.time() - start))
    return poem

if(__name__ == "__main__"):

	# load glove to a gensim model
    glove_model = KeyedVectors.load_word2vec_format('./storyline_for_reference/glove.6B.300d.word2vec.txt',binary=False)
    
    # system arguments
    topic = sys.argv[1]
    try:
    	seed = int(sys.argv[2])
    except:
    	seed = 1
    
    np.random.seed(seed) # seed for reproducibility
   
   	# this is where data directory and model directory are determined
    text_list = ("data\whitman\input.txt","whitman_model")
    t = text_list[0] #THIS TEXT IS THE VOCAB!
    save_dir = text_list[1] #THIS IS THE MODEL DIRECTORY
    text = open(t)
    text = text.read()
    with open("./cmudict-0.7b.txt") as f:
        lines = [line.rstrip("\n").split() for line in f if (";;;" not in line)]
    dictMeters = {}
    for i in range(len(lines)):
        line = lines[i]
        newLine = [line[0].lower()]
        if("(" in newLine[0] and ")" in newLine[0]):
            newLine[0] = newLine[0][:-3]
        chars = ""
        for word in line[1:]:
            for ch in word:
	            if(ch in "012"):
	                if(ch == "2"):
	                    chars+="1"
	                else:
	                    chars+=ch
        newLine+=[chars]
        lines[i] = newLine
        if(newLine[0] not in dictMeters): #THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
            dictMeters[newLine[0]]=[chars]
        else:
            if(chars not in dictMeters[newLine[0]]):
                dictMeters[newLine[0]]+=[chars]
    words = [simple_clean(word) for word in text.split()]
    uniques = set()
    for word in words:
        if word not in dictMeters:
            continue
        else:
            uniques.add(word)
    corpus = list(uniques)
    dictCorpusMeterGroups = createMeterGroups(corpus,dictMeters)
    dictWordTransitions = {}
    fsaLine = [State((0,i)) for i in range(11)]
    dictMeters["i"]+=["0"]
    dictMeters["the"]+=["0"]
    glove_words = glove_model.vocab.keys() #SHOULDN'T THIS HAVE BEEN MOST COMMON WORDS IN THE CORPUS?
	# CORPUS LIMITED TO WORDS IN GLOVE VOCAB
    word_counts = collections.Counter([x for x in words if x in dictMeters])
    corpi = [x[0] for x in word_counts.most_common() if x[0] in glove_words]
    dictCorpusMeterGroups = createMeterGroups(corpi,dictMeters)
    dictPartSpeechTags = createPartSpeechTags(corpi,dictMeters)
    dictPossiblePartsSpeech = possiblePartsSpeechPaths()
    fsa_ret = formLinkedTree(0,dictMeters,corpi,fsaLine,dictWordTransitions,dictCorpusMeterGroups)
    fsa_row = fsa_ret[0]
    dictWordTransitions = fsa_ret[1]
    vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
    stressed_v = [v+"1" for v in vowels] + [v+"2" for v in vowels]
    dict_M = {}
    dict_S = {}
    dict_R = {}
    with open("./cmudict-0.7b.txt") as f:
        lines = [line.rstrip("\n").split() for line in f if (";;;" not in line)]
        for line in lines:
            word = line[0].lower()
	        # syllables
            dict_S[word] = [''.join(i for i in syl if not i.isdigit()) for syl in line[1:]]
	        # rhyme
            r_index = max([''.join(line[1:]).rfind(v) for v in stressed_v])
            dict_R[word] = ''.join(i for i in ''.join(line[1:])[r_index:] if not i.isdigit())
	        # meters
            meters = ""
            for syl in line[1:]:
                for ch in syl:
                    if(ch in "012"):
                        if(ch == "2"):
                            meters+="1"
                        else:
                            meters+=ch
            dict_M[word] = meters
    words = []
    for word, meter in dict_M.items():
        if meter == '1'*(len(meter)%2) + '01'*(len(meter)//2) and word in glove_model.vocab.keys() and word in corpi: #fits meter req & exists in GloVe vocab
            words += [word]
    word2syllable = dict((word, dict_S[word]) for word in words) # {meter-qualifying word : syllables}
    word2rhyme = dict((word, dict_R[word]) for word in words) # {meter-qualifying word : rhyme}
	#rhyme2words = defaultdict(lambda:[])
    rhyme2words = {}
    for word, rhyme in word2rhyme.items():
        if rhyme not in rhyme2words:
            rhyme2words[rhyme] = [word]
        else:
            rhyme2words[rhyme] += [word]
    width = 20
    wordPools = [set([]) for n in range(4)]
    poem = genPoem(save_dir,topic,width,wordPools)
    with open("./output_poems\%s-%i.txt"%(topic,poem_ind), "w") as text_file:
        print("(saved in output_poems)")
        print("\n")
        print(topic)
        print("\n")
        print(poem)
        text_file.write(topic+"\n")
        text_file.write("\n")
        text_file.write(poem)
