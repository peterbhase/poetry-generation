import pickle
import random
import nltk
from nltk.tokenize import word_tokenize

all_strylines = pickle.load(open("output_plotsummaries.pkl", "rb"))
all_wordpools = pickle.load(open("all_wordpool.pkl", "rb"))

def stem_and_change(this_stryline):
    changed = []
    for word in this_stryline:
        changed.append(word_tokenize(word))
    return changed

def searching(keyword, strylines, wordpools):
    ans = []
    keyword_stemmed = word_tokenize(keyword)
    for i in range(0, len(strylines)):
        mine = stem_and_change(strylines[i])
        if keyword in strylines[i]:
            ans.append(i)
    if ans != []:
        random_pool = random.choice(ans)
        print (strylines[random_pool])
        return (wordpools[random_pool])
    else:
        return (random.choice(wordpools))

print (searching('wave', all_strylines, all_wordpools))