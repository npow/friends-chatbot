from __future__ import division
import numpy as np
import pandas as pd
import sys
import nltk
import pyprind
from nltk.corpus import wordnet as wn
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity 

SYNSETS = joblib.load('blobs/SYNSETS.pkl')
TAGS_HASH = joblib.load('blobs/TAGS_HASH.pkl')
data = pd.read_csv('data/friends-final.txt', sep='\t')
triturns = joblib.load('blobs/triturns.pkl')
filtered_triturns = joblib.load('blobs/filtered.pkl')
all_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
       'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB',
       'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN',
       'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

def get_synsets(text):
    if not text in SYNSETS:
        sent = nltk.pos_tag(nltk.word_tokenize(text))
        chunks = nltk.ne_chunk(sent, binary=False)
        s = set()
        def add_synsets(synsets):
            for synset in synsets:
                s.add(synset)
        #print "SENT: ", text
        for c in chunks:
            if hasattr(c, 'node'):
                if c.node == 'PERSON':
                    add_synsets(wn.synsets('person', pos=wn.NOUN))
                elif c.node == 'ORGANIZATION':
                    add_synsets(wn.synsets('organization', pos=wn.NOUN))                
                elif c.node == 'GPE':
                    add_synsets(wn.synsets('place', pos=wn.NOUN))
                elif c.node == 'LOCATION':
                    add_synsets(wn.synsets('location', pos=wn.NOUN))
                elif c.node == 'FACILITY':
                    add_synsets(wn.synsets('facility', pos=wn.NOUN))
                elif c.node == 'GSP':
                    add_synsets(wn.synsets('group', pos=wn.NOUN))                
                else:
                    print c, c.node, c.leaves()
            elif c[1][:2] in ['VB', 'JJ', 'ADV', 'NN']:
                pos = {'VB': wn.VERB, 'NN': wn.NOUN, 'ADV': wn.ADV, 'JJ': wn.ADJ}[c[1][:2]]
                add_synsets(wn.synsets(c[0], pos=pos))
            else:
                add_synsets(wn.synsets(c[0]))
        SYNSETS[text] = set([x.name for x in s])
    return SYNSETS[text]

def sem_sim(s1, s2):
    ss1 = get_synsets(s1)
    ss2 = get_synsets(s2)
    if ss1 == ss2:
        return 1
    return 2*len(ss1.intersection(ss2)) / (len(ss1) + len(ss2))

def cos_sim(s1, s2):
    d = [{}, {}]
    for p in all_tags:
        d[0][p] = d[1][p] = 0
    for i,s in enumerate([s1, s2]):
        if not s in TAGS_HASH:
            TAGS_HASH[s] = nltk.pos_tag(nltk.word_tokenize(s))
        tags = TAGS_HASH[s]
        for t in tags:
            if t[1] in d[i]:
                d[i][t[1]] += 1
    return cosine_similarity([d[0][p] for p in all_tags], [d[1][p] for p in all_tags])[0][0]

def sim(s1, s2, alpha=0.7):
    return alpha*sem_sim(s1, s2) + (1-alpha)*cos_sim(s1, s2)

def get_response(msg):
    best_val = 0
    best = None
    bar = pyprind.ProgBar(len(filtered_triturns), monitor=True)
    for t in filtered_triturns:
        question = t[0]
        answer = t[1]
        val = sim(msg, question)
        if (val > best_val):# or (val == best_val and len(answer) < len(msg))):
            best = answer
            best_val = val
        bar.update()
    return best

def main():
    """
    filtered_triturns = filter_triturns(0.8)
    print len(filtered_triturns)
    joblib.dump(filtered_triturns, 'blobs/filtered.pkl')
    return
    """
    while True:
        msg = raw_input('--> ')
        print get_response(msg)
        sys.stdout.flush()

def filter_triturns(thresh=0.7):
    L = []
    bar = pyprind.ProgBar(len(triturns), monitor=True)
    for i,tt in enumerate(triturns):
        a = data.irow(tt)['line']
        b = data.irow(tt+1)['line']
        c = data.irow(tt+2)['line']
        if sem_sim(a, b) > thresh:
            L.append([a,b])
        if sem_sim(b, c) > thresh:
            L.append([c,b])
        bar.update()
    return L

if __name__ == '__main__':
    main()
