# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:31:30 2017

@author: ftanada IA369Y
"""

import nltk
import nltk.classify.util
import csv
import os

print ('IA369Y - Computação Afetiva - Sentiment Analysis - Fabio Tanada')
print('Local directory:',os.getcwd())

#print('Downloading nltk')
#nltk.download()
from nltk.tokenize import word_tokenize
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')

print('\nProcessing input file')
headlines = []
headTokens = []
publishers = []
words = []
iCounter = 0;
with open('manchetesBrasildatabase.csv', 'r', encoding="utf-8",newline='') as csvfile:
   fieldNames = ['day', 'month','year','publisher','headline']
   spamreader = csv.DictReader(csvfile,fieldnames=fieldNames)
#   spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
   for row in spamreader:
#       print(row)
       iCounter = iCounter +1;
       head = row['headline']
#       head = head.decode(encoding='cp860',errors='strict')
#       print(head)
       headlines.append(head)
       publishers.append(row['publisher'])
       tokens = nltk.word_tokenize(str(head))
#       tokens = sent_tokenizer.tokenize(head)
       headTokens.append(tokens)
#       for word in tokens:
#           words.append(word)
#         print(', '.join(row))

print('Read',iCounter,'rows from manchetesBrasildatabase.csv')    
print('1st headline',headlines[0])
print('1st publisher',publishers[0])
print('Words:',headTokens[0])

print('\nProcessing nltk - removing stopwords and tagging')

from nltk.corpus import floresta
from nltk.corpus import mac_morpho

# stripping tagged words from floresta
def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t
twords = floresta.tagged_words()
twords = [(w.lower(), simplify_tag(t)) for (w,t) in twords]
#twords[:10]
#print(' '.join(word + '/' + tag for (word, tag) in twords[:10]))
 
# stopwords - filtering out
stopwords = nltk.corpus.stopwords.words('portuguese')
# adding some others
stopwords.append(',')
stopwords.append('.')
stopwords.append("'")

# tagging words from headlines
headTagged = []
iIndex = 0
for row in headTokens:
    tagged = []
#    print('row',row)
    for word in row:
        word = word.lower()
        if not (word in stopwords):
#           print('not stopword:',word)
           for key,tag in twords:
               if (key == word):
#                   print('tag',tag)
                   tagEntry = [key,tag]
                   tagged.append(tagEntry)
                   break
    headTagged.append(tagged)
    
print('Tagged:',headTagged[0])
  
print('\nProcessing sentiment over tagged')
# opening SentiLex-flex-PT
print('Reading SentiLex')
sentLex = []
polarity = []
with open('SentiLex-flex.csv', newline='') as csvfile2:
   fieldNames2 = ['word','word2','PoS','GN','TG','POL','ANOT']
   spamreader = csv.DictReader(csvfile2,fieldnames=fieldNames2)
   for row in spamreader:
#       print(row)
       sentLex.append(row['word'])
       polarity.append(int(row['POL']))

print('Reading nowns from Claudia Freitas')
sentLexNown = []
synClassNown = []
polarityNown = []
with open('LexicoClaudiaFreitas-n.csv', newline='') as csvfile2:
   fieldNames2 = ['word','word2','PoS','GN','TG','POL','ANOT']
   spamreader = csv.DictReader(csvfile2,fieldnames=fieldNames2)
   for row in spamreader:
#       print(row)
       sentLexNown.append(row['word'])
       synClassNown.append(row['PoS'])
       polarityNown.append(int(row['POL']))

print('Reading verbs from Claudia Freitas')
sentLexVerb = []
synClassVerb = []
polarityVerb = []
with open('LexicoClaudiaFreitas-v.csv', newline='') as csvfile3:
   fieldNames3 = ['word','word2','PoS','GN','TG','POL','ANOT']
   spamreader = csv.DictReader(csvfile3,fieldnames=fieldNames3)
   for row in spamreader:
#       print(row)
       sentLexVerb.append(row['word'])
       synClassVerb.append(row['PoS'])
       polarityVerb.append(int(row['POL']))

print('Evaluating sentiment by adj-nown-verb')
bias = []
defaultBias = 0
iBias = 0
stemmer = nltk.stem.RSLPStemmer()
for row in headTagged:
    bias.append(defaultBias)
    missing = True
    for key,tag in row:
        if (tag == 'adj'):
            # evaluate adj sentiment
            pol = 0
            i = 0
            missAdj = True
            for adj in sentLex:
                if (adj == key):
                    pol = polarity[i]
                    missing = False
                    missAdj = False
                    break
                i = i+1
            if (missAdj):
                print('Adjective not found in SentLex:',key)
#            print('Adj = ',key, 'bias =',pol)
            bias[iBias] = pol
    if (missing):
        # no adj in sentence, need analyze nowns
        for key,tag in row:
#            print(key,'=',tag)
            pol = 0
            if (tag == 'n'):
                # evaluate nown
                i = 0
                missNown = True
                for nownLex in sentLexNown:
                    if (key == nownLex):
                        pol = polarityNown[i]
                        missNown = False
                        break
                    i = i+1
                if (missNown):
                    print('Nown not found in Nown Lex:',key)
            else:
                if ((tag == 'v') or (tag == 'v-fin') or (tag == 'v-pcp') or 
                    (tag == 'v-inf')):
                    i = 0
                    missVerb = True
                    verb = stemmer.stem(key)
                    for verbLex in sentLexVerb:
                        if (verb == verbLex):
                            pol = polarityVerb[i]
                            missVerb = False
                            break
                        i = i+1
                    if (missVerb):
                        print('Verb not found in Verb Lex:',verb)
        bias[iBias] = pol
    iBias = iBias +1

# final report
print('\nFinal reporting')
i = 0
while (i < 10):
    print('Bias[0] = ',bias[i],'headline = ', headlines[i])
    i = i+1      
      
# generating outputfile
print('\nGenerating output report')
i = 0
with open('manchetesBrasildatabase - classified.csv', 'w',newline='') as csvfile:
   fieldNames = ['polarity','day', 'month','year','publisher','headline']
#   spamwriter = csv.DictWriter(csvfile,fieldnames=fieldNames)
   spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
   while (i < iCounter):
       spamwriter.writerow([bias[i],',',publishers[i],',',headlines[i]])
       i = i + 1
       
print('\nProgram finished')       
from nltk import ngrams, FreqDist
#from nltk.classify import NaiveBayesClassifier
