'''

L545/B659 Fall 2021
Final Project
Lily Kawaoto <lkawaoto@iu.edu>

Part a.1:
    "Extract a bag-of-words list of nouns, adjectives, and verbs for all targets individually. 
    Then create feature vectors for all training and test data (separately) for all targets."

'''

# ! pip install nltk
# ! pip install treetaggerwrapper
# ! pip install scikit-learn
# nltk.download('punkt')

import os, sys, nltk, re, string
# from nltk.corpus.reader import tagged
import pandas as pd
import numpy as np
import pprint, treetaggerwrapper
# import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# change paths if necessary
TRAIN_SET_PATH = './StanceDataset/train.csv'
TEST_SET_PATH = './StanceDataset/test.csv'
OUTPUT_FILE_TRAIN = 'clean_train_vec.npy'
OUTPUT_FILE_TEST = 'clean_test_vec.npy'



### 1: Read in train.csv. 
# @ train_tweets: list of Tweets from the csv file
train = pd.read_csv(TRAIN_SET_PATH, engine='python', dtype='str', encoding ='latin1') # 'latin1' resolves UniCode decode error
train_tweets = train['Tweet'].tolist()
print("Load {} training data from {}".format(len(train_tweets), TRAIN_SET_PATH))
# @ test_tweets: list of Tweets from the csv file
test = pd.read_csv(TEST_SET_PATH, engine='python', dtype='str', encoding ='latin1') # 'latin1' resolves UniCode decode error
test_tweets = test['Tweet'].tolist()
print("Load {} test data from {}\n".format(len(test_tweets), TEST_SET_PATH))


# Remove punctuations like '@', '#', '!'
remove_punctuation = string.punctuation + '‘’'
translator = str.maketrans('', '', remove_punctuation)
train_tweets = [sent.translate(translator) for sent in train_tweets]
print("Remove the punctuation in training tweets ({} sentences).".format(len(train_tweets)))
# pprint.pprint(train_tweets)

test_tweets = [sent.translate(translator) for sent in test_tweets]
# pprint.pprint(test_tweets)
print("Remove the punctuation in test tweets ({} sentences).\n".format(len(test_tweets)))

# Remove stopwords from test_tweets (holly)
stop_words = set(stopwords.words('english'))
word_tokens = [word_tokenize(sent) for sent in train_tweets]
filtered_tweets = []
for sent in word_tokens:
   filtered_sentence = []
   for w in sent:
      if w not in stop_words:
         filtered_sentence.append(w)
   sent = filtered_sentence
   filtered_tweets.append(sent)
filtered_tweets = [' '.join(t) for t in filtered_tweets]
#print(filtered_tweets)
train_tweets = filtered_tweets
#print(train_tweets)

# Remove stopwords from train_tweets (holly)
stop_words = set(stopwords.words('english'))
word_tokens = [word_tokenize(sent) for sent in test_tweets]
filtered_tweets = []
for sent in word_tokens:
   filtered_sentence = []
   for w in sent:
      if w not in stop_words:
         filtered_sentence.append(w)
   sent = filtered_sentence
   filtered_tweets.append(sent)
filtered_tweets = [' '.join(t) for t in filtered_tweets]
#print(filtered_tweets)
test_tweets = filtered_tweets
#print(test_tweets)

# Remove usernames in train_tweets (holly):
train_tweets = [re.sub('@[^\s]+', '', t) for t in train_tweets]

# Remove usernames in test_tweets (holly):
test_tweets = [re.sub('@[^\s]+', '', t) for t in test_tweets]



### 2: POS tagger on training Tweets
# See https://github.com/daimrod/lina-opinion-target-extractor/blob/master/treetaggerwrapper.py for where to put your TreeTagger directory
# @ tagged_train : list of sentences, each of which is a list of strings (words+POS+lemma)
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
# tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='/mnt/d/install/treetagger/') # Yuhui: I install the TreeTagger in a different directory
tagged_train = [] 
for t in train_tweets:
    tagged_train.append(tagger.tag_text(t))
# pprint.pprint(tagged_train[0])
print("Tag the training tweets ({} sentences) by TreeTagger".format(len(tagged_train)))

tagged_test = [tagger.tag_text(t) for t in test_tweets]
# pprint.pprint(tagged_test[0])
print("Tag the training tweets ({} sentences) by TreeTagger\n".format(len(tagged_test)))



### 3: Filter POS that are not nouns, adjectives, or verbs
# TreeTagger uses older abbreviations; see this document: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1603&context=cis_reports 
nav = ['NP', 'NPS', 'NN', 'NNS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# @ vocab : list of relevant words. Doesn't include POS tag. Includes misspellings and hashtag keywords (without spaces)
vocab = []
for sent in tagged_train:
    for w in sent: 
        temp = w.split('\t')
        if len(temp) == 1: # EX: line 252 'W'
            continue
        elif temp[1] in nav: 
            vocab.append(temp[0])
# print(vocab)

# @ vocab_test : list of relevant words. Doesn't include POS tag. Includes misspellings and hashtag keywords (without spaces)
vocab_test = []
for sent in tagged_test: 
    for w in sent: 
        temp = w.split('\t')
        if len(temp) == 1: # EX: line 252 'W'
            continue
        elif temp[1] in nav: 
            vocab_test.append(temp[0])
# print(vocab_test)



### 4: Create feature vector
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
# Yuhui: To make the feature length of each sentences the same, 
# we need to transform the training and test data together
corpus = train_tweets + test_tweets # 2914 training and 1956 test
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
X = X.toarray()
X_train = X[:2914, :]
np.save(OUTPUT_FILE_TRAIN, X_train)
print("Save the numpy array (shape: {}) of training data into {}".format(X_train.shape, OUTPUT_FILE_TRAIN))
X_test = X[2914:, :]
np.save(OUTPUT_FILE_TEST, X_test)
print("Save the numpy array (shape: {}) of training data into {}".format(X_test.shape, OUTPUT_FILE_TEST))



###################################################################
# Other things to consider for preprocessing:
# - remove just punctuation? Or, if it's '#', the entire hashtag?
# - remove misspellings and/or non-real words?
# - change all words to lowercase? If so, when?
