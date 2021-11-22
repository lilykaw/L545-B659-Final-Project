'''

L545/B659 Fall 2021
Final Project
Holly Redman <hredman@iu.edu>
Lily Kawaoto <lkawaoto@iu.edu>
Yuhui Hong <yuhhong@iu.edu>

Part a.1:
    "Extract a bag-of-words list of nouns, adjectives, and verbs for all targets individually. 
    Then create feature vectors for all training and test data (separately) for all targets."

Part a.2: 
    "Perform classification using Support Vector Machines (SVM) and default settings."

'''
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

from a_1_2_util import TweetsData

# change paths if necessary
TRAIN_SET_PATH = './StanceDataset/train.csv'
TEST_SET_PATH = './StanceDataset/test.csv'
TARGET_LIST = ['Hillary Clinton', 'Climate Change is a Real Concern', 'Legalization of Abortion', 'Atheism', 'Feminist Movement']
# Yuhui: Except 'Donald Trump' temporarily, we may need to disscuss how to process this target. 
# Targets in train: ['Hillary Clinton', 'Atheism', 'Legalization of Abortion', 
#                       'Climate Change is a Real Concern', 'Feminist Movement']
# Targets in test: ['Donald Trump', 'Hillary Clinton', 'Atheism', 'Climate Change is a Real Concern', 
#                          'Feminist Movement', 'Legalization of Abortion']
STANCE_DICT = {'AGAINST': 0, 'NONE': 1, 'FAVOR': 2}


### 1: Read in train.csv and test.csv. 
# 'latin1' resolves UniCode decode error
df_train = pd.read_csv(TRAIN_SET_PATH, engine='python', dtype='str', encoding ='latin1') 
df_test = pd.read_csv(TEST_SET_PATH, engine='python', dtype='str', encoding ='latin1')



### 2: Preprocess on data (details in `a_1_2_util.py`).
### 3: Extract a bag-of-words list of nouns, adj, and verbs from original Tweets.
data_train = TweetsData(df_train) # init a TweetsData
print("Load {} training data from {}".format(len(data_train), TRAIN_SET_PATH))
data_test = TweetsData(df_test) # init a TweetsData
print("Load {} test data from {}\n".format(len(data_test), TEST_SET_PATH))
# print("Targets in train: {}".format(data_train.get_targets()))
# print("Targets in test: {}".format(data_test.get_targets()))   


### 4: Perform SVM for different targets individually. 
for target in TARGET_LIST:
    # load data
    print(">>> {}".format(target))

    # Lily: we may have to add more code later to specify whether to use X_[train|test] or X2_[train|test].
    # For example, Parts A & B depend on the Noun/Adj/Verb bag-of-words vocabulary, but not Part C.
    X2_train, Y_train = data_train.get_data_of_target2(target) # X2 = 'BOW', Y = 'Stance'
    X2_test, Y_test = data_test.get_data_of_target2(target)
    # X_train, Y_train = data_train.get_data_of_target(target) # X = 'CleanTweet', Y = 'Stance'
    # X_test, Y_test = data_test.get_data_of_target(target)

    # encode X and Y
    # Yuhui: I put them outside the class TweetsData, because the training data 
    # and test data need to be encoded together. 
    split_flg = len(X2_train) # split training and test data later
    vectorizer = CountVectorizer()           
    X = vectorizer.fit_transform(X2_train + X2_test).toarray()      
    X2_train = X[:split_flg]
    X2_test = X[split_flg:]

    Y_train = np.array([STANCE_DICT[s] for s in Y_train])
    Y_test = np.array([STANCE_DICT[s] for s in Y_test])
    print("X2_train: {}, Y_train: {}".format(X2_train.shape, Y_train.shape))
    print("X2_test: {}, Y_test: {}".format(X2_test.shape, Y_test.shape))

    # train
    print("Training the SVM...")
    # clf = svm.SVC(decision_function_shape='ovo')
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(X2_train, Y_train)
    print("Done!")

    # test
    # print("Let's test the SVM!")
    Y_pred = clf.predict(X2_test)
    acc = accuracy_score(Y_test, Y_pred)
    print("Accuracy score: {}\n".format(acc))