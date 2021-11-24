'''

L545/B659 Fall 2021
Final Project
Holly Redman <hredman@iu.edu>
Lily Kawaoto <lkawaoto@iu.edu>
Yuhui Hong <yuhhong@iu.edu>

Part b.1: 
    Then extend your data set to include features using the MPQA Subjectivity lexicon 
    (http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/). Decide on a good way of using this 
    information in features. Explain your reasoning. How do the results change?

Part b.2: 
    Can you use the Arguing Lexicon (http://mpqa.cs.pitt.edu/lexicons/arg_lexicon/)? Do you
    find occurrences of the listed expressions? How do you convert the information into features? 
    Howdo these features affect classiffcation results?
    
'''

import pandas as pd
import numpy as np
import pprint
import treetaggerwrapper

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

from util import TweetsData

# change paths if necessary
TRAIN_SET_PATH = './StanceDataset/train.csv'
TEST_SET_PATH = './StanceDataset/test.csv'
LEXICON = './subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
TARGET_LIST = ['Hillary Clinton', 'Climate Change is a Real Concern', 'Legalization of Abortion', 'Atheism', 'Feminist Movement']
STANCE_DICT = {'AGAINST': 0, 'NONE': 1, 'FAVOR': 2}



def per_SVM(data_train, data_test, clf, target): 
    print(">>> {}".format(target))

    # Lily: we may have to add more code later to specify whether to use X_[train|test] or X2_[train|test].
    # For example, Parts A & B depend on the Noun/Adj/Verb bag-of-words vocabulary, but not Part C.
    # X_train, Y_train = data_train.get_data_of_target_bow(target) # X2 = 'BOW', Y = 'Stance'
    # X_test, Y_test = data_test.get_data_of_target_bow(target) 
    X_train, Y_train = data_train.get_data_of_target(target) # X = 'CleanTweet', Y = 'Stance'
    X_train_cnt_lexicon = data_train.get_cnt_lexicon_of_target(target)

    X_test, Y_test = data_test.get_data_of_target(target)
    X_test_cnt_lexicon = data_test.get_cnt_lexicon_of_target(target)


    # encode X and Y
    # Yuhui: I put them outside the class TweetsData, because the training data 
    # and test data need to be encoded together. 
    split_flg = len(X_train) # split training and test data later
    vectorizer = CountVectorizer()           
    X = vectorizer.fit_transform(X_train + X_test).toarray()      
    X_train = X[:split_flg]
    X_test = X[split_flg:]
    # add lexicons
    X_train_cnt_lexicon = np.array(X_train_cnt_lexicon)[:, np.newaxis]
    X_train = np.append(X_train, X_train_cnt_lexicon, axis=1)
    X_test_cnt_lexicon = np.array(X_test_cnt_lexicon)[:, np.newaxis]
    X_test = np.append(X_test, X_test_cnt_lexicon, axis=1)

    Y_train = np.array([STANCE_DICT[s] for s in Y_train])
    Y_test = np.array([STANCE_DICT[s] for s in Y_test])
    print("X_train: {}, Y_train: {}".format(X_train.shape, Y_train.shape))
    print("X_test: {}, Y_test: {}".format(X_test.shape, Y_test.shape))

    # train
    print("Training the SVM...")
    clf.fit(X_train, Y_train)
    print("Done!")

    # test
    Y_pred = clf.predict(X_test)
    print("Accuracy score: {}\n".format(accuracy_score(Y_test, Y_pred)))



if __name__ == "__main__": 
    # 0. Open lexicon file, use priorpolarity scores to determine 
    # if words from tweets are positive, negative, or neither
    with open(LEXICON) as f: 
        lexicon = f.read().splitlines() 
    neg_lexicon = []
    pos_lexicon = []
    for item in lexicon:
        if 'priorpolarity=positive' in item:
            pos_lexicon.append(item.split()[2].split('=')[1])
        elif 'priorpolarity=negative' in item:
            neg_lexicon.append(item.split()[2].split('=')[1])
        # Yuhui: ['type=weaksubj', 'len=1', 'word1=impassive', 'pos1=adj', 'stemmed1=n', 'polarity=negative', 'priorpolarity=weakneg'] is belong to the negative?
        elif 'polarity=positive' in item:
            pos_lexicon.append(item.split()[2].split('=')[1])
        elif 'polarity=negative' in item:
            neg_lexicon.append(item.split()[2].split('=')[1])



    ### 1: Read in train.csv and test.csv. 
    # 'latin1' resolves UniCode decode error
    df_train = pd.read_csv(TRAIN_SET_PATH, engine='python', dtype='str', encoding ='latin1') 
    df_test = pd.read_csv(TEST_SET_PATH, engine='python', dtype='str', encoding ='latin1')



    ### 2: Preprocess on data (details in `a_1_2_util.py`).
    ### 3: Extract a bag-of-words list of nouns, adj, and verbs from original Tweets.
    data_train = TweetsData(df_train, pos_lexicon, neg_lexicon) # init a TweetsData
    print("Load {} training data from {}".format(len(data_train), TRAIN_SET_PATH))
    data_test = TweetsData(df_test, pos_lexicon, neg_lexicon) # init a TweetsData
    print("Load {} test data from {}\n".format(len(data_test), TEST_SET_PATH))
    # print("Targets in train: {}".format(data_train.get_targets())) 
    # print("Targets in test: {}".format(data_test.get_targets()))  



    ### 4: Perform SVM for different targets individually. 
    print("Default SVM:\n")
    clf = svm.SVC(decision_function_shape='ovr')
    for target in TARGET_LIST: 
        per_SVM(data_train, data_test, clf, target)

