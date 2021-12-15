'''

L545/B659 Fall 2021
Final Project
Holly Redman <hredman@iu.edu>
Lily Kawaoto <lkawaoto@iu.edu>
Yuhui Hong <yuhhong@iu.edu>

Part c: Parse your training and test data using MALTparser and the predefined model. 
Then extract dependency triples form the data (word, head, label) 
and use those as features for the stance detection task instead of the bag-of-words model. 
How does that affect the results?

'''


import os
from nltk.util import trigrams
import pandas as pd
import numpy as np
import pprint

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from util import TweetsData

TRAIN_SET_PATH = './StanceDataset/train.csv'
TEST_SET_PATH = './StanceDataset/test.csv'
RESULTS_PATH = 'results.csv'
MALTPARSER_CONLL_DIR = './tweet_conll/MaltParser_Output'
TARGET_LIST = ['Hillary Clinton', 'Climate Change is a Real Concern', 'Legalization of Abortion', 'Atheism', 'Feminist Movement']
STANCE_DICT = {'AGAINST': 0, 'NONE': 1, 'FAVOR': 2}



def per_SVM(data_train, data_test, clf, target): 
    print(">>> {}".format(target))
    X_train, Y_train = data_train.get_data_of_target_parser(target) # X = 'MaltParser', Y = 'Stance'
    X_test, Y_test = data_test.get_data_of_target_parser(target)

    # encode X, Y and add lexicons feature into X
    split_flg = len(X_train) # split training and test data later
    vectorizer = CountVectorizer()           
    X = vectorizer.fit_transform(X_train + X_test).toarray()    
    X_train = X[:split_flg]
    X_test = X[split_flg:]

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
    acc = accuracy_score(Y_test, Y_pred)
    print("Accuracy score: {}\n".format(acc))
    return acc

def gen_dep_triples(file):
    # sep='\t' : .conll files are tab-separated
    # skip_blank_lines=False : to help with indexation after a new sentence
    # triplet format: (word, head, label)
    # returns a list of triples
    dep_triples = []
    df = pd.read_csv(file, sep='\t', 
                        names=['ID', 'WORD', 'LEMMA', 'POS', 'V1', 'V2', 'HEAD_INDEX', 'DEP', 'V3', 'V4'], 
                        skip_blank_lines=False)
        
    # extract triples
    new_sent_ind = 0
    # tmp_sent = []
    for i in range(len(df)):
        word = df.iloc[i, df.columns.get_loc("WORD")]
        if isinstance(word, float) or i==(len(df)-1): # float nans (sent breaker) or last Tweet 
            # dep_triples.append(tmp_sent)
            # tmp_sent = []
            new_sent_ind = i+1
            continue
        head_local_ind = df.iloc[i, df.columns.get_loc("HEAD_INDEX")]
        if word == '\t': # quotes (' ' and " ") are read as '\t' -> columns get shifted 1 to the left
            word = df.iloc[i, df.columns.get_loc("LEMMA")]
            head_local_ind = int(df.iloc[i, df.columns.get_loc("V2")])-1
            head = df.iloc[i, df.columns.get_loc("LEMMA")] if head_local_ind==-1 else df.iloc[new_sent_ind+head_local_ind, df.columns.get_loc("WORD")]
            if head=='\t':
                head = df.iloc[new_sent_ind+head_local_ind, df.columns.get_loc("LEMMA")]
            label = "ROOT" if head==word else df.iloc[i, df.columns.get_loc("HEAD_INDEX")]
        else: 
            head_local_ind = int(head_local_ind)-1
            head = word if head_local_ind==-1 else df.iloc[new_sent_ind+head_local_ind, df.columns.get_loc("WORD")]
            if head=='\t':
                head = df.iloc[new_sent_ind+head_local_ind, df.columns.get_loc("LEMMA")]  
            label = "ROOT" if head==word else df.iloc[i]['DEP'] 
        # tmp_sent.append((word, head, label))
        dep_triples.append((word, head, label))
    return dep_triples



if __name__ == "__main__": 
    ### 0. Read in .conll files and get a list of dependency triples [for training & test data]
    train_file_list = [os.path.join(MALTPARSER_CONLL_DIR, f) for f in os.listdir(MALTPARSER_CONLL_DIR) if f.endswith('_train_parsed.conll')]
    train_dep_triples = [gen_dep_triples(file) for file in train_file_list]

    test_file_list = [os.path.join(MALTPARSER_CONLL_DIR, f) for f in os.listdir(MALTPARSER_CONLL_DIR) if f.endswith('_test_parsed.conll')]
    test_dep_triples = [gen_dep_triples(file) for file in train_file_list]


    ### 1: Read in train.csv and test.csv. 
    # 'latin1' resolves UniCode decode error
    df_train = pd.read_csv(TRAIN_SET_PATH, engine='python', dtype='str', encoding ='latin1') 
    df_test = pd.read_csv(TEST_SET_PATH, engine='python', dtype='str', encoding ='latin1')

    if os.path.exists(RESULTS_PATH): 
        df_res = pd.read_csv(RESULTS_PATH, sep='\t')
        print("Load the previous results from {}".format(RESULTS_PATH))
        print(df_res)
    else:
        df_res = pd.DataFrame(TARGET_LIST, columns=['Target'])

    ### 2: Preprocess on data (details in `util.py`). 
    data_train = TweetsData(df_train, mode='MaltParser', args=(train_dep_triples, TARGET_LIST)) # init a TweetsData
    print("Load {} training data from {}".format(len(data_train), TRAIN_SET_PATH))
    data_test = TweetsData(df_test, mode='MaltParser', args=(test_dep_triples, TARGET_LIST)) # init a TweetsData
    print("Load {} test data from {}\n".format(len(data_test), TEST_SET_PATH))
    # print("Targets in train: {}".format(data_train.get_targets())) 
    # print("Targets in test: {}".format(data_test.get_targets())) 


    ### 3: Perform SVM for different targets individually. 
    print("Default SVM:\n")
    results = []
    clf = svm.SVC(decision_function_shape='ovr')
    for target in TARGET_LIST: 
        results.append(per_SVM(data_train, data_test, clf, target))