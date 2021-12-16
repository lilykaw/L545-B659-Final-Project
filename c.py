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
from io import StringIO
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
PARSER_MAP = {'hillary': 'Hillary Clinton', 'climate': 'Climate Change is a Real Concern', 'abortion': 'Legalization of Abortion', 'atheism': 'Atheism', 'feminist': 'Feminist Movement'}
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

def gen_dep_triples(data): 
    # returns triples of a sentence
    #   [(word1, head1, label1), (word2, head2, label2), ...]
    data = StringIO(data)
    df = pd.read_csv(data, 
                        sep='\t', 
                        names=['ID', 'WORD', 'LEMMA', 'POS', 'V1', 'V2', 'HEAD_INDEX', 'DEP', 'V3', 'V4'])
    df = df.astype({"ID": int}) # Convert some strings to integers
    # print(df, '\n=====================\n')

    dep_triples = []
    for i in range(len(df)):
        word = df.iloc[i]['WORD']
        head_id = df.iloc[i]['HEAD_INDEX'] # it is the ID not the Index
        if word == '\t': # Yuhui: discard? 
            continue
        else:
            head_id = int(head_id) # Convert some strings to integers
        head = df[df["ID"] == head_id].iloc[0]['WORD'] if head_id != 0 else word
        label = "ROOT" if head==word else df.iloc[i]['DEP']
        dep_triples.append((word, head, label))
    return dep_triples



if __name__ == "__main__": 
    ### 0. Read in .conll files and get a list of dependency triples 
    print("Loading the parser results...")
    train_file_list = [f for f in os.listdir(MALTPARSER_CONLL_DIR) if f.endswith('_train_parsed.conllu')]
    test_file_list = [f for f in os.listdir(MALTPARSER_CONLL_DIR) if f.endswith('_test_parsed.conllu') and f != 'donald_test_parsed.conllu']
    # A. Preprocess the triples saved in train_dep_triples
    # B. Join them with df_train
    # train_dep_triples: {'<target1>': [<triples1>, <triples2>, ....], [<triples1>, <triples2>, ....], 
    #                       '<target2>': [<triples1>, <triples2>, ....], [<triples1>, <triples2>, ....],}
    train_dep_triples = {}
    for file in train_file_list: 
        with open(os.path.join(MALTPARSER_CONLL_DIR, file), 'r') as f:
            data = f.read().split('\n\n')
        target_name = PARSER_MAP[file.split('_')[0]]
        train_dep_triples[target_name] = [gen_dep_triples(d) for d in data if d != '']
        print("{}: {} sentences.".format(file, len(train_dep_triples[target_name])))
    
    test_dep_triples = {}
    for file in test_file_list: 
        with open(os.path.join(MALTPARSER_CONLL_DIR, file), 'r') as f:
            data = f.read().split('\n\n')
        target_name = PARSER_MAP[file.split('_')[0]]
        train_dep_triples[target_name] = [gen_dep_triples(d) for d in data if d != '']
        print("{}: {} sentences.".format(file, len(train_dep_triples[target_name])))
    print("Done!")



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
    data_train = TweetsData(df_train, mode='MaltParser', args=train_dep_triples) # init a TweetsData
    print("Load {} training data from {}".format(len(data_train), TRAIN_SET_PATH))
    data_test = TweetsData(df_test, mode='MaltParser', args=test_dep_triples) # init a TweetsData
    print("Load {} test data from {}\n".format(len(data_test), TEST_SET_PATH))
    # print("Targets in train: {}".format(data_train.get_targets())) 
    # print("Targets in test: {}".format(data_test.get_targets())) 



    ### 3: Perform SVM for different targets individually. 
    print("Default SVM:\n")
    results = []
    clf = svm.SVC(decision_function_shape='ovr')
    for target in TARGET_LIST: 
        results.append(per_SVM(data_train, data_test, clf, target))