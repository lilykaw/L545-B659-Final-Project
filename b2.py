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
import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

from util import TweetsData



# change paths if necessary
TRAIN_SET_PATH = './StanceDataset/train.csv'
TEST_SET_PATH = './StanceDataset/test.csv'
SUB_LEXICON = './subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
ARG_LEXICON_DIR = './arglex_Somasundaran07'
MACRO_ARG_LEXICONS = ['modals.tff', 'spoken.tff', 'wordclasses.tff', 'pronoun.tff', 'intensifiers.tff']
TARGET_LIST = ['Hillary Clinton', 'Climate Change is a Real Concern', 'Legalization of Abortion', 'Atheism', 'Feminist Movement']
STANCE_DICT = {'AGAINST': 0, 'NONE': 1, 'FAVOR': 2}



def normalize_list(list_normal): 
    max_value = max(list_normal)
    min_value = min(list_normal)
    for i in range(len(list_normal)):
        list_normal[i] = (list_normal[i] - min_value) / (max_value - min_value)
    return list_normal 

def per_SVM(data_train, data_test, clf, target): 
    print(">>> {}".format(target))
    X_train, Y_train = data_train.get_data_of_target(target) # X = 'CleanTweet', Y = 'Stance'
    X_train_cnt_lexicon = normalize_list(data_train.get_cnt_arglexicon_of_target(target))
    X_test, Y_test = data_test.get_data_of_target(target)
    X_test_cnt_lexicon = normalize_list(data_test.get_cnt_arglexicon_of_target(target))

    # encode X, Y and add lexicons feature into X
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
    # 0. Open lexicon files, including 5 macro files and 17 files contain arguing lexicons
    macro_dict = {}
    arg_lexicons = []
    file_list = [f for f in os.listdir(ARG_LEXICON_DIR) if f.endswith('.tff')]
    for file_name in file_list:
        if file_name in MACRO_ARG_LEXICONS:
            # process the macro_dict
            with open(os.path.join(ARG_LEXICON_DIR, file_name), 'r') as f:
                data = f.read().splitlines() 
                for d in data:
                    matchg = re.match(r'@(.*)={(.*?)}', d) # match the pattern of @k={v}
                    if matchg:
                        k = '@' + matchg.group(1)
                        v = [w.strip().replace('\\', '') for w in matchg.group(2).split(',')]
                        macro_dict[k] = '|'.join(v) # make it a regular expression
        else:
            # process the argument lexicons
            with open(os.path.join(ARG_LEXICON_DIR, file_name), 'r') as f:
                data = f.read().splitlines()[1:] # regular expressions
                arg_lexicons.extend(data)
    # extand the arg_lexicons by macro_dict
    extend_arg_lexicons = []
    for lex in arg_lexicons:
        lex = lex.replace('\\', '') 
        pos_iter = re.finditer(r'@(\w)+', lex, flags=0)
        new_lex = lex
        for p in pos_iter: # replace the words with macro dictionary
            start_p = int(p.span()[0])
            end_p = int(p.span()[1])
            w = lex[start_p:end_p] # the positions are based on the 'lex', rather than the 'new_lex'
            new_lex = re.sub(w, macro_dict[w], new_lex)
        extend_arg_lexicons.append(new_lex)
    print("Generate {} argument lexicons!".format(len(extend_arg_lexicons)))

    
    ### 1: Read in train.csv and test.csv. 
    # 'latin1' resolves UniCode decode error
    df_train = pd.read_csv(TRAIN_SET_PATH, engine='python', dtype='str', encoding ='latin1') 
    df_test = pd.read_csv(TEST_SET_PATH, engine='python', dtype='str', encoding ='latin1')



    ### 2: Preprocess on data (details in `util.py`). 
    ### 3: Extract a bag-of-words list of nouns, adj, and verbs from original Tweets.
    data_train = TweetsData(df_train, mode='ArgLexicon', args=extend_arg_lexicons) # init a TweetsData
    print("Load {} training data from {}".format(len(data_train), TRAIN_SET_PATH))
    data_test = TweetsData(df_test, mode='ArgLexicon', args=extend_arg_lexicons) # init a TweetsData
    print("Load {} test data from {}\n".format(len(data_test), TEST_SET_PATH))
    # print("Targets in train: {}".format(data_train.get_targets())) 
    # print("Targets in test: {}".format(data_test.get_targets())) 



    ### 4: Perform SVM for different targets individually. 
    print("Default SVM:\n")
    clf = svm.SVC(decision_function_shape='ovr')
    for target in TARGET_LIST: 
        per_SVM(data_train, data_test, clf, target)
