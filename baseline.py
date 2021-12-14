'''
Date: 2021-12-14 13:55:02
LastEditors: yuhhong
LastEditTime: 2021-12-14 14:35:13

This is the baseline, which use all the words as features. We could compare the results from part A, B, C and D with this baseline. 
'''

import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

from util import TweetsData

# change paths if necessary
TRAIN_SET_PATH = './StanceDataset/train.csv'
TEST_SET_PATH = './StanceDataset/test.csv'
RESULTS_PATH = 'results.csv'
TARGET_LIST = ['Hillary Clinton', 'Climate Change is a Real Concern', 'Legalization of Abortion', 'Atheism', 'Feminist Movement']
STANCE_DICT = {'AGAINST': 0, 'NONE': 1, 'FAVOR': 2}



# Yuhui: I make it as a function for reusing in `explore_SVM_settings.ipynb`. 
# Also, we could adjust the settings for different targets seperately (shown in the __main__). 
def per_SVM(data_train, data_test, clf, target): 
    print(">>> {}".format(target))

    X_train, Y_train = data_train.get_data_of_target(target) # X = 'CleanTweet', Y = 'Stance'
    X_test, Y_test = data_test.get_data_of_target(target)

    # encode X and Y
    # Yuhui: I put them outside the class TweetsData, because the training data 
    # and test data need to be encoded together. 
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



if __name__ == "__main__":
    ### 1: Read in train.csv and test.csv. 
    # 'latin1' resolves UniCode decode error
    df_train = pd.read_csv(TRAIN_SET_PATH, engine='python', dtype='str', encoding ='latin1') 
    df_test = pd.read_csv(TEST_SET_PATH, engine='python', dtype='str', encoding ='latin1')
    # init the outputs
    if os.path.exists(RESULTS_PATH): 
        df_res = pd.read_csv(RESULTS_PATH, sep='\t')
        print("Load the previous results from {}".format(RESULTS_PATH))
        print(df_res)
    else:
        df_res = pd.DataFrame(TARGET_LIST, columns=['Target'])



    ### 2: Preprocess on data (details in `a_1_2_util.py`). 
    ### 3: Extract a bag-of-words list of nouns, adj, and verbs from original Tweets.
    data_train = TweetsData(df_train, mode='AllWords') # init a TweetsData
    print("Load {} training data from {}".format(len(data_train), TRAIN_SET_PATH))
    data_test = TweetsData(df_test, mode='AllWords') # init a TweetsData
    print("Load {} test data from {}\n".format(len(data_test), TEST_SET_PATH))
    # print("Targets in train: {}".format(data_train.get_targets())) 
    # print("Targets in test: {}".format(data_test.get_targets()))



    ### 4: Perform SVM for different targets individually. 
    print("Default SVM ==============>> \n")
    results = []
    clf = svm.SVC(decision_function_shape='ovr')
    for target in TARGET_LIST: 
        results.append(per_SVM(data_train, data_test, clf, target))



    ### 5: Improve the results when optimize the settings. 
    # The details of exploring the setting are in `explore_SVM_settings.ipynb`.  
    print("Optimized SVM ==============>> \n")
    opt_results = []
    clf = svm.SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced')
    opt_results.append(per_SVM(data_train, data_test, clf, target='Hillary Clinton'))

    clf = svm.SVC(C=100, kernel='rbf', gamma='scale', class_weight='balanced')
    opt_results.append(per_SVM(data_train, data_test, clf, target='Climate Change is a Real Concern'))

    clf = svm.SVC(C=100, kernel='rbf', gamma='scale', class_weight='balanced')
    opt_results.append(per_SVM(data_train, data_test, clf, target='Legalization of Abortion'))

    clf = svm.SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced')
    opt_results.append(per_SVM(data_train, data_test, clf, target='Atheism'))

    clf = svm.SVC(C=0.1, kernel='rbf', gamma='scale', class_weight='balanced')
    opt_results.append(per_SVM(data_train, data_test, clf, target='Feminist Movement'))



    ### 6: Save the results to a file. 
    # add new columns for this experiment
    df_res['Default_AllWords'] = results
    df_res['Optimized_AllWords'] = opt_results
    print(df_res)
    df_res.to_csv(RESULTS_PATH, sep='\t', index=False)
    print("Save the results into {}".format(RESULTS_PATH))
