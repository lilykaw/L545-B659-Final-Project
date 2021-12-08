'''

L545/B659 Fall 2021
Final Project
Holly Redman <hredman@iu.edu>
Lily Kawaoto <lkawaoto@iu.edu>
Yuhui Hong <yuhhong@iu.edu>

Part d: 
    - What happens if you use all words ONLY as features?
    - What happens when you use bi-grams along with unigrams as features?
    - What happens when you use uni-grams, bigrams and trigrams as features?
'''
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

from util import TweetsData

# change paths if necessary
TRAIN_SET_PATH = './StanceDataset/train.csv'
TEST_SET_PATH = './StanceDataset/test.csv'
RESULTS_PATH = 'results.csv'
TARGET_LIST = ['Hillary Clinton', 'Climate Change is a Real Concern', 'Legalization of Abortion', 'Atheism', 'Feminist Movement']
# Yuhui: Except 'Donald Trump' temporarily, we may need to disscuss how to process this target. 
# Targets in train: ['Hillary Clinton', 'Atheism', 'Legalization of Abortion', 
#                       'Climate Change is a Real Concern', 'Feminist Movement']
# Targets in test: ['Donald Trump', 'Hillary Clinton', 'Atheism', 'Climate Change is a Real Concern', 
#                          'Feminist Movement', 'Legalization of Abortion']
STANCE_DICT = {'AGAINST': 0, 'NONE': 1, 'FAVOR': 2}



# select the unigram, bigram or trigram by TF-IDF
def make_vocab(dataset, ngram_range, n=100): 
    # calculate the TF-IDF 
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, use_idf=True)
    tfIdf = vectorizer.fit_transform(dataset)
    df = pd.DataFrame(tfIdf[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    
    # choes the top-n unigram, bigram or trigram, and make a vocabulary
    vocab = {w: idx for idx, w in enumerate(df.index[:n])}
    print("Choose top-{} of {} unigrams/bigrams/trigrams as a vocabulary.".format(n, len(df)))
    return vocab

# Yuhui: I make it as a function for reusing in `explore_SVM_settings.ipynb`. 
# Also, we could adjust the settings for different targets seperately (shown in the __main__). 
def per_SVM(data_train, data_test, clf, target, ngram_range=(1, 1), n=100): 
    print(">>> {}".format(target))

    X_train, Y_train = data_train.get_data_of_target(target) # X2 = 'BOW', Y = 'Stance'
    X_test, Y_test = data_test.get_data_of_target(target)

    # encode X and Y
    split_flg = len(X_train) # split training and test data later
    # make a vocabulary by TF-IDF
    vocab = make_vocab(X_train + X_test, ngram_range=ngram_range, n=n)
    vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=ngram_range) 
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
    # There is a parameter in CountVectorizer, which can control the n-gram features we extract. 
    # For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, 
    # and (2, 2) means only bigrams. 
    print("Default SVM with unigrams:\n")
    uni_results = []
    clf = svm.SVC(decision_function_shape='ovr')
    for target in TARGET_LIST: 
        uni_results.append(per_SVM(data_train, data_test, clf, target, ngram_range=(1, 1), n=2000))

    print("Default SVM with unigrams and bigrams:\n")
    bi_results = []
    clf = svm.SVC(decision_function_shape='ovr')
    for target in TARGET_LIST: 
        bi_results.append(per_SVM(data_train, data_test, clf, target, ngram_range=(1, 2), n=4000))

    print("Default SVM with unigrams bigrams and trigrams:\n")
    tri_results = []
    clf = svm.SVC(decision_function_shape='ovr')
    for target in TARGET_LIST: 
        tri_results.append(per_SVM(data_train, data_test, clf, target, ngram_range=(1, 3), n=8000))



    ### 5: Save the results to a file. 
    # add new columns for this experiment
    df_res['Default_Unigrams'] = uni_results
    df_res['Default_Uni-Bigrams'] = bi_results
    df_res['Default_Uni-Bi-Trigrams'] = tri_results
    df_res.to_csv(RESULTS_PATH, sep='\t', index=False)
    print("Save the results into {}".format(RESULTS_PATH))