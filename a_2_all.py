'''
Date: 2021-11-10 21:33:22
LastEditors: yuhhong
LastEditTime: 2021-11-18 22:28:36
'''
'''

L545/B659 Fall 2021
Final Project
Yuhui Hong <yuhhong@iu.edu>

Part a.2: 
    "Perform classification using Support Vector Machines (SVM) and default settings."

'''
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score



# change paths if necessary
TRAIN_SET_PATH = './StanceDataset/train.csv'
TEST_SET_PATH = './StanceDataset/test.csv'
TRAIN_FEATURE_PATH = 'clean_train_vec.npy'
TEST_FEATURE_PATH = 'clean_test_vec.npy'



# Load the labels and features
# Yuhui: There is no 'Donald Trump' in training data
df_train = pd.read_csv(TRAIN_SET_PATH, engine='python', dtype='str', encoding ='latin1')
df_test = pd.read_csv(TEST_SET_PATH, engine='python', dtype='str', encoding ='latin1')

# generate a dictionary for encoding labels
labels = set()
for t, s in zip(df_train['Target'], df_train['Sentiment']):
    labels.add(t+' > '+s)
for t, s in zip(df_test['Target'], df_test['Sentiment']):
    labels.add(t+' > '+s)
encode_dict = {}
for i, label in enumerate(list(labels)):
    encode_dict[label] = i
print("We will encode the labels as: \n{}\n".format(encode_dict))

Y_train = np.array([encode_dict[t+' > '+s] for t, s in zip(df_train['Target'], df_train['Sentiment'])])
Y_test = np.array([encode_dict[t+' > '+s] for t, s in zip(df_test['Target'], df_test['Sentiment'])])
print("Load {} targets of training data,\n\t{} features of test data.\n".format(Y_train.shape, Y_test.shape))

X_train = np.load(TRAIN_FEATURE_PATH)
X_test = np.load(TEST_FEATURE_PATH)
print("Load {} features of training data,\n\t{} features of test data.\n".format(X_train.shape, X_test.shape))



# Train
print("Training the SVM...")
# clf = svm.SVC(decision_function_shape='ovo')
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(X_train, Y_train)
print("Done!")



# Test
print("Let's test the SVM!")
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("Accuracy score: {}".format(acc))
# output the restuls
