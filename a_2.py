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



# dictionaries for label encoding and decoding
encode_dict = {'Hillary Clinton': 0, 'Climate Change is a Real Concern': 1, 'Legalization of Abortion': 2, 'Atheism': 3, 'Feminist Movement': 4, 'Donald Trump': 5}
decode_dict = {0: 'Hillary Clinton', 1: 'Climate Change is a Real Concern', 2: 'Legalization of Abortion', 3: 'Atheism', 4: 'Feminist Movement', 5: 'Donald Trump'}

# Load the labels and features
# Yuhui: There is no 'Donald Trump' in training data
df_train = pd.read_csv(TRAIN_SET_PATH, engine='python', dtype='str', encoding ='latin1')
df_test = pd.read_csv(TEST_SET_PATH, engine='python', dtype='str', encoding ='latin1')
Y_train = np.array([encode_dict[t] for t in df_train['Target']])
Y_test = np.array([encode_dict[t] for t in df_test['Target']])
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
# TO-DO: output the restuls