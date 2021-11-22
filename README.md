<!--
 * @Date: 2021-11-09 11:26:11
 * @LastEditors: yuhhong
 * @LastEditTime: 2021-11-20 15:00:28
-->
# L545-B659-Final-Project

Indiana University, Bloomington

## Set Up

Please install the following packages: 

- python=3.5
- numpy
- pandas
- scikit-learn
- nltk
- treetaggerwrapper

Tips: 

1. NLTK is not suitable for Python 3.7.

2. Remember to add the path to treetagger to your environment variables. 


## Part A

- Extract a bag-of-words list of nouns, adjectives, and verbs for all targets individually. Then create feature vectors for all training and test data (separately) for all targets.
- Perform classification using Support Vector Machines (SVM) and default settings. `a_2_all.py` is the codes for classify both target and sentiment. `a_2.py` is the codes for only classify target. 
- Improve the results when optimize the settings. 

---

Note (11/21/21): After filtering Vocabulary to be only Nouns, Adjectives, and Verbs, accuracy scores went down slightly.
Results:

```
Load 2914 training data from ./StanceDataset/train.csv
Load 1956 test data from ./StanceDataset/test.csv

>>> Hillary Clinton
X2_train: (689, 3127), Y_train: (689,)
X2_test: (295, 3127), Y_test: (295,)
Training the SVM...
Done!
Accuracy score: 0.6305084745762712

>>> Climate Change is a Real Concern
X2_train: (395, 2417), Y_train: (395,)
X2_test: (169, 2417), Y_test: (169,)
Training the SVM...
Done!
Accuracy score: 0.727810650887574

>>> Legalization of Abortion
X2_train: (653, 2836), Y_train: (653,)
X2_test: (280, 2836), Y_test: (280,)
Training the SVM...
Done!
Accuracy score: 0.6892857142857143

>>> Atheism
X2_train: (513, 2430), Y_train: (513,)
X2_test: (220, 2430), Y_test: (220,)
Training the SVM...
Done!
Accuracy score: 0.7227272727272728

>>> Feminist Movement
X2_train: (664, 3058), Y_train: (664,)
X2_test: (285, 3058), Y_test: (285,)
Training the SVM...
Done!
Accuracy score: 0.6210526315789474
```

---


Results: Vocabulary uses all words in training data

```
Load 2914 training data from ./StanceDataset/train.csv
Load 1956 test data from ./StanceDataset/test.csv

>>> Hillary Clinton
X_train: (689, 3977), Y_train: (689,)
X_test: (295, 3977), Y_test: (295,)
Training the SVM...
Done!
Accuracy score: 0.6237288135593221

>>> Climate Change is a Real Concern
X_train: (395, 3061), Y_train: (395,)
X_test: (169, 3061), Y_test: (169,)
Training the SVM...
Done!
Accuracy score: 0.727810650887574

>>> Legalization of Abortion
X_train: (653, 3694), Y_train: (653,)
X_test: (280, 3694), Y_test: (280,)
Training the SVM...
Done!
Accuracy score: 0.6928571428571428

>>> Atheism
X_train: (513, 3316), Y_train: (513,)
X_test: (220, 3316), Y_test: (220,)
Training the SVM...
Done!
Accuracy score: 0.7318181818181818

>>> Feminist Movement
X_train: (664, 3984), Y_train: (664,)
X_test: (285, 3984), Y_test: (285,)
Training the SVM...
Done!
Accuracy score: 0.6421052631578947
```

---

Results: 

```
Load (2914,) targets of training data,
        (1956,) features of test data.

Load (2914, 13459) features of training data,
        (1956, 13459) features of test data.

Training the SVM...
Done!
Let's test the SVM!
Accuracy score: 0.40081799591002043
```

Note (11/17/2021): After removing stopwords, accuracy improves to 0.42995910020449896. After that, removing numbers and pronouns did not affect the accuracy score at all. 
Removing usernames also did not affect the score.
```
Load (2914,) targets of training data,
	(1956,) features of test data.

Load (2914, 13442) features of training data,
	(1956, 13442) features of test data.

Training the SVM...
Done!
Let's test the SVM!
Accuracy score: 0.42995910020449896
```

Note (11/19/2021): After optimizing the parameters of SVM, accuracy improves to 0.44683026584867075. 
```
Load (2914,) targets of training data,
	(1956,) features of test data.

Load (2914, 13442) features of training data,
	(1956, 13442) features of test data.

Training the SVM...
Done!
Let's test the SVM!
Accuracy score: 0.44683026584867075
```
