<!--
 * @Date: 2021-11-09 11:26:11
 * @LastEditors: yuhhong
 * @LastEditTime: 2021-11-10 20:52:51
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
- Perform classification using Support Vector Machines (SVM) and default settings. 

Results: 

```
Load (2914,) targets of training data,
        (1956,) features of test data.

Load (2914, 13459) features of training data,
        (1956, 13459) features of test data.

Training the SVM...
Let's test the SVM!
Accuracy score: 0.40081799591002043
```

Note (11/17/2021): After removing stopwords, accuracy improves to 0.42995910020449896
