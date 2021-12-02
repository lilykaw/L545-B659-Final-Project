<!--
 * @Date: 2021-11-09 11:26:11
 * @LastEditors: yuhhong
 * @LastEditTime: 2021-12-02 12:31:07
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
- Improve the results when optimize the settings. In `explore_SVM_settings.ipynb`, the 'Hillary Clinton' is used as an example aiming to explore the affect of each parameter of SVM. Then we get a general step of adjust SVM settings and optimized them for different targets one by one in `a.py`. The general steps are: 
	- Check that which kernel performance best with their default related parameters. 
    In most of the situations, it is linear kernel or RBF kernel. If the dimension of feature is large enough compare to the number of samples, the data can be linearly seperatable in high dimensionality, the linear kernel will performance great and fast. If the dimension of feature is not large enough, the RBF kernel could be a good choice. 
	- Adjust the regularization parameter.
	- Check that whether a balanced class weight need to be used. 

---

Note (11/24/21): Based on the results of BOW, the SVM is optimized, and the performance improves. 

```
>>> Hillary Clinton
X_train: (689, 3127), Y_train: (689,)
X_test: (295, 3127), Y_test: (295,)
Training the SVM...
Done!
Accuracy score: 0.6338983050847458

>>> Climate Change is a Real Concern
X_train: (395, 2417), Y_train: (395,)
X_test: (169, 2417), Y_test: (169,)
Training the SVM...
Done!
Accuracy score: 0.7396449704142012

>>> Legalization of Abortion
X_train: (653, 2836), Y_train: (653,)
X_test: (280, 2836), Y_test: (280,)
Training the SVM...
Done!
Accuracy score: 0.6892857142857143

>>> Atheism
X_train: (513, 2430), Y_train: (513,)
X_test: (220, 2430), Y_test: (220,)
Training the SVM...
Done!
Accuracy score: 0.7227272727272728

>>> Feminist Movement
X_train: (664, 3058), Y_train: (664,)
X_test: (285, 3058), Y_test: (285,)
Training the SVM...
Done!
Accuracy score: 0.6421052631578947
```

---

Note (11/21/21): After filtering Vocabulary to be only Nouns, Adjectives, and Verbs, accuracy scores went down slightly.

```
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

Note (11/20/21) Vocabulary consists of all words.

```
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

## Part B

- Then extend your data set to include features using the [MPQA Subjectivity lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/). Decide on a good way of using this information in features. Explain your reasoning. How do the results change? 

- Can you use the [Arguing Lexicon](http://mpqa.cs.pitt.edu/lexicons/arg_lexicon/)? Do you find occurrences of the listed expressions? How do you convert the information into features? How do these features affect classiffcation results?

---

Note (12/02/2021): Add arguing lexicons for 17 classes seperately.

```
>>> Hillary Clinton
X_train: (689, 3994), Y_train: (689,)
X_test: (295, 3994), Y_test: (295,)
Training the SVM...
Done!
Accuracy score: 0.6271186440677966

>>> Climate Change is a Real Concern
X_train: (395, 3078), Y_train: (395,)
X_test: (169, 3078), Y_test: (169,)  
Training the SVM...
Done!
Accuracy score: 0.7218934911242604

>>> Legalization of Abortion
X_train: (653, 3711), Y_train: (653,)
X_test: (280, 3711), Y_test: (280,)
Training the SVM...
Done!
Accuracy score: 0.6928571428571428

>>> Atheism
X_train: (513, 3333), Y_train: (513,)
X_test: (220, 3333), Y_test: (220,)
Training the SVM...
Done!
Accuracy score: 0.7318181818181818

>>> Feminist Movement
X_train: (664, 4001), Y_train: (664,)
X_test: (285, 4001), Y_test: (285,)
Training the SVM...
Done!
Accuracy score: 0.6421052631578947
```

---

Note (11/30/2021): Add arguing lexicons, and normalizing the feature to $[0, 1]$. 

```
>>> Hillary Clinton
X_train: (689, 3978), Y_train: (689,)
X_test: (295, 3978), Y_test: (295,)
Training the SVM...
Done!
Accuracy score: 0.6237288135593221

>>> Climate Change is a Real Concern
X_train: (395, 3062), Y_train: (395,)
X_test: (169, 3062), Y_test: (169,)
Training the SVM...
Done!
Accuracy score: 0.7396449704142012

>>> Legalization of Abortion
X_train: (653, 3695), Y_train: (653,)
X_test: (280, 3695), Y_test: (280,)
Training the SVM...
Done!
Accuracy score: 0.6892857142857143

>>> Atheism
X_train: (513, 3317), Y_train: (513,)
X_test: (220, 3317), Y_test: (220,)
Training the SVM...
Done!
Accuracy score: 0.7318181818181818

>>> Feminist Movement
X_train: (664, 3985), Y_train: (664,)
X_test: (285, 3985), Y_test: (285,)
Training the SVM...
Done!
Accuracy score: 0.6385964912280702
```

---

Note (11/24/21): After normalizing the lexicon feature to $[0, 1]$.

```
>>> Hillary Clinton
X_train: (689, 3978), Y_train: (689,)
X_test: (295, 3978), Y_test: (295,)
Training the SVM...
Done!
Accuracy score: 0.6271186440677966

>>> Climate Change is a Real Concern
X_train: (395, 3062), Y_train: (395,)
X_test: (169, 3062), Y_test: (169,)
Training the SVM...
Done!
Accuracy score: 0.7337278106508875

>>> Legalization of Abortion
X_train: (653, 3695), Y_train: (653,)
X_test: (280, 3695), Y_test: (280,)
Training the SVM...
Done!
Accuracy score: 0.6892857142857143

>>> Atheism
X_train: (513, 3317), Y_train: (513,)
X_test: (220, 3317), Y_test: (220,)
Training the SVM...
Done!
Accuracy score: 0.7318181818181818

>>> Feminist Movement
X_train: (664, 3985), Y_train: (664,)
X_test: (285, 3985), Y_test: (285,)
Training the SVM...
Done!
Accuracy score: 0.631578947368421
```

---

Note (11/24/21): Add features of MPQA Subjectivity lexicon by the count of positive words minus the count of negative words. 

```
>>> Hillary Clinton
X_train: (689, 3978), Y_train: (689,)
X_test: (295, 3978), Y_test: (295,)
Training the SVM...
Done!
Accuracy score: 0.6169491525423729

>>> Climate Change is a Real Concern
X_train: (395, 3062), Y_train: (395,)
X_test: (169, 3062), Y_test: (169,)
Training the SVM...
Done!
Accuracy score: 0.6982248520710059

>>> Legalization of Abortion
X_train: (653, 3695), Y_train: (653,)
X_test: (280, 3695), Y_test: (280,)
Training the SVM...
Done!
Accuracy score: 0.6785714285714286

>>> Atheism
X_train: (513, 3317), Y_train: (513,)
X_test: (220, 3317), Y_test: (220,)
Training the SVM...
Done!
Accuracy score: 0.7318181818181818

>>> Feminist Movement
X_train: (664, 3985), Y_train: (664,)
X_test: (285, 3985), Y_test: (285,)
Training the SVM...
Done!
Accuracy score: 0.5929824561403508
```

## Part C

- Parse your training and test data using MALTparser and the predifined model. Then extract dependency triples form the data (word, head, label) and use those as features for the stance detection task instead of the bag-of-words model. How does that affect the results? 

