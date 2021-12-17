# Notes

Here is all the notes we got. 

## Part A

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

Note (12/02/2021): Add arguing lexicons for 17 classes separately.

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

```
Default SVM:

>>> Hillary Clinton
X_train: (689, 14925), Y_train: (689,)
X_test: (295, 14925), Y_test: (295,)
Training the SVM...
Done!
Accuracy score: 0.5864406779661017

>>> Climate Change is a Real Concern
X_train: (395, 9590), Y_train: (395,)
X_test: (169, 9590), Y_test: (169,)
Training the SVM...
Done!
Accuracy score: 0.7337278106508875

>>> Legalization of Abortion
X_train: (653, 14327), Y_train: (653,)
X_test: (280, 14327), Y_test: (280,)
Training the SVM...
Done!
Accuracy score: 0.6857142857142857

>>> Atheism
X_train: (513, 12524), Y_train: (513,)
X_test: (220, 12524), Y_test: (220,)
Training the SVM...
Done!
Accuracy score: 0.7272727272727273

>>> Feminist Movement
X_train: (664, 15643), Y_train: (664,)
X_test: (285, 15643), Y_test: (285,)
Training the SVM...
Done!
Accuracy score: 0.6456140350877193
```

## Part D