'''
Date: 2021-11-20 13:10:42
LastEditors: yuhhong
LastEditTime: 2021-11-20 14:56:45
'''
import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TweetsData(object):
    def __init__(self, df): 
        self.df = df
        # check the input dataframe
        assert 'Tweet' in df.columns and 'Target' in df.columns and 'Stance' in df.columns
        
        # preprocess the tweets
        self.df['CleanTweet'] = self.preprocess()
        
    def preprocess(self): 
        # 1. remove the punctuations 
        remove_punctuation = string.punctuation + '‘’'
        translator = str.maketrans('', '', remove_punctuation)
        tweets = [sent.translate(translator) for sent in self.df['Tweet']]
        # 2. remove stopwords (holly)
        stop_words = set(stopwords.words('english'))
        word_tokens = [word_tokenize(sent) for sent in tweets]
        filtered_tweets = []
        for sent in word_tokens:
            filtered_sentence = [w for w in sent if w not in stop_words]
            filtered_tweets.append(' '.join(filtered_sentence))
        # 3. remove usernames (holly)
        clean_tweets = [re.sub('@[^\s]+', '', t) for t in filtered_tweets]
        return clean_tweets

    def __len__(self):
        return len(self.df)

    def print_df(self):
        print(self.df)
        return

    def get_targets(self):
        targets = set()
        for t in self.df['Target']:
            targets.add(t)
        return list(targets)

    def get_data_of_target(self, target): 
        # filter the data by target
        target_df = self.df[self.df['Target']==target]
        X = target_df['CleanTweet'].to_list()
        Y = target_df['Stance'].to_list()
        return X, Y
