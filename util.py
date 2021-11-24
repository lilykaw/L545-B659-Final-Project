'''
Date: 2021-11-20 13:10:42
LastEditors: yuhhong
LastEditTime: 2021-11-24 15:03:17
'''
import re
import string
import numpy as np
import treetaggerwrapper

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TweetsData(object):
    def __init__(self, df, pos_lexicon=None, neg_lexicon=None): 
        self.df = df
        # check the input dataframe
        assert 'Tweet' in df.columns and 'Target' in df.columns and 'Stance' in df.columns
        
        # preprocess the tweets
        self.df['CleanTweet'] = self.preprocess()

        # added column: new vocab only consists of Nouns, Adjectives, and Verbs
        # basically, filtered sentences of 'CleanTweet'
        self.df['BOW'] = self.get_nav()

        # added column: the count of negative or positive lexicons
        # check that positive and negative lexicons are added at the same time
        assert pos_lexicon != None and neg_lexicon != None or pos_lexicon == None and neg_lexicon == None
        if pos_lexicon != None: 
            self.df['PosLexicon'] = self.gen_lexicon_feature(pos_lexicon)
        if neg_lexicon != None: 
            self.df['NegLexicon'] = self.gen_lexicon_feature(neg_lexicon)
        if pos_lexicon != None and neg_lexicon != None: 
            self.df['CntLexicon'] = self.df['PosLexicon'] - self.df['NegLexicon']
        
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

    def get_data_of_target_bow(self, target): 
        # filter the data by target
        target_df = self.df[self.df['Target']==target]
        X = target_df['BOW'].to_list()
        Y = target_df['Stance'].to_list()
        return X, Y

    def get_nav(self):
        # changes sentences in 'CleanTweet' into only Nouns, Adjectives, & Verbs
        # note: see treetaggerwrapper manual on @daimrod 's GitHub for TreeTagger directory
        NAV_LIST = ['NP', 'NPS', 'NN', 'NNS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        # tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
        tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='/mnt/d/install/treetagger/')
        cleantweets = self.df['CleanTweet'].to_list()
        
        pos_tweets = [tagger.tag_text(tweet) for tweet in cleantweets] # tagger format: 'word\tPOS\tlemma'
        nav = []
        for tweet in pos_tweets:
            str = ''
            for tagged_word in tweet: 
                word, POS, lemma = tagged_word.split('\t')
                if POS in NAV_LIST:
                    str += word + ' '
            nav.append(str)
        return nav

    def get_cnt_lexicon_of_target(self, target): 
        return self.df[self.df['Target']==target]['CntLexicon']

    def gen_lexicon_feature(self, lexicons): 
        count_lexicons = []
        for tweet in self.df['CleanTweet']: 
            count_lexicons.append(sum([1 for w in tweet.split() if w in lexicons]))
        return count_lexicons
