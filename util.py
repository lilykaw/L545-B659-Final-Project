'''
Date: 2021-11-20 13:10:42
LastEditors: yuhhong
LastEditTime: 2021-12-14 14:27:49
'''
import re
import string
import numpy as np
import pandas as pd
import treetaggerwrapper
import pprint

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TweetsData(object):
    def __init__(self, df, mode='AllWords', args=None): 
        # Yuhui: Let's use mode to control which feature we will use. 
        # The number of parameters may lead conflict later. 
        self.df = df
        # check the input dataframe
        assert 'Tweet' in df.columns and 'Target' in df.columns and 'Stance' in df.columns
        assert mode=='AllWords' or mode == 'Bow' or mode == 'SubLexicon' or mode == 'ArgLexicon' or mode == 'MaltParser'
        # preprocess the tweets
        self.df['CleanTweet'] = self.preprocess()

        # Baseline uses the mode of 'AllWords'. Nothing special need to be done in this mode. 

        # Part A
        if mode == 'Bow':
            # added column: new vocab only consists of Nouns, Adjectives, and Verbs
            # basically, filtered sentences of 'CleanTweet'
            self.df['BOW'] = self.gen_nav()
            
        # Part B - subjectivity lexicons
        elif mode == 'SubLexicon': 
            self.df['BOW'] = self.gen_nav()

            pos_lexicon, neg_lexicon = args
            # added column: the count of negative or positive lexicons
            # check that positive and negative lexicons are added at the same time
            assert pos_lexicon != None and neg_lexicon != None or pos_lexicon == None and neg_lexicon == None
            if pos_lexicon != None: 
                self.df['PosLexicon'] = self.gen_lexicon_feature(pos_lexicon)
            if neg_lexicon != None: 
                self.df['NegLexicon'] = self.gen_lexicon_feature(neg_lexicon)
            if pos_lexicon != None and neg_lexicon != None: 
                # self.df['CntSubLex'] = self.df['PosLexicon'] - self.df['NegLexicon']
                self.df['CntSubLex'] = [[p, n, p-n] for p, n in zip(self.df['PosLexicon'], self.df['NegLexicon'])]
  
        # Part B - arguing lexicons
        elif mode == 'ArgLexicon': 
            self.df['BOW'] = self.gen_nav()

            regex_patterns = args
            # added column: the count of arguing lexicons
            self.df['CntArgLex'] = self.gen_lexicon_feature_redict(regex_patterns)

        # Part C - parsing
        elif mode == 'MaltParser':
            dep_triples = args
            # added column: represent the triples of parser in a one-hot vector
            self.gen_parser_feature(dep_triples) # it is a inplace function
        

    def preprocess(self): 
        # 1. remove the punctuations 
        remove_punctuation = string.punctuation + '??????'
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
        # clean_tweets = [re.sub('@[^\s]+', '', t) for t in filtered_tweets]
        # return clean_tweets
        ''' If we remove usernames, that step would actually need to go before removing punctuation (because the regex
        expression looks for '@', which is in the set of punctuation. But when I did that, the accuracy
        decreased. So I don't think usernames should be removed. '''
        return filtered_tweets

    def __len__(self):
        return len(self.df)

    def print_df(self):
        print(self.df)
        return
    

    # Please put the "gen" functions here ================================
    # These functions are used to generate feature for the Dataset, in initialization. 
    def gen_nav(self):
        # changes sentences in 'CleanTweet' into only Nouns, Adjectives, & Verbs
        # note: see treetaggerwrapper manual on @daimrod 's GitHub for TreeTagger directory
        NAV_LIST = ['NP', 'NPS', 'NN', 'NNS', 
        'JJ', 'JJR', 'JJS', 
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VD', 'VDG', 'VDN', 'VDZ', 'VDP', 'VH', 'VHD', 'VHG', 'VHN', 'VHZ', 'VHP', 'VV', 'VVD', 'VVG', 'VVN', 'VVP', 'VVZ']
        # tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
        tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='/mnt/d/install/treetagger/')
        cleantweets = self.df['CleanTweet'].to_list()

        pos_tweets = [tagger.tag_text(tweet) for tweet in cleantweets] # tagger format: 'word\tPOS\tlemma'
        # Yuhui: Uncommend these line when use. 
        # f = open('tweets_tagged.txt', 'w')
        # for tweet in pos_tweets:
        #    f.write(tweet)
        # f.close()
        nav = []
        for tweet in pos_tweets:
            str = ''
            for tagged_word in tweet: 
                word, POS, lemma = tagged_word.split('\t')
                if POS in NAV_LIST:
                    str += word + ' '
            nav.append(str)
        return nav

    def gen_lexicon_feature(self, lexicons): 
        count_lexicons = []
        for tweet in self.df['BOW']: 
            count_lexicons.append(sum([1 for w in tweet.split() if w in lexicons]))
        return count_lexicons

    def gen_lexicon_feature_redict(self, lexicons): 
        count_lexicons = []
        for tweet in self.df['BOW']: 
            count_lexicons_each = [] # a list of count of lexicons for each sentence
            for lex_list in lexicons:
                count_lexicons_each.append(sum([len(re.findall(lex, tweet)) for lex in lex_list]))
            count_lexicons.append(count_lexicons_each)
        return count_lexicons # shape: (n, 17)

    def gen_parser_feature(self, dep_triples): 
        targets = [t for t in self.get_targets() if t != 'Donald Trump']
        for t in targets: 
            # print(len(self.df[self.df['Target']==t]), len(dep_triples[t]))
            self.df.loc[self.df['Target']==t, 'ParserArrows'] = dep_triples[t]
        return 



    # Please put the "get" functions here ================================
    # These functions are used to export the features. 
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

    def get_cnt_sublexicon_of_target(self, target): 
        return self.df[self.df['Target']==target]['CntSubLex'].to_list()

    def get_cnt_arglexicon_of_target(self, target):  
        return self.df[self.df['Target']==target]['CntArgLex'].to_list()

    def get_parser_of_target(self, target):
        return self.df[self.df['Target']==target]['ParserArrows'].to_list()