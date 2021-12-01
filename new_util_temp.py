# import re
import regex as re
import string
import numpy as np
import treetaggerwrapper

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



class TweetsData(object):
    def __init__(self, df, *args):
        self.df = df
        # check the input dataframe
        assert 'Tweet' in df.columns and 'Target' in df.columns and 'Stance' in df.columns
        # preprocess the tweets
        self.df['CleanTweet'] = self.preprocess()

        # Part A
        if len(args) == 0:
            self.df = df
            # added column: new vocab only consists of Nouns, Adjectives, and Verbs
            # basically, filtered sentences of 'CleanTweet'
            self.df['BOW'] = self.get_nav()
            
        # Part B - subjectivity lexicons
        elif len(args) == 2:
            pos_lexicon = args[0]
            neg_lexicon = args[1]
            # added column: the count of negative or positive lexicons
            # check that positive and negative lexicons are added at the same time
            assert pos_lexicon != None and neg_lexicon != None or pos_lexicon == None and neg_lexicon == None
            if pos_lexicon != None: 
                self.df['PosLexicon'] = self.gen_lexicon_feature(pos_lexicon)
            if neg_lexicon != None: 
                self.df['NegLexicon'] = self.gen_lexicon_feature(neg_lexicon)
            if pos_lexicon != None and neg_lexicon != None: 
                self.df['CntLexicon'] = self.df['PosLexicon'] - self.df['NegLexicon']
  
        # Part B - arguing lexicons
        elif len(args) == 1:
            assert type(args) == tuple
            macro_dict = args[0][0]
            regex_patterns = args[0][1] # how to unpack a tuple?
            # added column: the count of arguing lexicons
            self.df['CntArgLex'] = self.gen_regex_lexicon_feature(macro_dict, regex_patterns)
        

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

    def get_cnt_lexicon_of_target(self, target, colName): 
        return self.df[self.df['Target']==target][colName].to_list()

    def gen_lexicon_feature(self, lexicons): 
        count_lexicons = []
        for tweet in self.df['CleanTweet']: 
            count_lexicons.append(sum([1 for w in tweet.split() if w in lexicons]))
        return count_lexicons
    
    def gen_regex_lexicon_feature(self, macro_dict, regex_patterns):
        count_lexicons = []
        # all possible regex patterns
        for tweet in self.df['CleanTweet']:
            # count_lexicons.append(sum([1 for pat in regex_patterns if re.search(pat, tweet)]))
            cnt = 0
            for pat in regex_patterns:
                if re.search(pat, tweet):
                    cnt += 1
            count_lexicons.append(cnt)
        return count_lexicons
    
def expand_regex_macros(regex_pattern, macro_dict, poss_patterns):
# expand macros in regex_pattern, if applicable, and return the possible RegEx patterns
# e.g. in difficulty.tff, we have this line: "(@BE) (@INTENSADV1)?easy"
    macros_found = set()
    if "=" not in regex_pattern:
        try:
            macros_found.update(re.findall(r'@\w+\b', regex_pattern))
        except AttributeError:
            pass
    
    if len(macros_found) == 0:
        return [regex_pattern]
    else:
        for m in macros_found:
            val = m[1:] # remove '@' to do look-u
            dict_values = macro_dict.get(val)
            if type(dict_values) is not list:
                dict_values = [dict_values]
            for v in dict_values:
                r = regex_pattern.replace(m, v) # take turns replacing macro with the values of key(m) in macro_dict
                poss_patterns += [r]
                expand_regex_macros(r, macro_dict, poss_patterns)
        return poss_patterns
