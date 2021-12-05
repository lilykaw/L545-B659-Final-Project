import csv
import numpy as np 
from nltk.tokenize import sent_tokenize, word_tokenize

def write_tweets(name, part, tweets):
   f = open('{}_{}.txt'.format(name, part), 'w')
   for tweet in tweets:
      f.write(tweet)
      f.write('\n')
   f.close()
def process(filename):
   tt = {}
   hillary = []
   abortion = []
   climate_change = []
   atheism = []
   feminist_movement = []
   donald = []
   with open(filename, 'r', encoding='latin1') as csv_file:
      csv_reader = csv.DictReader(csv_file, delimiter=',')
      targets = []
      tweets = []
      for lines in csv_reader:
         tt[lines['Tweet']] = lines['Target']

   for k,v in tt.items():
      if v == 'Hillary Clinton':
         hillary.append(k)
      elif v == 'Legalization of Abortion':
         abortion.append(k)
      elif v == 'Climate Change is a Real Concern':
         climate_change.append(k)
      elif v == 'Atheism':
         atheism.append(k)
      elif v == 'Donald Trump':
         donald.append(k)
      else:
         feminist_movement.append(k)
   if csv_file == 'train.csv':
      write_tweets('hillary', 'train', hillary)
      write_tweets('abortion', 'train', abortion)
      write_tweets('climate_change', 'train', climate_change)
      write_tweets('atheism', 'train', atheism)
      write_tweets('feminist_movement', 'train', feminist_movement)
   else:
      write_tweets('hillary', 'test', hillary)
      write_tweets('abortion', 'test', abortion)
      write_tweets('climate_change', 'test', climate_change)
      write_tweets('atheism', 'test', atheism)
      write_tweets('feminist_movement', 'test', feminist_movement)
      write_tweets('donald', 'test', donald)

process('train.csv')
process('test.csv')
