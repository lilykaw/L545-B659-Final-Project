# Extend data set to include features using the MPQA Subjectivity lexicon
# Decide on a good way of using this information in features
# The info includes:
# - type (strongsubj or weaksubj)
# - len (not useful because all of the lengths are 1)
# - word1
# - pos1 (noun, adjective, adverb, verb...)
# - stemmed1 (y for yes, n for no)
# - priorpolarity (positive or negative)

# Plan (based on https://export.arxiv.org/ftp/arxiv/papers/1703/1703.02019.pdf):
# 1) For each word in Tweets, check if word lemma is in subjectivity lexicon (file name: subjclueslen1-HLTEMNLP05.tff)
# 2) If word from Tweets is in the lexicon, determine the priorpolarity (positive or negative)
# 3) Assign +1 for positive polarity and -1 for negative polarity. Assign 0 if the word is not in the lexicon.
## Note: for this, we can use all words for bag of words since there are more than just adjectives, nouns and verbs in the lexicon
## To do: figure out how to assign the polarity scores and include these as features...

