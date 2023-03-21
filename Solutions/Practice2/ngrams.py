import nltk
from nltk.probability import *

def getBestBigrams(words, punctuations, number):
    bigram_measures = nltk.collocations.BigramAssocMeasures
    bigrams_finder = nltk.collocations.BigramCollocationFinder.from_words(words)
    bigrams_finder.apply_word_filter(lambda w: w.lower() in punctuations)
    return bigrams_finder.nbest(bigram_measures.pmi, number)

def getBestTrigrams(words, punctuations, number):
    trigram_measures = nltk.collocations.TrigramAssocMeasures
    trigrams_finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
    trigrams_finder.apply_word_filter(lambda w: w.lower() in punctuations)
    return trigrams_finder.nbest(trigram_measures.pmi, number)
