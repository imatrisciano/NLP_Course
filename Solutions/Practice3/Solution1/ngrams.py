import nltk
from nltk.probability import *

def getBestUnigrams(words, number):
    freqDist = FreqDist(words)
    return freqDist.most_common(number)


def getBestBigrams(words, punctuations, number):
    bigram_measures = nltk.collocations.BigramAssocMeasures
    bigrams_finder = nltk.collocations.BigramCollocationFinder.from_words(words, 3)
    bigrams_finder.apply_word_filter(lambda w: w.lower() in punctuations)
    return bigrams_finder.nbest(bigram_measures.pmi, number)

def getBestTrigrams(words, punctuations, number):
    trigram_measures = nltk.collocations.TrigramAssocMeasures
    trigrams_finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
    trigrams_finder.apply_word_filter(lambda w: w.lower() in punctuations)
    return trigrams_finder.nbest(trigram_measures.pmi, number)
