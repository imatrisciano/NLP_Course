import nltk
from nltk import FreqDist


def downloadDataset(datasetName: str):
    try:
        nltk.data.find(datasetName)
    except LookupError:
        nltk.download(datasetName)


def getMostCommonWordsSet(words, n: int) -> set:
    """
    Computes the FreqDist of words and returns the n most common
    :param n: quante parole restituire
    :return: Set delle parole più frequenti, ordinate per frequenza decrescente
    """

    freqDist = FreqDist(words)
    most_common = [x[0] for x in freqDist.most_common(n)]
    return most_common

def getWordsInCategories(dataset, categories) -> list:
    fileids = dataset.fileids(categories=categories)
    return dataset.words(fileids)


