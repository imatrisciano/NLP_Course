import nltk
import string

from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from util import downloadDataset

class MyNormalizer:
    def __init__(self):
        self.punctuation = MyNormalizer._getPunctuation()
        #downloadDataset("punkt")
        downloadDataset("wordnet")
        downloadDataset("stopwords")

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()


    @staticmethod
    def _getPunctuation() -> list:
        punctuation = set(string.punctuation)
        punctuation.add('``')
        punctuation.add('\'\'')

        return punctuation

    def removePunctuation(self, words) -> list:
        # removes punctuation tokens, such as '.' or ','
        words = [x for x in words if x not in self.punctuation]

        punctuation_str = ''.join(self.punctuation)

        #removes punctuation characters that are part of a word
        words = [w.translate(str.maketrans('', '', punctuation_str)) for w in words]
        words = [w for w in words if len(w) > 0] #removes words that are now empty
        return words

    def makeLowercase(self, words) -> list:
        return [x.lower() for x in words]

    def lemmalizeWords(self, words) -> list:
        return [self.lemmatizer.lemmatize(w) for w in words]

    def stemWords(self, words) -> list:
        return [self.stemmer.stem(w) for w in words]

    def removeStopwords(self, words, language='english') -> list:
        """
        :param words: Must be lowercase
        :param language: Document language, default is english
        :return: a list of words
        """
        ignored_words = nltk.corpus.stopwords.words(language)
        return [w for w in words if w not in ignored_words]

    def normalizeWords(self, words, make_lowercase=True, remove_punctuation=True, lemmalize=False, stem=False,
                       remove_stopwords=False) -> list:
        assert (lemmalize is False or stem is False)  # do not allow lemmalization and stemming at the same time

        if make_lowercase:
            words = self.makeLowercase(words)
        if remove_punctuation:
            words = self.removePunctuation(words)
        if lemmalize:
            words = self.lemmalizeWords(words)
        if stem:
            words = self.stemWords(words)
        if remove_stopwords:
            words = self.removeStopwords(words)

        return words
