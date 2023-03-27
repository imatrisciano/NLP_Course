import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from dataset_reader import read_dataset
from WordEmbedding import *
from normalization import MyNormalizer
from ngrams import *

itwac_path = '../../data/word2vec/itwac128.sqlite'
twitter_path = '../../data/word2vec/twitter128.sqlite'
haspedee_dataset_path = '../../data/hate_speech/haspeede2020/haspeede2_dev_taskAB.tsv'
id_to_label = {"0": "NoHate", "1": "Hate"}


embedding_size = 128
n_unigrams = 5
n_bigrams = 3
n_trigrams = 1


def evaluate(y_ref, y_predicted):
    a = accuracy_score(y_ref, y_predicted)
    p = precision_score(y_ref, y_predicted, pos_label="1")
    r = recall_score(y_ref, y_predicted, pos_label="1")
    f1 = f1_score(y_ref, y_predicted, pos_label="1")

    print(f"accuracy: {a}")
    print(f"precision: {p}")
    print(f"recall: {r}")
    print(f"f1: {f1}")


def customFilter(words: list) -> list:
    blacklist = ["user", "url"]
    return [x for x in words if x not in blacklist]


def makeFeaturesVector(embeddings: WordEmbedding, unigrams, bigrams, trigrams) -> np.array:
    feature_vector = np.array([])

    for i in range(n_unigrams):
        if i < len(unigrams):
            unigram = unigrams[i]
            word = unigram[0]
            embedding = embeddings.getWordEmbedding(word)
            feature_vector = np.append(feature_vector, embedding)
        else:
            feature_vector = np.pad(feature_vector, (0, embedding_size))

    for i in range(n_bigrams):
        if i < len(bigrams):
            bigram = bigrams[i]
            for word in bigram:
                embedding = embeddings.getWordEmbedding(word)
                feature_vector = np.append(feature_vector, embedding)
        else:
            feature_vector = np.pad(feature_vector, (0, 2*embedding_size))


    for i in range(n_trigrams):
        if i < len(trigrams):
            trigram = trigrams[i]
            for word in trigram:
                embedding = embeddings.getWordEmbedding(word)
                feature_vector = np.append(feature_vector, embedding)
        else:
            feature_vector = np.pad(feature_vector, (0, 3*embedding_size))

    return feature_vector

def getFeatureVectorSize():
    return (n_unigrams + 2*n_bigrams + 3*n_trigrams) * embedding_size

if __name__ == "__main__":
    data_path = twitter_path
    #print(os.getcwd())

    # Reading embeddings
    # Each row contains a word and the corresponding embedding (128 dimensions)
    embeddings = WordEmbedding(data_path)

    """
    # By executing this we can observe that capitalization does matter in the embeddings dataset as the embedding
    # vectors of 'Che' and 'che' are different
    print(embeddings.searchWord("che"))
    print(embeddings.searchWord("Che"))
    """

    documents, labels, text_to_id_map = read_dataset(haspedee_dataset_path)
    wordNormalizer = MyNormalizer(language="italian")

    documentFeatures = np.array([])
    items_count = 200 #len(documents)
    #for document, label in zip(documents, labels):
    #for i in range(len(documents)):
    for i in range(items_count):
        document = documents[i]
        label = labels[i]

        print(f"\rExtracting features from document {i}/{items_count}...", end="")

        #print(f"Document: {document}")
        words = wordNormalizer.normalizeWords(document, make_lowercase=True, remove_punctuation=True, remove_stopwords=True)
        words = customFilter(words)
        #print(f"Normalized document: {words}")
        #print(f"Label: {id_to_label[label]}")

        unigrams = getBestUnigrams(words, n_unigrams)
        bigrams = getBestBigrams(words, MyNormalizer.getPunctuation(), n_bigrams)
        trigrams = getBestTrigrams(words, MyNormalizer.getPunctuation(), n_trigrams)

        #print(f"Unigrams: {unigrams}")
        #print(f"Bigrams: {bigrams}")
        #print(f"Trigrams: {trigrams}")
        #print("\r\n")
        features = makeFeaturesVector(embeddings, unigrams, bigrams, trigrams)
        assert(len(features) == getFeatureVectorSize())
        documentFeatures = np.append(documentFeatures, features)
        #print("\r\n\r\n\r\n")

    documentFeatures = documentFeatures.reshape((items_count, getFeatureVectorSize()))


    # TODO: evaluate(y, y_predicted)











