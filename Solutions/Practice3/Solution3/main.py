import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_validate

from dataset_reader import read_dataset
from EmbeddingDatasetReader import *
from normalization import MyNormalizer
from ngrams import *


itwac_path = '../../../data/word2vec/itwac128.sqlite'
twitter_path = '../../../data/word2vec/twitter128.sqlite'
haspedee_dataset_path = '../../../data/hate_speech/haspeede2020/haspeede2_dev_taskAB.tsv'
#haspedee_test_dataset_path = '../../../data/hate_speech/haspeede2020/haspeede2_test_taskAB-tweets.tsv'
id_to_label = {"0": "NoHate", "1": "Hate"}


embedding_size = 128


def evaluate(y_ref, y_predicted):
    a = accuracy_score(y_ref, y_predicted)
    p = precision_score(y_ref, y_predicted)
    r = recall_score(y_ref, y_predicted)
    f1 = f1_score(y_ref, y_predicted)

    print(f"accuracy: {a}")
    print(f"precision: {p}")
    print(f"recall: {r}")
    print(f"f1: {f1}")


def customFilter(words: list) -> list:
    blacklist = ["user", "url"]
    return [x for x in words if x not in blacklist and not contains_digits(x)]

def contains_digits(s):
    return any(char.isdigit() for char in s)

def svc(x, y, cv=5):

    param_grid = {'C': [0.01, 0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['auto']}
    grid = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=-1, verbose=1,
                        scoring=("accuracy", "precision", "recall", "f1"),
                        refit="f1")
    grid.fit(x, y)

    best_model = grid.best_estimator_
    results = pd.DataFrame(grid.cv_results_)  # results contains all the metrics for every parameter combination
    best_model_metrics = results.iloc[[grid.best_index_]]  # select only the row of the best classifier
    return best_model, best_model_metrics

def extractDocumentFeatures(words:list) -> np.array:
    """
    # By executing this we can observe that capitalization does matter in the embeddings dataset as the embedding
    # vectors of 'Che' and 'che' are different
    print(embeddings.getWordEmbedding("che"))
    print(embeddings.getWordEmbedding("Che"))
    """

    words = wordNormalizer.normalizeWords(words, make_lowercase=False, remove_punctuation=True, remove_stopwords=True)
    words = customFilter(words)

    embeddingSum = np.zeros(embedding_size)

    for word in words:
        wordEmbedding = embeddings.getWordEmbedding(word)
        embeddingSum += wordEmbedding

    return embeddingSum

if __name__ == "__main__":
    data_path = twitter_path

    # Reading embeddings
    # Each row contains a word and the corresponding embedding (128 dimensions)
    embeddings = EmbeddingDatasetReader(data_path)



    documents, labels, text_to_id_map = read_dataset(haspedee_dataset_path)
    train_documents, test_documents, train_labels, test_labels = train_test_split(documents, labels, test_size=0.1)
    wordNormalizer = MyNormalizer(language="italian")

    train_features = []
    for doc in train_documents:
        train_features.append(extractDocumentFeatures(doc))

    classifier, metrics = svc(train_features, train_labels)


    test_features = []
    for doc in test_documents:
        test_features.append(extractDocumentFeatures(doc))

    predicted_labels = classifier.predict(test_features)
    evaluate(test_labels, predicted_labels)

    """
    Output:
    accuracy: 0.7485380116959064
    precision: 0.6863468634686347
    recall: 0.6813186813186813
    f1: 0.6838235294117647
    """















