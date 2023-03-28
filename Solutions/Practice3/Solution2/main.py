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
    """
    param_grid = {'C': [10],
                  'kernel': ['rbf'],
                  'gamma': ['auto']}
    """
    grid = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=1, verbose=2,
                        scoring=("accuracy", "precision", "recall", "f1"),
                        refit="f1")
    grid.fit(x, y)

    best_model = grid.best_estimator_
    results = pd.DataFrame(grid.cv_results_)  # results contains all the metrics for every parameter combination
    best_model_metrics = results.iloc[[grid.best_index_]]  # select only the row of the best classifier
    return best_model, best_model_metrics



if __name__ == "__main__":
    print("This solution does not work.")
    
    data_path = twitter_path

    # Reading embeddings
    # Each row contains a word and the corresponding embedding (128 dimensions)
    embeddings = EmbeddingDatasetReader(data_path)

    """
    # By executing this we can observe that capitalization does matter in the embeddings dataset as the embedding
    # vectors of 'Che' and 'che' are different
    print(embeddings.searchWord("che"))
    print(embeddings.searchWord("Che"))
    """

    documents, labels, text_to_id_map = read_dataset(haspedee_dataset_path)
    train_documents, test_documents, train_labels, test_labels = train_test_split(documents, labels, test_size=0.1)
    wordNormalizer = MyNormalizer(language="italian")

    everyHateDocument = [train_documents[i] for i in range(len(train_documents)) if train_labels[i] == 1]
    everyNonHateDocument = [train_documents[i] for i in range(len(train_documents)) if train_labels[i] == 0]

    allHateText = ' '.join(everyHateDocument)
    allHateText = wordNormalizer.normalizeWords(allHateText, make_lowercase=True, remove_punctuation=True, remove_stopwords=True)
    allHateText = customFilter(allHateText)

    allNonHateText = ' '.join(everyNonHateDocument)
    allNonHateText = wordNormalizer.normalizeWords(allNonHateText, make_lowercase=True, remove_punctuation=True, remove_stopwords=True)
    allNonHateText = customFilter(allNonHateText)


    hateUnigrams = getBestUnigrams(allHateText, 50)
    nonHateUnigrams = getBestUnigrams(allNonHateText, 50)
    print(f"Hate Unigrams: {hateUnigrams}")
    print(f"Non hate Unigrams: {nonHateUnigrams}")


    hateUnigramWords = [x[0] for x in hateUnigrams]
    nonHateUnigramWords = [x[0] for x in nonHateUnigrams]
    exclusiveHateUnigrams = [x for x in hateUnigrams if x[0] not in nonHateUnigramWords]
    exclusiveNonHateUnigrams = [x for x in nonHateUnigrams if x[0] not in hateUnigramWords]
    print(f"Exclusive Hate Unigrams: {exclusiveHateUnigrams}")
    print(f"Exclusive Non hate Unigrams: {exclusiveNonHateUnigrams}")

    print("\r\n")

    """
    hateBigrams = getBestBigrams(allHateText, MyNormalizer.getPunctuation(), 50)
    nonHateBigrams = getBestBigrams(allNonHateText, MyNormalizer.getPunctuation(), 50)
    print(f"Hate Bigrams: {hateBigrams}")
    print(f"Non hate Bigrams: {nonHateBigrams}")

    exclusiveHateBigrams = [x for x in hateBigrams if x[0] not in nonHateBigrams]
    exclusiveNonHateBigrams = [x for x in nonHateBigrams if x[0] not in hateBigrams]

    print(f"Exclusive Hate Bigrams: {exclusiveHateBigrams}")
    print(f"Exclusive Non hate Bigrams: {exclusiveNonHateBigrams}")

    print("\r\n")

    hateTrigrams = getBestTrigrams(allHateText, MyNormalizer.getPunctuation(), 10)
    nonHateTrigrams = getBestTrigrams(allNonHateText, MyNormalizer.getPunctuation(), 10)
    print(f"Hate Trigrams: {hateTrigrams}")
    print(f"Non hate Trigrams: {nonHateTrigrams}")

    print("\r\n")
    """

    _train_data = []
    _train_labels = []
    for unigram in exclusiveHateUnigrams:
        word = unigram[0]
        entry = embeddings.getWordEntry(word)
        if entry != None:
            embedding = embeddings.getEmbeddingVectorFromEntry(entry)
            _train_data.append(embedding)
            _train_labels.append(1)
    for unigram in exclusiveNonHateUnigrams:
        word = unigram[0]
        entry = embeddings.getWordEntry(word)
        if entry != None:
            embedding = embeddings.getEmbeddingVectorFromEntry(entry)
            _train_data.append(embedding)
            _train_labels.append(0)

    classifier, metrics = svc(_train_data, _train_labels)


    test_features = []
    for document, gold_label in zip(test_documents, test_labels):
        document = wordNormalizer.normalizeWords(document, make_lowercase=True, remove_punctuation=True,
                                                    remove_stopwords=True)
        document = customFilter(document)
        documentUnigrams = getBestUnigrams(document, 50)
        embeddingSum = np.zeros(128)
        for unigram in documentUnigrams:
            word = unigram[0]
            embeddingSum += embeddings.getWordEmbedding(word)
        test_features.append(embeddingSum)

    predicted_labels = classifier.predict(test_features)
    evaluate(test_labels, predicted_labels)















