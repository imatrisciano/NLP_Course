import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from Solutions.Task_April_20.dataset_reader import read_dataset
from Solutions.Task_April_20.neural_network import train_deep_neural_network
from performance import evaluate, show_cv_performance

model_name = "nickprock/sentence-bert-base-italian-uncased" # https://huggingface.co/nickprock/sentence-bert-base-italian-uncased
# model_name = "nickprock/sentence-bert-base-italian-xxl-uncased"  # https://huggingface.co/nickprock/sentence-bert-base-italian-xxl-uncased
# model_name = "xlm-roberta-base"

haspedee_dataset_path = '../../data/hate_speech/haspeede2020/haspeede2_dev_taskAB.tsv'
# haspedee_test_dataset_path = '../../../data/hate_speech/haspeede2020/haspeede2_test_taskAB-tweets.tsv'
id_to_label = {0: "NoHate", 1: "Hate"}

if __name__ == "__main__":
    print("Loading transformer model...")
    import torch_directml
    directML = torch_directml.device()
    bert_model = SentenceTransformer(model_name, device=directML)

    frase1 = "Pippo Baudo è capellone"
    frase2 = "Pippo Baudo è capellone 🧑‍🦰🧑‍🦰🧑‍🦰 "

    embedding1 = bert_model.encode(frase1)
    embedding2 = bert_model.encode(frase2)

    equalEmbedding = np.array_equal(embedding1, embedding2)
    print(f"Sentence1: {frase1}")
    print(f"Sentence2: {frase2}")
    print(f"The two sentences have the same embedding? {equalEmbedding}")
    if not equalEmbedding:
        difference = embedding1 - embedding2
        print(f"Embedding difference: {difference}")






    print("Loading dataset...")
    documents, labels, text_to_id_map = read_dataset(haspedee_dataset_path)
    filtered_documents = []
    for doc in documents:
        filtered_doc = doc.replace("@user", "").replace("URL", "")
        filtered_documents.append(filtered_doc)

    train_documents, test_documents, train_labels, test_labels = train_test_split(filtered_documents, labels, test_size=0.1)

    print("Extracting training features...")
    train_features = bert_model.encode(train_documents)

    classifier, metrics = train_deep_neural_network(train_features, train_labels, 768)

    print(f" > Best classifier is {classifier}")
    print(metrics)

    print("Extracting and test features...")
    test_features = bert_model.encode(test_documents)
    print("Predicting test data...")
    predicted_labels = classifier.predict(test_features)

    evaluate(test_labels, predicted_labels)
    show_cv_performance(classifier, metrics)

    documents_successfully_turned_into_HATE = 0
    number_of_analyzed_HATE_documents = 0
    documents_successfully_turned_into_NON_HATE = 0
    number_of_analyzed_NON_HATE_documents = 0

    for doc, label in zip(test_documents, test_labels):
        if label == 0:
            number_of_analyzed_NON_HATE_documents = number_of_analyzed_NON_HATE_documents + 1
            print(f"Analyzing NON-HATE sentence {doc}")
            doc_with_angry_emoji = f"{doc} 😠😠😡"

            documents = [doc, doc_with_angry_emoji]
            features = bert_model.encode(documents)
            y = classifier.predict(features)

            y1 = y[0][0]
            y2 = y[1][0]

            if y1 != y2:
                documents_successfully_turned_into_HATE = documents_successfully_turned_into_HATE + 1

            print(f"> The original sentence was classified as {id_to_label[y1]}")
            print(f"> The sentence with '😠😠😡' was classified as {id_to_label[y2]}")
        else:
            number_of_analyzed_HATE_documents = number_of_analyzed_HATE_documents + 1
            print(f"Analyzing HATE sentence {doc}")
            doc_with_love_emoji = f"{doc} ❤️😍😘"

            documents = [doc, doc_with_love_emoji]
            features = bert_model.encode(documents)
            y = classifier.predict(features)

            y1 = y[0][0]
            y2 = y[1][0]

            if y1 != y2:
                documents_successfully_turned_into_NON_HATE = documents_successfully_turned_into_NON_HATE + 1

            print(f"> The original sentence was classified as {id_to_label[y1]}")
            print(f"> The sentence with '❤️😍😘' was classified as {id_to_label[y2]}")

    print(f"{documents_successfully_turned_into_HATE}/{number_of_analyzed_NON_HATE_documents} NON-HATE documents were turned into HATE by adding '😠😠😡'")
    print(f"{documents_successfully_turned_into_NON_HATE}/{number_of_analyzed_HATE_documents} HATE documents were turned into NON-HATE by adding '❤️😍😘'")

"""
OUTPUT:
    9/399 NON-HATE documents were turned into HATE by adding '😠😠😡'
    18/285 HATE documents were turned into NON-HATE by adding '❤️😍😘'
"""