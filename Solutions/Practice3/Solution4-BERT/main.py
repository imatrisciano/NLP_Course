import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler

from dataset_reader import read_dataset
from normalization import MyNormalizer

from sentence_transformers import SentenceTransformer
from neural_network import train_deep_neural_network

#model_name = "nickprock/sentence-bert-base-italian-uncased" # https://huggingface.co/nickprock/sentence-bert-base-italian-uncased
model_name = "nickprock/sentence-bert-base-italian-xxl-uncased"  # https://huggingface.co/nickprock/sentence-bert-base-italian-xxl-uncased

#model_name = "xlm-roberta-base"


haspedee_dataset_path = '../../../data/hate_speech/haspeede2020/haspeede2_dev_taskAB.tsv'
# haspedee_test_dataset_path = '../../../data/hate_speech/haspeede2020/haspeede2_test_taskAB-tweets.tsv'
id_to_label = {"0": "NoHate", "1": "Hate"}


def evaluate(y_ref, y_predicted):
    a = accuracy_score(y_ref, y_predicted)
    p = precision_score(y_ref, y_predicted)
    r = recall_score(y_ref, y_predicted)
    f1 = f1_score(y_ref, y_predicted)

    print("Metrics con test set:")
    print(f" accuracy: {a}")
    print(f" precision: {p}")
    print(f" recall: {r}")
    print(f" f1: {f1}")


def customFilter(words: list) -> list:
    blacklist = ["user", "url"]
    return [x for x in words if x not in blacklist and not contains_digits(x)]


def contains_digits(s):
    return any(char.isdigit() for char in s)


def svc(x, y, cv=5) -> (svm.SVC, pd.DataFrame):
    """
    This piece of code was written in collaboration with github.com/alessandroquirile/ for another project
    :param x: training data
    :param y: labels
    :param cv: number of cross validation folds
    :return: best_model, metrics
    """

    param_grid = {'C': [0.01, 0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['auto']}
    grid = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=-1, verbose=0,
                        scoring=("accuracy", "precision", "recall", "f1"),
                        refit="f1")
    grid.fit(x, y)

    best_model = grid.best_estimator_
    results = pd.DataFrame(grid.cv_results_)  # results contains all the metrics for every parameter combination
    best_model_metrics = results.iloc[[grid.best_index_]]  # select only the row of the best classifier
    return best_model, best_model_metrics


if __name__ == "__main__":
    print("Loading transformer model...")
    import torch_directml
    directML = torch_directml.device()
    bert_model = SentenceTransformer(model_name, device=directML)

    print("Loading dataset...")
    documents, labels, text_to_id_map = read_dataset(haspedee_dataset_path)
    #wordNormalizer = MyNormalizer(language="italian")

    filtered_documents = []
    for doc in documents:
        filtered_doc = doc.replace("@user", "").replace("URL", "")
        filtered_documents.append(filtered_doc)

    train_documents, test_documents, train_labels, test_labels = train_test_split(filtered_documents, labels, test_size=0.1)


    print("Extracting training features...")
    train_features = bert_model.encode(train_documents)

    """
    print("Fitting features scaler...")
    scaler = StandardScaler()  # define the scaler
    scaler.fit(train_features)  # fit on the training dataset

    train_features = scaler.transform(train_features)

    print("Training classification model...")
    # classifier, metrics = svc(train_features, train_labels)
    """

    classifier, metrics = train_deep_neural_network(train_features, train_labels, 768)

    print(f" > Best classifier is {classifier}")
    print(metrics)

    print("Extracting and test features...")
    test_features = bert_model.encode(test_documents)
    # test_features = scaler.transform(test_features)

    print("Predicting test data...")
    predicted_labels = classifier.predict(test_features)
    evaluate(test_labels, predicted_labels)


    def show_cv_performance(model, scores):
        accuracy_avg, accuracy_std, precision_avg, precision_std, recall_avg, recall_std, f1_avg, f1_std = \
            get_performance_from_scores(scores)

        print(
            f"Model: {model}\n"
            f"Accuracy: {accuracy_avg} ± {accuracy_std}\n"
            f"Precision: {precision_avg} ± {precision_std}\n"
            f"Recall: {recall_avg} ± {recall_std}\n"
            f"F1: {f1_avg} ± {f1_std}"
        )


    def get_performance_from_scores(scores):
        accuracy_avg = scores['mean_test_accuracy'].iloc[0]
        accuracy_std = scores['std_test_accuracy'].iloc[0]
        precision_avg = scores['mean_test_precision'].iloc[0]
        precision_std = scores['std_test_precision'].iloc[0]
        recall_avg = scores['mean_test_recall'].iloc[0]
        recall_std = scores['std_test_recall'].iloc[0]
        f1_avg = scores['mean_test_f1'].iloc[0]
        f1_std = scores['std_test_f1'].iloc[0]

        return accuracy_avg, accuracy_std, precision_avg, precision_std, recall_avg, recall_std, f1_avg, f1_std

    show_cv_performance(classifier, metrics)
    """
    model_name = "nickprock/sentence-bert-base-italian-uncased"
        Output with SVC and features scaling:
            Metrics con test set:
            accuracy: 0.7660818713450293
            precision: 0.7163636363636363
            recall: 0.7060931899641577
            f1: 0.7111913357400721
        Output con Deep Neaural Network:
            Test set results:
                accuracy: 0.7733918128654971
                precision: 0.7386363636363636
                recall: 0.693950177935943
                f1: 0.7155963302752294
            
            Cross validation results:
                Accuracy: 0.7702680747359869 ± 0.005070507403518203
                Precision: 0.7217562130252551 ± 0.014111879409211896
                Recall: 0.7015263090455042 ± 0.006276554603057393
                F1: 0.7113689850575797 ± 0.005145460905179358
                
    model_name = "nickprock/sentence-bert-base-italian-xxl-uncased" 
        Output con Deep Neaural Network:     
            Metrics con test set:
                accuracy: 0.7763157894736842
                precision: 0.7231833910034602
                recall: 0.7411347517730497
                f1: 0.7320490367775833
                
            Metriche cross validation:
                Accuracy: 0.7800162469536962 ± 0.01486751622126186
                Precision: 0.7420579305251603 ± 0.022944387947300568
                Recall: 0.6967848076563965 ± 0.02355621556768913
                F1: 0.7186221163673477 ± 0.021827567587715226

    model_name = "xlm-roberta-base", 1024 epochs
        Metrics con test set:
            accuracy: 0.7646198830409356
            precision: 0.728744939271255
            recall: 0.656934306569343
            f1: 0.6909788867562379
            
        Metriche cross validation:    
            Accuracy: 0.7567831031681559 ± 0.013205085800952802
            Precision: 0.6995359128840348 ± 0.035042433099694736
            Recall: 0.704077673835479 ± 0.02571905162313388
            F1: 0.7007740741192054 ± 0.01491529835221565

Process finished with exit code 0

        
Process finished with exit code 0

    """
