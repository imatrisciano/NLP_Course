from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
"""