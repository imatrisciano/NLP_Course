import pandas as pd
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from sklearn.model_selection import cross_validate


def create_model(input_size):
    model = keras.Sequential(
        [
            layers.Dense(512, input_dim=input_size, kernel_initializer='normal', activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]
    )

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

def train_deep_neural_network(x_train, y_train, input_size, plot=False):
    params = {"input_size": input_size}
    model = KerasClassifier(build_fn=create_model, **params)
    param_grid = {"epochs": [256], "batch_size": [8192]}
    grid = GridSearchCV(model, param_grid, cv=5,
                        scoring=("accuracy", "precision", "recall", "f1"),
                        refit="f1")
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    results = pd.DataFrame(grid.cv_results_)  # results contains all the metrics for every parameter combination
    best_model_metrics = results.iloc[[grid.best_index_]]  # select only the row of the best classifier


    """
    if plot:
        print(model.summary())
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model's Training & Validation loss across epochs")
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
    """

    return best_model, best_model_metrics
