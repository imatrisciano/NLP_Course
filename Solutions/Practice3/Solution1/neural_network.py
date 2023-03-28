from keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras


def train_deep_neural_network(x_train, y_train, input_size, plot=False):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

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

    history = model.fit(x_train, y_train, epochs=128, batch_size=16384, verbose=1, validation_data=(x_val, y_val))

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

    return model
