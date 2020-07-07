from keras import Sequential
from keras.layers import Convolution2D


def get_model(weights_path=None):
    model = Sequential()
    model.add(
        Convolution2D(
            32, 9, activation="relu", input_shape=(400, 400, 3), padding="same"
        )
    )
    model.add(Convolution2D(16, 5, activation="relu", padding="same"))
    model.add(Convolution2D(3, 5, activation="relu", padding="same"))
    if weights_path:
        model.load_weights(weights_path)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model
