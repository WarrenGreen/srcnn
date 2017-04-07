from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint

import numpy as np
import os
from PIL import Image

BIT_DEPTH = 8
MAX_VAL = 2**8-1

TEST_PATH = "data/test/"
TEST_LABELS_PATH = "data/test_labels/"

def load_data(xPath, yPath):
    X, Y = np.zeros((1876,400,400,3)), np.zeros((1876,400,400,3))
    index = 0
    for file in os.listdir(xPath):
        index += 1

        img = Image.open(xPath + file)
        imgArray = np.asarray(img, dtype='uint8')
        imgArray =  imgArray / (MAX_VAL * 1.0)
        X[index] = imgArray

        img = Image.open(yPath + file)
        imgArray = np.asarray(img, dtype='uint8')
        imgArray =  imgArray / (MAX_VAL * 1.0)
        Y[index] = imgArray

    return X, Y

def get_model():
    model = Sequential()
    model.add(Convolution2D(32, 9, activation="relu", input_shape=(400,400,3), padding="same"))
    model.add(Convolution2D(16, 5, activation="relu", padding="same"))
    model.add(Convolution2D(3, 5, activation="relu", padding="same"))
    model.load_weights("models/weights.h5")
    model.compile(optimizer="adam", loss="mse")

    return model

model = get_model()
X,Y = load_data(TEST_PATH, TEST_LABELS_PATH)
score = model.evaluate(X, Y)
print("{}: {:.2f}".format(model.metrics_names[1], score[1]*100))


