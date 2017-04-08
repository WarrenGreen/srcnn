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

def load_data(xPath, yPath, filename):
    X, Y = np.zeros((1,400,400,3)), np.zeros((1,400,400,3))
    index=0
    img = Image.open(xPath + filename)
    imgArray = np.asarray(img, dtype='uint8')
    imgArray =  imgArray / (MAX_VAL * 1.0)
    X[index] = imgArray

    img = Image.open(yPath + filename)
    imgArray = np.asarray(img, dtype='uint8')
    imgArray =  imgArray / (MAX_VAL * 1.0)
    Y[index] = imgArray

    return X, Y

def get_model():
    model = Sequential()
    model.add(Convolution2D(32, 9, activation="relu", input_shape=(400,400,3), padding="same"))
    model.add(Convolution2D(16, 5, activation="relu", padding="same"))
    model.add(Convolution2D(3, 5, activation="relu", padding="same"))
    model.load_weights("models/weights2.h5")
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    return model

filename = "05454.jpg"
model = get_model()
X,Y = load_data(TEST_PATH, TEST_LABELS_PATH,filename)
out_array = model.predict(X)
print (out_array.max())
num, rows, cols, channels = out_array.shape
for i in range(rows):
	for j in range(cols):
		for k in range(channels):
			if out_array[0][i][j][k] > 1.0:
				out_array[0][i][j][k] = 1.0

out_img = Image.fromarray(np.uint8(out_array[0] * 255))
out_img.save("out_"+filename)

