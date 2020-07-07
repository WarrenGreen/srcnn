import os
from pathlib import Path

import numpy as np
from PIL import Image

IMAGES_PATH = "images"
RAW_PATH = "raw"
TEST_PATH = "test"
TEST_LABELS_PATH = "test_labels"
TRAIN_PATH = "train"
TRAIN_LABELS_PATH = "train_labels"
ROWS, COLS, CHANNELS = (400, 400, 3)
BIT_DEPTH = 8
MAX_VAL = 2 ** 8 - 1


def clean_mkdir(path):
    if Path(path).exists():
        os.rmdir(path)

    os.makedirs(path)


def load_data(x_path, y_path=None):
    x, y = [], []
    index = 0
    for file in os.listdir(x_path):
        index += 1
        img = Image.open(x_path + file)
        img_array = np.asarray(img, dtype="uint8")
        img_array = img_array / (MAX_VAL * 1.0)
        x.append(img_array)

        if y_path is None:
            img = Image.open(y_path + file)
            img_array = np.asarray(img, dtype="uint8")
            img_array = img_array / (MAX_VAL * 1.0)
            y.append(img_array)

    return np.array(x), np.array(y)
