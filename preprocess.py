import os
import shutil
from pathlib import Path, PurePath

from PIL import Image
from sklearn.model_selection import train_test_split

from util import RAW_PATH, TRAIN_PATH, TEST_PATH, clean_mkdir, ROWS, COLS, IMAGES_PATH


def generate_samples(image_path, data_path):
    """
    Split up a larger image into correctly sized chucks for the model.

    Args:
        image_path:
        data_path:

    """
    output_path = str(Path(data_path) / RAW_PATH)
    clean_mkdir(output_path)
    filename = PurePath(image_path).stem
    img_src = Image.open(image_path)

    rows = img_src.height
    cols = img_src.width

    count = 0
    # iterate starting X
    for i in range(0, cols - COLS - 1, COLS / 2):
        # iterate starting Y
        for j in range(0, rows - ROWS - 1, ROWS / 2):
            img_out = img_src.crop((i, j, i + ROWS, j + COLS))
            img_out.save("{}/{}_{:05d}.jpg".format(output_path, filename, count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{}_{:05d}.jpg".format(output_path, filename, count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{}_{:05d}.jpg".format(output_path, filename, count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{}_{:05d}.jpg".format(output_path, filename, count))
            count += 1


def split_sets(data_path: Path):
    input_path = data_path / RAW_PATH

    train_path = data_path / TRAIN_PATH
    test_path = data_path / TEST_PATH

    clean_mkdir(str(train_path))
    clean_mkdir(str(test_path))

    filenames = []
    for filename in os.listdir(str(input_path)):
        if filename.endswith(".jpg"):
            filenames.append(filename)

    train_filenames, test_filenames = train_test_split(
        filenames, train_size=0.8, test_size=0.2
    )

    for filename in train_filenames:
        shutil.copyfile(str(input_path / filename), str(train_path / filename))

    for filename in test_filenames:
        shutil.copyfile(str(input_path / filename), str(test_path / filename))


def generate_dirty(data_path: Path):
    """
    Generate the X values by downsampling clean imagery.

    Args:
        data_path:

    """
    input_train_path = data_path / "train_labels"
    input_test_path = data_path / "test_labels"

    output_train_path = data_path / "train"
    output_test_path = data_path / "test"

    clean_mkdir(str(output_train_path))
    clean_mkdir(str(output_test_path))

    for file in os.listdir(str(input_train_path)):
        img = Image.open(str(input_train_path / file))
        temp = img.resize((int(ROWS / 2), int(COLS / 2)), Image.BILINEAR)
        temp = temp.resize((ROWS, COLS), Image.BILINEAR)
        temp.save(str(output_train_path / file))

    for file in os.listdir(str(input_test_path)):
        img = Image.open(str(input_test_path / file))
        temp = img.resize((int(ROWS / 2), int(COLS / 2)), Image.BILINEAR)
        temp = temp.resize((ROWS, COLS), Image.BILINEAR)
        temp.save(str(output_test_path / file))


def preprocess_dataset(data_path):
    data_path = Path(data_path)
    images_path = data_path / IMAGES_PATH
    for filename in os.listdir(str(images_path)):
        generate_samples(str(images_path / filename), data_path)

    split_sets(data_path)
    generate_dirty(data_path)
