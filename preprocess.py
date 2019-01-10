from PIL import Image
import os
from random import Random
import shutil


def generate_samples():
    img_src = Image.open("aerial.jpg")

    rows = img_src.height
    cols = img_src.width

    out_path = "data/raw"

    count = 0
    #iterate starting X
    for i in range(0, cols-399, 200):
        #iterate starting Y
        for j in range(0, rows-399, 200):
            img_out = img_src.crop((i, j, i+400, j+400))
            img_out.save("{}/{:05d}.jpg".format(out_path,count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{:05d}.jpg".format(out_path,count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{:05d}.jpg".format(out_path,count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{:05d}.jpg".format(out_path,count))
            count += 1

def split_sets():
    input_path = "data/raw/"

    train_path = "data/train_labels/"
    test_path  = "data/test_labels/"

    rand = Random()

    for file in os.listdir(input_path):
        if file.endswith(".jpg"):
            r = rand.random()
            if r <= .8:
                shutil.copyfile(input_path+file, train_path+file)
            else:
                shutil.copyfile(input_path+file, test_path+file)


def generate_dirty():
    input_train_path = "data/train_labels/"
    input_test_path = "data/test_labels/"

    output_train_path = "data/train/"
    output_test_path = "data/test/"

    rows, cols = (400,400)

    for file in os.listdir(input_train_path):
        img = Image.open(input_train_path+file)
        temp = img.resize((int(rows/2), int(cols/2)), Image.BILINEAR)
        temp = temp.resize((rows,cols), Image.BILINEAR)
        temp.save(output_train_path+file)

    for file in os.listdir(input_test_path):
        img = Image.open(input_test_path+file)
        temp = img.resize((int(rows/2), int(cols/2)), Image.BILINEAR)
        temp = temp.resize((rows,cols), Image.BILINEAR)
        temp.save(output_test_path+file)


generate_samples()
split_sets()
generate_dirty()
