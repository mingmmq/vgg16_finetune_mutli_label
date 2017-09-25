from __future__ import absolute_import
from keras.preprocessing import image
import get_pascal_data_list as pascal_dict
import numpy as np
from keras.utils import np_utils
import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def show_image(image_data, lables=""):
    arr = np.ascontiguousarray(image_data.transpose(1, 2, 0))
    img = Image.fromarray(arr, 'RGB')
    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 10)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), lables, (255, 255, 0), font=font)
    img.show()

def load_data(data_path=""):
    if data_path == "":
        print("data_path required: VOC2012 or VOC 2007")
        exit()
    path = "/".join(["../pascal/VOCdevkit",data_path])
    (x_train, y_train) = load_data_by_type(path, "train")
    (x_test, y_test)  = load_data_by_type(path, "val")

    return (x_train, y_train), (x_test, y_test)

def load_data_by_type(path, type):
    if type == "train":
        data, files = pascal_dict.getImageAndLabels(path, '_train.txt')
    else:
        data, files = pascal_dict.getImageAndLabels(path, '_val.txt')

    num_train_samples = 0
    for key in data:
        # #next two lines are used for filter out those only with one label
        # if len(data[key]) > 1:
        #     continue
        if num_train_samples == 16:
            break
        #
        num_train_samples += 1
    # num_train_samples = len(data.keys())
    # num_train_samples = 32

    x_train = np.zeros((num_train_samples, 3, 224, 224), dtype='uint8')
    # y_train = np.zeros((num_train_samples, 20), dtype='uint8')
    i = 0
    labels = []
    image_names = []

    # In what order will the key be iterated? the order in linux is different from in macos
    for key in data:
        # #for one label only image for testing
        # if len(data[key]) > 1:
        #     continue

        image_names.append(key)
        img = image.load_img(key)
        d = image.img_to_array(img, data_format="channels_first").astype(dtype="uint8")
        dr = cv2.resize(d.transpose(1, 2, 0), (224, 224)).transpose(2, 0, 1)
        x_train[i,:,:,:] = dr

        # from PIL import Image
        # di = cv2.resize(d.transpose(1, 2, 0), (224, 224))
        # img = Image.fromarray(di, 'RGB')
        # img.save('my.png')
        # img.show()
        # print(data[key])

        # while len(data[key]) < num_classes:
        #     data[key].append(data[key][0])

        # y_train[i,:] = data[key]
        labels.append(data[key])


        if i + 1 == num_train_samples:
            break
        i += 1


    for j in range(len(image_names)):
        lns  = ""
        for l in labels[j]:
            lns += files[l]
        # print(image_names[j] + " " + str(labels[j]) + "" + lns )


    y_train = to_categoricals(labels, 20)


    # arr = np.ascontiguousarray(x_train[0].transpose(1, 2, 0))
    # img = Image.fromarray(arr, 'RGB')
    # img.show()

    print("samples counts:", num_train_samples)
    return (x_train, y_train)



def to_categoricals(y, num_classes):
    # y = np.array(y, dtype='int')
    # if not num_classes:
    #     num_classes = np.max(y) + 1
    n = len(y)
    categorical = np.zeros((n, num_classes)).astype('float64')
    for i in range(0, n):
        # categorical[i, y[i]] = 1
        categorical[i, y[i][0]] = 1
    return categorical


if __name__ == "__main__":
    load_data()

