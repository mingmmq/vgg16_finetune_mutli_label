from __future__ import absolute_import
from keras.preprocessing import image
import get_pascal_data_list as pascal_dict
import numpy as np
import cv2

def load_data(version=""):
    if version == "":
        print("data_path required: VOC2012 or VOC2007")
        exit()
    path = "/".join(["../pascal/VOCdevkit", version])
    (x_train, y_train) = load_data_by_type(path, "train")
    (x_test, y_test)  = load_data_by_type(path, "val")

    return (x_train, y_train), (x_test, y_test)

def load_data_by_type(path, type):
    #this is used the set the grid line numbers, and the rejected images are listed by another program
    grid_rows = 7
    if type == "train":
        data = pascal_dict.getImageAndAnnotations(path, '_train.txt', grid_rows)
    else:
        data = pascal_dict.getImageAndAnnotations(path, '_val.txt', grid_rows)

    num_train_samples = 0
    for key in data:
        num_train_samples += 1

    #todo: why should we resize to 244 244 with 3 channels
    x_train = np.zeros((num_train_samples, 3, 224, 224), dtype='uint8')
    i = 0
    labels = []
    image_names = []

    # In what order will the key be iterated? the order in linux is different from in macos
    files = []
    for key in data:
        image_path = "/".join([path, "JPEGImages", key+".jpg"])
        image_names.append(key)
        img = image.load_img(image_path)
        d = image.img_to_array(img, data_format="channels_first").astype(dtype="uint8")
        dr = cv2.resize(d.transpose(1, 2, 0), (224, 224)).transpose(2, 0, 1)
        x_train[i,:,:,:] = dr

        labels.append(data[key])
        files.append(key)

        if i + 1 == num_train_samples:
            break
        i += 1

    y_train = to_categoricals(path, labels, 20, grid_rows)

    print("samples counts:", num_train_samples)
    return (x_train, y_train)



def to_categoricals(path, y, num_classes, grid_rows):
    cat_dict = pascal_dict.getCategoryDict('/'.join([path, "ImageSets/Main"]))
    print(cat_dict)

    n = len(y)
    categorical = np.zeros((n, num_classes * grid_rows * grid_rows)).astype('float64')
    for i in range(0, n):
        # print("\n", files[i], y[i])
        for key in y[i]:
            lists = [int(j + grid_rows * grid_rows * cat_dict[key]) for j in y[i][key]]
            # print(key, lists)
            categorical[i,  lists] = 1

    return categorical


if __name__ == "__main__":
    load_data()

