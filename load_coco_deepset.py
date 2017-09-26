import cv2
import numpy as np

import coco_deep_set


nb_train_samples = 3000 # 3000 training samples
nb_valid_samples = 100 # 100 validation samples
num_classes = 20

def load_coco_data(image_path, grid_rows):
    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = coco_deep_set.load_data(image_path, grid_rows)

    # For Theano
    # # Switch RGB to BGR order
    # X_train = X_train[:, ::-1, :, :]
    # X_valid = X_valid[:, ::-1, :, :]
    #
    # # Subtract ImageNet mean pixel
    # X_train[:, 0, :, :] -= 103.939
    # X_train[:, 1, :, :] -= 116.779
    # X_train[:, 2, :, :] -= 123.68
    #
    #
    #
    # X_valid[:, 0, :, :] -= 103.939
    # X_valid[:, 1, :, :] -= 116.779
    # X_valid[:, 2, :, :] -= 123.68

    return X_train, Y_train, X_valid, Y_valid
