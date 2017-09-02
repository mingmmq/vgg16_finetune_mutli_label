import cv2
import numpy as np

import pascal2012
from keras import backend as K

nb_train_samples = 3000 # 3000 training samples
nb_valid_samples = 100 # 100 validation samples
num_classes = 20

def load_pascal2012_data(img_rows, img_cols):
    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = pascal2012.load_data()

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

    # # Resize trainging images, th means theano / else is the tensorflow
    # if K.image_dim_ordering() == 'th':
    #   X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
    #   X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    # else:
    #   X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
    #   X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])
    #
    # # Transform targets to keras compatible format
    # Y_train = to_categoricals(Y_train[:nb_train_samples], num_classes)
    # Y_valid = to_categoricals(Y_valid[:nb_valid_samples], num_classes)
    return X_train, Y_train, X_valid, Y_valid
