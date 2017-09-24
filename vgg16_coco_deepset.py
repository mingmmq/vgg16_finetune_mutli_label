# -*- coding: utf-8 -*-
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1,floatX=float32"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Reshape, Activation, Conv2D
from keras import objectives

from sklearn.metrics import log_loss

from load_cifar10 import load_cifar10_data
from load_pascal_deepset import  load_pascal_data
from load_coco_deepset import load_coco_data
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
_EPSILON = K.epsilon()
import sklearn.metrics as skm
import pdb



def cus_acc(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def precision(y_true, y_pred):
    """Precision metric.		

    Only computes a batch-wise average of precision.		

    Computes the precision, a metric for multi-label classification of		
    how many selected items are relevant.		
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.		

    Only computes a batch-wise average of recall.		

    Computes the recall, a metric for multi-label classification of		
    how many relevant items are selected.		
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def acc(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def vgg16_model(img_rows, img_cols, channel=1, num_labels=None):
    """VGG 16 Model for Keras

    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('../models/vgg16_weights_th_dim_ordering_th_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    # for layer in model.layers[:25]:
    #     layer.trainable = False
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_labels, activation='sigmoid'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss=loss_function,
                  metrics=[acc, precision, recall, f1])

    return model


def _loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)

def _loss_tensor_bak(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    out = -(y_true * K.log(y_pred) + (1.0 - y_true)*K.log(1.0-y_pred))
    return K.mean(out, axis=-1)

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())

    #find the 1 labels sum in each row
    sum_of_each = K.round(K.sum(y_true, axis=1))
    #get 3 times of the to keep the data
    keep_of_each = sum_of_each * 3;
    #get the max of these number
    max = K.max(keep_of_each)
    #get the shape of y_true
    shape = K.shape(y_true)

    #generate the random tensor based on the shape, and make tie binomial
    random_tensor = K.random_binomial(shape=shape, p= (shape[1]-max)/(shape[1]))
    n_true = K.clip(y_true + random_tensor, K.epsilon(), 1.0-K.epsilon())

    out = -(y_true * K.log(y_pred)*left_weight + (1.0 - y_true) * K.log(1.0 - y_pred) * right_weight)
    return K.mean(out, axis=-1)


class My_Callback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, batch, logs=None):
        # pdb.set_trace()
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_pred = self.model.predict(x_val)

        #turn them into tensors
        y_true = K.variable(y_val)
        y_pred = K.variable(y_pred)

        #calculate the rate
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        recall = true_positives / possible_positives
        precision = true_positives / predicted_positives

        all_Ones = K.sum(K.random_binomial(shape=K.shape(y_true), p=1.0)) #this is to replace K.ones
        pred_positive_rate = predicted_positives / all_Ones
        true_negative = K.sum(all_Ones - predicted_positives - possible_positives + true_positives)
        accuracy = (true_positives + true_negative) /  all_Ones

        loss_original = _loss_tensor_bak(y_true, y_pred)
        loss_now = _loss_tensor(y_true, y_pred)


        # find the 1 labels sum in each row
        sum_of_each = K.round(K.sum(y_true, axis=1))
        # get 3 times of the to keep the data
        keep_of_each = sum_of_each * 3;
        # get the max of these number
        max = K.max(keep_of_each)
        # get the shape of y_true
        shape = K.shape(y_true)
        # generate the random tensor based on the shape, and make tie binomial
        random_tensor = K.random_binomial(shape=shape, p=(shape[1] - max) / (shape[1]))
        n_true = K.clip(y_true + random_tensor, K.epsilon(), 1.0 - K.epsilon())
        sum_of_n_true = K.sum(K.round(n_true))
        sum_of_y_pred = K.sum(K.round(y_pred))


        #print related infromation
        print("\nepoch end: positive rate: %f, precision: %f, recall: %f, accuracy: %f, loss original: %f, loss_now: %f, n_true_sum: %f, y_pred_sum: %f \n"
              %(K.eval(pred_positive_rate), K.eval(precision), K.eval(recall), K.eval(accuracy), K.eval(K.mean(loss_original)), K.eval(K.mean(loss_now)) ,K.eval(sum_of_n_true), K.eval(sum_of_y_pred)))
        return

def parse_arguments():
    import argparse
    global learning_rate
    global grids_per_row
    global nb_epoch
    global left_weight
    global right_weight
    global pascal_version
    global loss_function

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate')
    parser.add_argument('--grid', help="grid per row and column")
    parser.add_argument('--epochs', help="number of epochs")
    parser.add_argument('--lw', help="left weight on the loss function")
    parser.add_argument('--rw', help="right weight on the loss function")
    parser.add_argument('--pv', help="pascal version")
    parser.add_argument('--lf', help="loss function")
    args = parser.parse_args()


    learning_rate = float(args.lr) if args.lr else 0.01
    grids_per_row = args.grid if args.grid else 7
    nb_epoch = args.epochs if args.epochs else 60
    left_weight = args.lw if args.lw else 1
    right_weight = args.rw if args.rw else 1
    pascal_version = args.pv if args.pv else "VOC2007"
    loss_function = _loss_tensor if args.lf else "binary_crossentropy"

if __name__ == '__main__':
    parse_arguments()
    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_labels = 90 * grids_per_row * grids_per_row
    batch_size = 16 


    image_path = "../coco/"
    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    # X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    X_train, Y_train, X_valid, Y_valid = load_coco_data(image_path)


    # Load our model
    model = vgg16_model(img_rows, img_cols, channel, num_labels)

    my_callback = My_Callback(validation_data=(X_valid, Y_valid))

    # Start Fine-tuning
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        shuffle=True,
                        verbose=1,
                        validation_data=(X_valid, Y_valid),
                        callbacks=[my_callback]
              )

    model.save_weights('trained_model.h5')

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    print(predictions_valid)


    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    print(score)

    # # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.show()
    plt.savefig('losses.png')
    plt.clf()

    print(history.history)


    plt.plot(history.history['precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['f1'])
    plt.title('train precision recall')
    plt.ylabel('score')
    plt.legend(['precision', 'recall', 'f1'], loc='upper left')
    plt.show()
    plt.savefig('train_precision_recall.png')
    plt.clf()

    plt.plot(history.history['val_precision'])
    plt.plot(history.history['val_recall'])
    plt.plot(history.history['val_f1'])
    plt.title('val precision recall')
    plt.ylabel('score')
    plt.legend(['precision', 'recall', 'f1'], loc='upper left')
    plt.show()
    plt.savefig('val_precision_recall.png')
    plt.clf()
