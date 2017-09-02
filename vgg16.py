# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss

from load_cifar10 import load_cifar10_data
from load_pascal2012 import load_pascal2012_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
import sklearn.metrics as skm

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
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
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('imagenet_models/vgg16_weights_th_dim_ordering_th_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    for layer in model.layers: layer.trainable = False
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='sigmoid'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy',precision, recall, f1])

    return model

class Metrics(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.precisions = []
        self.recalls = []
        self.count  = 0


    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.precisions.append(logs.get('precession'))
        self.recalls.append(logs.get('recall'))

        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]

        print(y_pred[0])
        print(y_true[0])

        precision, recall, threshold = skm.precision_recall_curve(y_true.flatten(), y_pred.flatten().round())

        print(precision)
        print(recall)
        print(threshold)

        self.precisions.append(precision)
        self.recalls.append(recall)

        #
        # precision, recall, thresholds = skm.precision_recall_curve(y_true, y_pred)

        plt.plot(precision, recall)
        plt.title('Precision Recall epoch %d'%self.count)
        plt.ylabel('loss')
        plt.xlabel('recall')
        plt.show()
        plt.savefig('precision_recall %d .png'%self.count)

        print(self.precisions)
        print(self.recalls)

        return



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
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((pre*rec)/(pre+rec))

def show_image(image_data, lables=""):
    arr = np.ascontiguousarray(image_data.transpose(1, 2, 0))
    img = Image.fromarray(arr, 'RGB')
    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 10)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), lables, (255, 255, 0), font=font)
    img.show()


if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 20
    batch_size = 16 
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    # X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    X_train, Y_train, X_valid, Y_valid = load_pascal2012_data(img_rows, img_cols)

    i = 0
    for img in X_train[:4]:
        show_image(img, "".join(str(Y_train[i])))
        i += 1

    i = 2000
    for img in X_valid[2000:2010]:
        show_image(img, "".join(str(Y_valid[i])))
        i += 1

    # Load our model
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

    metrics = Metrics()

    # Start Fine-tuning
    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              callbacks=[metrics],
              )

    model.save_weights('trained_model.h5')

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    print(score)

    # # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('loss epoch.png')

