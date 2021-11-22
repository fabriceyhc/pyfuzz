#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/9/21 6:39 PM
# @Author  : Jiyuan Wang
# @File    : minist.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import sys


def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim = num_pixels,
                  activation = 'relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=opt)
    return model

def set_up():
    global num_of_classes
    num_of_classes = 10

    sys.path.append("/data")

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape[0])

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    X_train = X_train / 255
    X_test = X_test / 255

    global num_pixels
    num_pixels = 784
    X_train = X_train.reshape(X_train.shape[0],
                              num_pixels)
    X_test = X_test.reshape(X_test.shape[0],
                            num_pixels)
    print(X_train.shape)

    model = create_model()
    print(model.summary())

    history = model.fit(X_train, y_train, validation_split=0.1,
                        epochs=10, batch_size=200, verbose=1, shuffle=1)

    score = model.evaluate(X_test, y_test, verbose=0)
    print(type(score))
    print('Test Score:', score[0])
    print('Test Accuracy:', score[1])
    return model

def predict(img_array, mnist_model):
    prediction_pro = mnist_model.predict(img_array)
    print("probability:",prediction_pro)
    print("predicted digit:", np.argmax(prediction_pro,axis=1))
    return prediction_pro


if __name__ == '__main__':

    model = set_up()

    import requests
    from PIL import Image

    url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    plt.imshow(img)

    import cv2

    img_array = np.asarray(img)
    resized = cv2.resize(img_array, (28, 28))
    gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # (28, 28)
    image = cv2.bitwise_not(gray_scale)

    image = image / 255
    image = image.reshape(1, 784)
    prediction = predict(image,mnist_model=model)