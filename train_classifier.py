from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import data_preprocess

# import matplotlib.pyplot as plt

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
EXTRA_DIR = os.path.join(DATA_DIR, 'extra')

batch_size = 128
epochs = 15
IMG_HEIGHT = 64
IMG_WIDTH = 64
learning_rate = 0.0001


def build_custom_cnn():

    # construct model
    model = models.Sequential()

    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # lr = learning_rate
    # optimizers = {"SGD": optimizers.SGD(lr=lr), "RMSprop": optimizers.RMSprop(lr=lr),
    #               "Adadelta": optimizers.Adadelta(lr=lr), "Adam": optimizers.Adam(lr=lr)}
    #
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


def train_custom_cnn():

    # load train data
    train_images, train_labels = data_preprocess.load_data('train')
    # load test data
    test_images, test_labels = data_preprocess.load_data('test')

    model = build_custom_cnn()

    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical')

    val_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='categorical')

    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)


if __name__ == "__main__":
    build_custom_cnn()