from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
from tensorflow.keras import layers, models, optimizers, callbacks
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

input_shape = (64, 64, 3)


def build_custom_cnn():

    # construct model
    model = models.Sequential()

    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=optimizer,
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

    save_model = callbacks.ModelCheckpoint("weights.hdf5",
                                           monitor='val_acc',
                                           mode='max',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           period=1)

    early_stopping = callbacks.EarlyStopping(monitor='val_acc',
                                             min_delta=0,
                                             patience=5,
                                             verbose=0,
                                             mode='max')

    tensorboard = callbacks.TensorBoard(histogram_freq=10,
                                        batch_size=32,
                                        write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None)

    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical')

    val_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            class_mode='categorical')

    history = model.fit(train_images, train_labels,
                        batch_size=batch_size,
                        epochs=10,
                        validation_data=(test_images, test_labels),
                        callbacks=[early_stopping, save_model, tensorboard])

    # model.fit(train_data[0], train_data[1],
    #           batch_size=FLAGS.batch_size,
    #           epochs=FLAGS.epochs,
    #           verbose=1,
    #           validation_data=val_data,
    #           callbacks=[early_stopping, save_model, tensorboard])
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_loss, test_acc)


if __name__ == "__main__":
    train_custom_cnn()