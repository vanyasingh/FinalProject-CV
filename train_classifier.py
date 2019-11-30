from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt≤


def custom_cnn():
    # load data

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

    # model.add(Conv2D(filters=32, kernel_size=5, strides=5, border_mode='same',
    #                  input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(Conv2D(32, 5, 5, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.20))
    #
    # model.add(Conv2D(64, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.20))
    #
    # model.add(Conv2D(64, 1, 1, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, 1, 1, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.20))
    #
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(num_classes))
    # model.add(Activation('softmax'))

    #  Stochastic Gradient Descent
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # history = model.fit(train_images, train_labels, epochs=10,
    #                     validation_data=(test_images, test_labels))