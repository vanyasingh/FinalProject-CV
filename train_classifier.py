import os
import sys
import numpy as np
import h5py
import cv2

import tensorflow as tf
import h5py
import os
from tensorflow.keras import layers, models, optimizers, callbacks, backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing import image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_data(val_size):
    with h5py.File('SVHN_train.hdf5', 'r') as f:
        shape = f["X"].shape
        x_train = f["X"][:shape[0]-val_size]
        y_train = f["Y"][:shape[0]-val_size].flatten()
        x_val = f["X"][shape[0]-val_size:]
        y_val = f["Y"][shape[0] - val_size:].flatten()

    with h5py.File('SVHN_test.hdf5', 'r') as f:
        x_test = f["X"][:]
        y_test = f["Y"][:].flatten()

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_cnn_model():
    model = Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    #     model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    #     model.add(layers.BatchNormalization())
    #     model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    #     model.add(layers.MaxPooling2D((2, 2)))
    #     model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))

    #     lr = 0.0001
    #     opt = {"SGD": optimizers.SGD(lr=lr), "Adam": optimizers.Adam(lr=lr)}

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


def train_custom_cnn():
    train_data, val_data, test_data = load_data(val_size=10000)

    model = build_cnn_model()

    save_model = callbacks.ModelCheckpoint("weights.hdf5",
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False)

    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy',
                                             min_delta=0,
                                             patience=3,
                                             verbose=0,
                                             mode='max')
    tensorboard = callbacks.TensorBoard(histogram_freq=10,
                                        batch_size=32,
                                        write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None)

    print("TRAINING MODEL")
    history = model.fit(train_data[0], train_data[1],
                        batch_size=100,
                        epochs=50,
                        validation_data=val_data,
                        callbacks=[early_stopping, save_model])

    print("DONE TRAINING MODEL")
    print("EVALUATING MODEL")
    model = models.load_model("weights.hdf5")
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    print('Test loss: {:.4f}'.format(score[0]))
    print('Test accuracy: {:.4f}'.format(score[1]))
    print("DONE")


def predict(model, img_path, batch_size):
    model = models.load_model('weights.hdf5')
    # normalize image pixel values into range [0,1]
    img_generator = image.ImageDataGenerator(preprocessing_function=lambda img: img/255.0)
    validation_generator = img_generator.flow_from_directory(directory=img_path, target_size=(32, 32), shuffle=False,
                                                             batch_size=batch_size, color_mode="rgb")

    score = model.evaluate_generator(validation_generator)
    print("Accuracy: {:.4f}".format(score[1]))


if __name__ == "__main__":
    train_custom_cnn()
    # prepare_data_detection()