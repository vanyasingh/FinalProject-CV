import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import h5py


def prepare_data_detection():
    images = {}
    image_name = 'test1.png'
    image = cv2.imread(os.path.join('data2', image_name))
    image = image/255.
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    label = [2, 8, 0]
    images[image_name] = image




def predict(model, img_path, batch_size):
    model = load_model('weights.hdf5')
    model.summary()
    # normalize image pixel values into range [0,1]
    img_generator = image.ImageDataGenerator(preprocessing_function=lambda img: img/255.0)
    validation_generator = img_generator.flow_from_directory(directory=img_path, target_size=(32, 32), shuffle=False,
                                                             batch_size=batch_size, color_mode="rgb")

    score = model.evaluate_generator(validation_generator)
    print("Accuracy: {:.4f}".format(score[1]))
