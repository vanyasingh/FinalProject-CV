import numpy as np
import os

import scipy.io as sio
import h5py

DATA_DIR = "data2"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
EXTRA_DIR = os.path.join(DATA_DIR, 'extra')
MAX_LABELS = 6
NUM_LABELS = 11


def load_images(path):
    train_images = sio.loadmat(path+'/train_32x32.mat')
    test_images = sio.loadmat(path+'/test_32x32.mat')

    return train_images, test_images


def normalize_images(images):
    imgs = images["X"]
    imgs = np.transpose(imgs, (3, 0, 1, 2))

    labels = images["y"]
    # replace label "10" with label "0"
    labels[labels == 10] = 0

    # normalize images so pixel values are in range [0,1]
    scalar = 1 / 255.
    imgs = imgs * scalar

    return imgs, labels


def save_data(images, labels, name):
    with h5py.File(name+".hdf5", "w") as f:
        f.create_dataset("X", data=images, shape=images.shape, dtype='float32', compression="gzip")
        f.create_dataset("Y", data=labels, shape=labels.shape, dtype='int32', compression="gzip")


if __name__ == '__main__':
    train_images, test_images = load_images(DATA_DIR)

    # train_images_normalized, train_labels = normalize_images(train_images)
    # save_data(train_images_normalized, train_labels, "SVHN_train")

    test_images_normalized, test_labels = normalize_images(test_images)
    save_data(test_images_normalized, test_labels, "SVHN_test")