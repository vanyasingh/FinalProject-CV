import cv2
import numpy as np
import scipy.io as sio
import os
import h5py

# import matplotlib.pyplot as plt

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
EXTRA_DIR = os.path.join(DATA_DIR, 'extra')


def load_data_cnn(directory, dataset_name):

    mat_data = sio.loadmat(os.path.join(directory, 'trainDigits.mat'))
    data = mat_data["digitStruct"]

    bbox = data['bbox'].squeeze()
    name = data['name'].squeeze()
