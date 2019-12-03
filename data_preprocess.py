import numpy as np
import os
import scipy.io as sio
import h5py
import tqdm
import cv2

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
EXTRA_DIR = os.path.join(DATA_DIR, 'extra')
MAX_LABELS = 6
NUM_LABELS = 11


def get_box_data(index, hdf5_data):
    # print(index)
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def get_image(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def create_label_array(labels):

    num_digits = len(labels)
    labels_array = np.ones([MAX_LABELS], dtype=np.int32) * 10
    one_hot_labels = np.zeros((MAX_LABELS, NUM_LABELS), dtype=np.int32)

    for n in range(num_digits):
        if labels[n] == 10:
            labels[n] = 0
        labels_array[n] = labels[n]

    for n in range(len(labels_array)):
        one_hot_labels[n] = one_hot_encode(labels_array[n])

    return one_hot_labels


def one_hot_encode(number):
    """ Creates and returns a hot-hot-encoding representation of a given number
    Args:
        number: The number to be encoded
    Returns:
        The one-hot representation of the given number as a numpy array
    """
    one_hot = np.zeros(shape=NUM_LABELS, dtype=np.int32)
    one_hot[number] = 1

    return one_hot


def get_train_data(path):

    print("RUNNING PROCESS DATA")

    mat_data = h5py.File(path+'/digitStruct.mat')
    size = mat_data['/digitStruct/name'].size

    train_images = []
    train_labels = []

    dim = (64, 64)

    for i in tqdm.tqdm(range(10)):
        image_name = get_image(i, mat_data)
        image = cv2.imread(os.path.join(TRAIN_DIR, image_name))
        # height, width, ch = image.shape

        x = int(min(box['top']))
        y = int(min(box['left']))
        h = int(max(box['height'])) + x
        w = int(sum(box['width'])) + y

        box = get_box_data(i, mat_data)

        # crop and resize image
        crop_im = image[x:h, y:w, :]
        fin_im = cv2.resize(crop_im, dim, interpolation=cv2.INTER_AREA)
        fin_im = np.float32(fin_im) / 255.

        train_images.append(fin_im)

        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # mser = cv2.MSER()
        # regions = mser.detect(img, None)
        # print(regions)

        train_labels.append(create_label_array(box['label']))

    return train_images, train_labels


def save_data(images, labels, name):
    print("SAVING DATA")
    h5f = h5py.File(name+".h5", "w")
    h5f.create_dataset(name + "_dataset", data=images)
    h5f.create_dataset(name + "_labels", data=labels)


def load_data(name):
    """Loads an hdf5 file that contains the image data and labels
    Args:
        name: The name of the file to be loaded
    Returns:
        The data and labels arrays
    """
    h5f = h5py.File(name + ".h5", "r")
    data = h5f[name + "_dataset"][:]
    labels = h5f[name + "_labels"][:]

    return data, labels

def load_svhn_data(path, val_size):
    with h5py.File(path+'/SVHN_train.hdf5', 'r') as f:
        shape = f["X"].shape
        x_train = f["X"][:shape[0]-val_size]
        y_train = f["Y"][:shape[0]-val_size].flatten()
        x_val = f["X"][shape[0]-val_size:]
        y_val = f["Y"][shape[0] - val_size:].flatten()

    with h5py.File(path+'/SVHN_test.hdf5', 'r') as f:
        x_test = f["X"][:]
        y_test = f["Y"][:].flatten()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == "__main__":

    # train_images, train_labels = get_train_data(TRAIN_DIR)
    # save_data(train_images, train_labels, "train")

    test_images, test_labels = get_train_data(TEST_DIR)
    save_data(test_images, test_labels, "test")

#
#     # save_data(images, labels)


