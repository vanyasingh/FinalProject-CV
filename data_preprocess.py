import numpy as np
import os
import scipy.io as sio
import h5py
import tqdm
import cv2
# import DigitStructWrapper


DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
EXTRA_DIR = os.path.join(DATA_DIR, 'extra')


def load_data(path):

    data = sio.loadmat(path+'_32x32.mat')
    return data


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


def get_box_data(index, hdf5_data):

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


def get_processed_data(path):

    mat_data = h5py.File(path+'/digitStruct.mat')
    size = mat_data['/digitStruct/name'].size

    print(size)

    train_images = []
    train_labels = []
    train_bbox = []

    for _i in tqdm.tqdm(range(size)):
        image_name = get_image(_i, mat_data)
        train_images.append(cv2.imread(os.path.join(TRAIN_DIR, image_name)))

        box = get_box_data(_i, mat_data)
        train_bbox.append(box)

        labels = box['label']
        train_labels.append(labels)

    train_images.astype('float32')

    return train_images, train_labels, train_bbox



# if __name__ == "__main__":
#
#     # train_data = load_data(TRAIN_DIR)
#     # test_data = load_data(TEST_DIR)
#     # extra_data = load_data(EXTRA_DIR)
#     #
#     # # train_images_normalized, train_labels = normalize_images(train_data)
#     # # save_data(train_images_normalized, train_labels, "SVHN_train")
#     # #
#     # # test_images_normalized, test_labels = normalize_images(test_data)
#     # # save_data(test_images_normalized, test_labels, "SVHN_test")
#     #
#     # extra_images_normalized, extra_labels = normalize_images(extra_data)
#     # save_data(extra_images_normalized, extra_labels, "SVHN_extra")
#
#     # mat_data = h5py.File(TRAIN_DIR+'/digitStruct.mat')
#     # size = mat_data['/digitStruct/name'].size
#     #
#     # print(size)
#     #
#     # train_images = []
#     # train_labels = []
#     # train_bbox = []
#     #
#     # for _i in tqdm.tqdm(range(size)):
#     #     image_name = get_image(_i, mat_data)
#     #     box = get_box_data(_i, mat_data)
#     #     train_bbox.append(box)
#     #     train_images.append(cv2.imread(os.path.join(TRAIN_DIR, image_name)))
#     #     labels = box['label']
#     #     train_labels.append(labels)
#
#     train_images, train_labels, train_bbox = get_processed_data(TRAIN_DIR)
#
#     # save_data(images, labels)
#

