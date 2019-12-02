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

    # data = sio.loadmat(path+'_32x32.mat')
    # mat_data = sio.loadmat(os.path.join(path, 'digitStruct.mat'))
    mat_data = h5py.File(os.path.join(path, 'digitStruct.mat'))
    data = mat_data["digitStruct"]

    bbox = data['bbox']
    name = data['name']

    return name, bbox


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

        box = get_box_data(i, mat_data)

        x = int(min(box['top']))
        y = int(min(box['left']))
        h = int(max(box['height'])) + x
        w = int(sum(box['width'])) + y

        # crop and resize image
        crop_im = image[x:h, y:w, :]
        fin_im = cv2.resize(crop_im, dim, interpolation=cv2.INTER_AREA)
        fin_im = np.float32(fin_im) / 255.

        train_images.append(fin_im)

        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # mser = cv2.MSER()
        # regions = mser.detect(img, None)
        # print(regions)

        print(box['label'])
        pad = np.zeros(6)



    #
    #     labels = box['label'].squeeze()
    #     train_labels.append(np.int32(labels))
    #
    #
    # train_images = np.array(train_images)
    #
    # # print(train_images.shape)
    # train_labels = np.array(train_labels)
    # print(train_labels)
    # print(train_labels.shape)
    # train_bbox = np.array(train_bbox)

    # train_images = np.asarray(train_images)
    # train_images.astype('float32')

    # train_images /= 255

    return train_images, train_labels, train_bbox


def save_data(images, labels, bbox, name):
    print("SAVING DATA")
    hf = h5py.File(name+".h5", 'w')
    hf.create_dataset('X', data=images)
    hf.create_dataset('Y', data=labels)

    # with h5py.File(name+".hdf5", "w") as f:
    #     f.create_dataset("X", data=images, shape=images.shape,dtype='float32', compression="gzip")
    #     f.create_dataset("Y", data=labels, shape=labels.shape, dtype='int32', compression="gzip")
    #     # f.create_dataset("bbox", data=bbox, shape=bbox.shape, dtype='int32', compression="gzip")


if __name__ == "__main__":
    # image = cv2.imread(os.path.join(TRAIN_DIR, '1.png'))
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    train_images, train_labels, train_bbox = get_train_data(TRAIN_DIR)
    save_data(train_images, train_labels, train_bbox, "SVHN_train")

#
#     # save_data(images, labels)


