import numpy as np
import os
import scipy.io as sio
import h5py

# import DigitStructWrapper


DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
EXTRA_DIR = os.path.join(DATA_DIR, 'extra')


def load_cnn(channel_num=1, feat_norm=False):
    htr = h5py.File('data/train.h5', 'r')
    hts = h5py.File('data/test.h5', 'r')
    hte = h5py.File('data/extraTrain.h5', 'r')

    # Exploratory

    if channel_num == 1:
        digits = htr["digitsBW"]
        testdigits = hts["digitsBW"]
        negdigits = htr["negdigitsBW"]
        extdigits = hte["digitsBW"]
    else:
        digits = htr["digits"]
        testdigits = hts["digits"]
        negdigits = htr["negdigits"]
        extdigits = hte["digits"]

    trainlabs = htr["labs5"]
    testlabs = hts["labs5"]
    neglabs = htr["neglab"]
    extlabs = hte["labs5"]

    digits = digits[:]
    testdigits = testdigits[:]
    negdigits = negdigits[:]
    extdigits = extdigits[:]
    trainlabs = trainlabs[:]
    testlabs = testlabs[:]
    neglabs = neglabs[:]
    negdigits = negdigits[:]

    seed = 25
    np.random.seed(seed)
    countNeg = 30000
    countX = 90000

    negIdx = np.random.randint(0, negdigits.shape[0], countNeg)
    numNegTs = np.arange(negdigits.shape[0] - 500, negdigits.shape[0], 1)
    numtran = np.arange(0, countX, 1)  # Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) % Tried on 12K
    # numtest = np.arange(extdigits.shape[0] - 1,extdigits.shape[0],1)

    # negIdx = np.random.randint(0,negdigits.shape[0],3000)# Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(
    # 50%) Tried on 12k numtran = range(0,50,1)  # Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) %
    # Tried on 12K

    xtrdigits = extdigits[numtran, :]
    xtrlab = extlabs[numtran, :]

    ntrdigits = negdigits[negIdx]
    ntrlab = neglabs[negIdx, :]
    ntest = negdigits[numNegTs, :]
    ntslab = neglabs[numNegTs, :]

    preTrain = np.vstack((digits, xtrdigits, ntrdigits)).astype('float32')  #
    preTest = np.vstack((testdigits, ntest)).astype('float32')  # xtest

    # Lets remove digits > 4. Only 9 cases of n = 5
    trainlabs = np.vstack((trainlabs, xtrlab, ntrlab)).astype('uint8')  # xtrlab
    testlabs = np.vstack((testlabs, ntslab)).astype('uint8')  # xtslab
    ind = np.argwhere(trainlabs[:, 0] < 5)
    ind = ind[:, 0]
    preTrain = preTrain[ind, :]
    trainlabs = trainlabs[ind, :]

    nb = np.reshape(np.asarray(trainlabs[:, 0] > 0, dtype='uint8'), (trainlabs.shape[0], 1))
    trl = np.hstack((trainlabs, nb))

    ind = np.argwhere(testlabs[:, 0] < 5)
    ind = ind[:, 0]
    preTest = preTest[ind, :]
    testlabs = testlabs[ind, :]

    nb = np.reshape(np.asarray(testlabs[:, 0] > 0, dtype='uint8'), (testlabs.shape[0], 1))
    tsl = np.hstack((testlabs, nb))

    # train = np.float64(preTrain/255.)
    # test  = np.float64(preTest/255.)
    train = np.float64(preTrain)
    test = np.float64(preTest)

    for i in range(preTrain.shape[0]):
        if channel_num > 1:
            for channel in range(0, channel_num, 1):
                train[i][:, :, channel] -= np.mean(preTrain[i][:, :, channel].flatten(), axis=0)
        else:
            train[i] -= np.mean(preTrain[i].flatten(), axis=0)

    for i in range(preTest.shape[0]):
        if channel_num > 1:
            for channel in range(0, channel_num, 1):
                test[i][:, :, channel] -= np.mean(preTest[i][:, :, channel].flatten(), axis=0)
        else:
            test[i] -= np.mean(preTest[i].flatten(), axis=0)

    if feat_norm:
        M = np.mean(train, axis=0)
        train = train - M
        sd = np.std(train, axis=0)
        train = train / sd

        test = test - M
        test = test / sd
        featNorm = {'mean': M, 'std': sd}
        if channel_num > 1:
            with open('datasets/BGRnorm.pickle', 'wb') as handle:
                pickle.dump(featNorm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('datasets/BWnorm.pickle', 'wb') as handle:
                pickle.dump(featNorm, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # np.save('datasets/BWnorm.npz', featNorm)

    numtrain = train.shape[0]
    numtest = test.shape[0]
    row = train.shape[1]
    col = train.shape[2]

    train = np.reshape(train, (numtrain, row, col, channel_num))
    test = np.reshape(test, (numtest, row, col, channel_num))

    p = 0.9
    seed = 25
    np.random.seed(seed)
    split = np.int32(np.round((p * numtrain)))  # .85

    idx = np.random.permutation(numtrain)
    trIdx = idx[0:split]
    vlIdx = idx[split:numtrain]

    trlab = [np.reshape(trl[:, 0], (numtrain, 1)).astype('uint8'),
             np.reshape(trl[:, 1], (numtrain, 1)).astype('uint8'),
             np.reshape(trl[:, 2], (numtrain, 1)).astype('uint8'),
             np.reshape(trl[:, 3], (numtrain, 1)).astype('uint8'),
             np.reshape(trl[:, 4], (numtrain, 1)).astype('uint8'),
             np.reshape(trl[:, 6], (numtrain, 1)).astype('uint8')]

    tslab = [np.reshape(tsl[:, 0], (numtest, 1)).astype('uint8'),
             np.reshape(tsl[:, 1], (numtest, 1)).astype('uint8'),
             np.reshape(tsl[:, 2], (numtest, 1)).astype('uint8'),
             np.reshape(tsl[:, 3], (numtest, 1)).astype('uint8'),
             np.reshape(tsl[:, 4], (numtest, 1)).astype('uint8'),
             np.reshape(tsl[:, 6], (numtest, 1)).astype('uint8')]

    ctrlab = [trlab[0][trIdx], trlab[1][trIdx], trlab[2][trIdx], trlab[3][trIdx], trlab[4][trIdx], trlab[5][trIdx]]
    cvlab = [trlab[0][vlIdx], trlab[1][vlIdx], trlab[2][vlIdx], trlab[3][vlIdx], trlab[4][vlIdx], trlab[5][vlIdx]]
    ctslab = [tslab[0], tslab[1], tslab[2], tslab[3], tslab[4], tslab[5]]

    data = {'trainX': train[trIdx], 'trainY': ctrlab,
            'testX': test, 'testY': ctslab,
            'valdX': train[vlIdx], 'valdY': cvlab}

    return data


def load_data(path):
    data = sio.loadmat(path + '_32x32.mat')
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
    with h5py.File(name + ".hdf5", "w") as f:
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


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


if __name__ == "__main__":
    data = load_cnn(channel_num=3, feat_norm=True)
    # train_data = load_data(TRAIN_DIR)
    # test_data = load_data(TEST_DIR)
    extra_data = load_data(EXTRA_DIR)

    # train_images_normalized, train_labels = normalize_images(train_data)
    # save_data(train_images_normalized, train_labels, "SVHN_train")
    #
    # test_images_normalized, test_labels = normalize_images(test_data)
    # save_data(test_images_normalized, test_labels, "SVHN_test")

    extra_images_normalized, extra_labels = normalize_images(extra_data)
    save_data(extra_images_normalized, extra_labels, "SVHN_extra")
