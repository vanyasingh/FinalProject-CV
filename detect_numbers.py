import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_DIR = "data2"
OUT_DIR =  "result"


def predict(img, model):
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0)
    # img = img.reshape((img.shape[2],img.shape[1],img.shape[0]))
    p = model.predict_proba(img)
    print(p)
    y_classes = p.argmax(axis=-1)
    m = np.max(p[0])
    v = p.argmax(axis=-1)
    return m, v[0]

if __name__ == "__main__":

    #  Load the CNN models for detection and classification
    model = load_model("weights.hdf5")
    #  Load in the image of interest and convert to grayscale
    INPUT_DIR = "data2"
    OUTPUT_DIR = "result"

    file_list = ["test1","test2","test3","test4","test5","test6","test7","test8"]

    for f in file_list:
        #  configure read and write names
        read_filename = INPUT_DIR + "/" + f + ".png"
        write_filename = OUTPUT_DIR + "/" + f + ".png"
        vis = cv2.imread(read_filename)
        vis = vis / 255.
        vis = vis.astype(np.float32)
        m, v = predict(vis, model)
        print(m, v)


        # # print(vis.dtype)
        # img = vis.copy()
        # img = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        #
        # if f == "dark":
        #     img = cv2.medianBlur(img, 9)
        # else:
        #     img = cv2.medianBlur(img, 7)
        #
        # mser = cv2.MSER_create()
        # msers, regions = mser.detectRegions(img)
        # print(regions)
        #
        # #  Get the bounding boxes for the regions of interest
        # bbs = list()
        # hulls = list()
        # for i, region in enumerate(regions):
        #     (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        #
        #     #  Rule: no wide rectangles
        #     if w > 1.25*h:
        #         continue
        #
        #     #  Rule: no rectangles that are too long
        #     if h > 3*w:
        #         h = 3 * w
        #
        #     bb = ((y, y+h, x, x+w))
        #     crop_im = vis[bb[0]:bb[1], bb[2]:bb[3]]
        #     crop_im = crop_im / 255.
        #     crop_im = crop_im.astype(np.float32)
        #
        #     m, v = predict(crop_im, class_model)
        #     print(m, v)
        #     bbs.append(bb)
        #     hull = cv2.convexHull(region.reshape(-1, 1, 2))
        #     hulls.append(hull)
        #
        # #  Generate some stats on the hulls/bbs
        # hull_length = list()
        # for h in hulls:
        #     hull_length.append(len(h))
        # mean_hull = np.mean(hull_length)
        # std_hull = np.std(hull_length)
        #
        # #  Remove some bounding boxes based on the detector CNN
        # s = list()
        # for i, bb in enumerate(bbs):
        #     if len(hulls[i]) > mean_hull - .2*std_hull:
        #         crop = vis[bb[0]:bb[1],bb[2]:bb[3]]
        #         # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        #         m, v = predict(crop, detect_model)
        #         if v == 1:
        #             y = bb[0]
        #             x = bb[2]
        #             h = bb[1]-bb[0]
        #             w = bb[3]-bb[2]
        #             A = h * w
        #             m, v = predict(crop, class_model)
        #             s.append((bb, A, v))
        #
        # #  Poor man's NMS
        # sorted(s, key=lambda tup: tup[1])
        # s = s[::-1]
        #
        # r = list()
        # for i in range(len(s)):
        #     r.append(0)
        #
        # for i in range(len(s)):
        #     for j in range(len(s)):
        #         if i != j and \
        #         (s[i][0][2] < s[j][0][2]) and \
        #         (s[i][0][3] > s[j][0][3]) and \
        #         (s[i][0][0] < s[j][0][0]) and \
        #         (s[i][0][1]> s[j][0][1]):
        #             # i is big, j is small
        #             if s[i][2] != 0 and s[j][2] == 0:
        #                 r[j] = 1
        #             elif s[j][2] == s[i][2]:
        #                 r[j] = 1
        #
        # s2 = list()
        # r2 = list()
        # for i in range(len(r)):
        #     if r[i] == 0:
        #         s2.append(s[i])
        #         r2.append(0)
        #
        # for i in range(len(s2)):
        #     for j in range(len(s2)):
        #         if i != j and \
        #         (s2[i][0][2] < s2[j][0][2]) and \
        #         (s2[i][0][3] > s2[j][0][3]) and \
        #         (s2[i][0][0] < s2[j][0][0]) and \
        #         (s2[i][0][1]> s2[j][0][1]):
        #             # i is big, j is small
        #             if s2[i][2] != s2[j][2]:
        #                 r2[i] = 1
        #
        # cand = list()
        # for i in range(len(r2)):
        #     if r2[i] == 0:
        #         cand.append(s2[i][0])
        #
        # #  Apply the CNN's the bounding boxes
        # for i,bb in enumerate(cand):
        #     y = bb[0]
        #     x = bb[2]
        #     h = bb[1]-bb[0]
        #     w = bb[3]-bb[2]
        #     crop = vis[bb[0]:bb[1],bb[2]:bb[3]]
        #     crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        #     m, v = predict(crop, class_model)
        #     print(m, v)
        #     if m > .90:
        #         cv2.putText(vis, str(v), (x, y+int(0.2*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
        #         cv2.rectangle(vis, (x, y), (x+w,y+h), (0, 255, 0), 1)
        #
        # cv2.imwrite(filename=write_filename, img=vis)

print("=== DONE ===")



    # model = load_model('weights.hdf5')
    # model.summary()
    # # normalize image pixel values into range [0,1]
    # img_generator = image.ImageDataGenerator(preprocessing_function=lambda img: img/255.0)
    # validation_generator = img_generator.flow_from_directory(directory=img_path, target_size=(32, 32), shuffle=False,
    #                                                          batch_size=batch_size, color_mode="rgb")
    #
    # score = model.evaluate_generator(validation_generator)
    # print("Accuracy: {:.4f}".format(score[1]))
