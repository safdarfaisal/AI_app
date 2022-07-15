import pickle
import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import sys

#def readImage(filePath):
#    image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
#    face_cas = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')
#    faces = face_cas.detectMultiScale (image, scaleFactor=1.3, minNeighbors=4)
#    (x, y, w, h) = faces [0]
#    image = image[y: y+w, x: x+h]
#    image = cv2.resize(image, (180.180))
#    # cv2.imshow("helloWorld", image)
#    # cv2.waitKey(0)
#    return image

def readImage(filePath):
    image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    #face_cas = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')
    #faces = face_cas.detectMultiScale (image, scaleFactor=1.1, minNeighbors=3)
    cnn_face_detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
    faces = cnn_face_detector(image, 1)
    #(x, y, w, h) = faces [0]
    rect =  faces[0].rect
    #image = image[y: y+w, x: x+h]
    image = image[rect.top():rect.bottom(), rect.left():rect.right()]
    image = cv2.resize(image, (180,180))
    #cv2.imshow("helloWorld", image)
    #cv2.waitKey(0)
    return image

def predict(filePath, modelFolder):
    try:
        newModel = tf.keras.models.load_model(os.path.join(modelFolder, 'model.h5'))
        y_classes = pickle.load(open(os.path.join(modelFolder, 'classes.pickle'),"rb"))
    except Exception as E:
        print(E)
        return 0,0
    img = readImage(filePath)
    img2 = np.array(img)
    img2 = img2.reshape( -1, 180,180, 1)
    y = newModel.predict(img2/255.)
    index = np.argmax(y[0])
    for i in range(7):
        print(y_classes[i], ": ", y[0][i])
    cv2.imshow("helloWorld", img)
    cv2.waitKey(0)
    probability = y[0][index]
    predicted_class  = y_classes[index]
    return predicted_class , probability

testFolder = os.path.join("/Users/fsiyavud/Downloads/AI App/model_files/indian/val", sys.argv[1])
for image in os.listdir(testFolder):
    testFile = os.path.join(testFolder, image)
    print(testFile)
    pred_class, prob = predict(testFile, "/Users/fsiyavud/Downloads/AI App/model_files/testing")
    #pred_class, prob = predict(testFile, "/Users/fsiyavud/Downloads/AI App/model_files/2022-07-10T06-23-10.865390_model")
    print(pred_class)
    print(prob)

