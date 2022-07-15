from my_logging import setup_logging
import logging
from tqdm.auto import tqdm
import pickle
import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import shutil

# %matplotlib inline

# def detect_face(image):
#     gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
#     face_cas = cv2.CascadeClassifier ('

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

def saveSamples (dirName):
    list1 = os.listdir(dirName)
    corrupt_data = 0
    preparedPath = "/Users/fsiyavud/Downloads/AI App/model_files/indian/saved"
    os.makedirs(preparedPath, exist_ok=True)
    try:
        for n , a in enumerate(list1):
            current_dir = os.path.join(dirName , a)
            for images in os.listdir(current_dir):
                fPath = os.path.join(current_dir, images)
                try:
                    img = readImage(fPath)
                    dPath = os.path.join(preparedPath, a)
                    os.makedirs(dPath, exist_ok=True)
                    cv2.imwrite(os.path.join(dPath, images), img)
                    #cv2.imshow("helloworld", img)
                    #cv2.waitKey(0)
                except Exception as E:
                    print(E, " ", fPath)
                    corrupt_data = 1
    except Exception as E:
        print(E)
        return
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_train.reshape(-1,180,180,1).shape)

def prepareFromSaved (dirName):
    list1 = os.listdir(dirName)
    x_train = []
    y_train = []
    corrupt_data = 0
    try:
        for n , a in enumerate(list1):
            current_dir = os.path.join(dirName , a)
            for images in os.listdir(current_dir):
                fPath = os.path.join(current_dir, images)
                try:
                    img = cv2.imread(fPath)
                    x_train.append(img)
                    y_train.append(n)
                    # cv2.imshow("helloworld", img)
                    # cv2.waitKey(0)
                except Exception as E:
                    print(E, " ", fPath)
                    corrupt_data = 1
    except Exception as E:
        print(E)
        return 0,0,0
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_train.reshape(-1,180,180,1).shape)
    return x_train, y_train, list1

def prepareSamples (dirName):
    list1 = os.listdir(dirName)
    x_train = []
    y_train = []
    corrupt_data = 0
    try:
        for n , a in enumerate(list1):
            current_dir = os.path.join(dirName , a)
            for images in os.listdir(current_dir):
                fPath = os.path.join(current_dir, images)
                try:
                    img = readImage(fPath)
                    x_train.append(img)
                    y_train.append(n)
                    # cv2.imshow("helloworld", img)
                    # cv2.waitKey(0)
                except Exception as E:
                    print(E, " ", fPath)
                    corrupt_data = 1
    except Exception as E:
        print(E)
        return 0,0,0
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_train.reshape(-1,180,180,1).shape)
    return x_train, y_train, list1

def train(imageArray, labelArray, list):
    no_classes = len(list)
    imageArray = imageArray.reshape(-1, 180,180, 1)
    imageArray = imageArray/255.
    setup_logging()
    _logger = logging.getLogger('train')
    _logger.setLevel(logging.DEBUG)
    model_name = "testing"
    try:
        model_path = f'model_files/{model_name}'
        shutil.rmtree(model_path)
        os.mkdir(model_path)
        print(f"Created Directory model/{model_name} successfully")
    except Exception as E:
        print(f"Model with name '{model_name}' already exist.")
        model_name = "{}_model".format(datetime.datetime.utcnow().isoformat().replace(":", "-"))
        print("Randomly Initiating New model Name ...")
        print(f"New Model Name is {model_name}")
        model_path = f'model_files/{model_name}'
        os.mkdir(model_path)
        print(f"Created Directory model/{model_name} successfully")

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(128 ,(3,3) , input_shape = imageArray.shape[1:] , activation = "relu" ))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
    #model.add(tf.keras.layers.Dropout(0.20))

    model.add(tf.keras.layers.Conv2D(64 ,(3,3)  , activation = "relu" ))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
    #model.add(tf.keras.layers.Dropout(0.20))


    model.add(tf.keras.layers.Conv2D(64, (3 , 3) , activation="relu"))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    #model.add(tf.keras.layers.Dropout(0.20))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128 , activation = tf.nn.relu))

    model.add(tf.keras.layers.Dense(no_classes , activation  = tf.nn.softmax))

    model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] , run_eagerly = True)
    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    print(f"{model_name} : Training Started ")
    model.fit(imageArray, labelArray , epochs = int(5 * no_classes)  , callbacks = [tqdm_callback] , verbose=1)

    # tqdm_obect = tqdm(tqdm_callback, unit_scale=True, dynamic_ncols=True)
    # tqdm_obect.set_description("Model Trained !")
    model.save(f'{model_path}/model.h5')

    pickle_out= open(f'{model_path}/classes.pickle', "wb")
    pickle.dump(list, pickle_out)
    pickle_out.close


#imageArray, labelArray, dirlist = prepareSamples("/Users/fsiyavud/Downloads/AI App/model_files/indian/train")

#train(imageArray, labelArray, dirlist)
saveSamples("/Users/fsiyavud/Downloads/AI App/model_files/indian/train")

# prepareSamples("C:/Users/Godhome/Documents/AI App/model_files/train")
