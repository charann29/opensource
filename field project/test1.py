import numpy as np 
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import os
from cv2 import imread, createCLAHE 
import cv2
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import pickle
'''
train_directory = "dataset/flair"

X = []
Y = []
for root, dirs, directory in os.walk(train_directory):
    for i in range(len(directory)):
        img = cv2.imread(train_directory+"/"+directory[i],0)
        img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
        X.append(img)
        img = cv2.imread("dataset/label/"+directory[i],0)
        img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
        Y.append(img)

X = np.asarray(X)
Y = np.asarray(Y)

#np.save("model/train.txt",X)
#np.save("model/label.txt",Y)

'''

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def getModel(input_size=(64,64,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])


#X = np.load('model/train.txt.npy')
#Y = np.load('model/label.txt.npy')
'''
dim = 64
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, random_state = 1)
X_train = X_train.reshape(len(X_train),dim,dim,1)
y_train = y_train.reshape(len(y_train),dim,dim,1)
X_test = X_test.reshape(len(X_test),dim,dim,1)
y_test = y_test.reshape(len(y_test),dim,dim,1)
images = np.concatenate((X_train,X_test),axis=0)
mask  = np.concatenate((y_train,y_test),axis=0)

tr = X_train[12]
yr = y_train[12]

cv2.imshow('tr',tr)
cv2.imshow('yr',yr)
cv2.waitKey(0)



model = getModel(input_size=(64,64,1))
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
print(model.summary())
model.compile(optimizer=Adam(lr=2e-4), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy'])

train_vol, validation_vol, train_seg, validation_seg = train_test_split((images-127.0)/127.0, 
                                                            (mask>127).astype(np.float32), 
                                                            test_size = 0.1,random_state = 2018)

train_vol, test_vol, train_seg, test_seg = train_test_split(train_vol,train_seg, 
                                                            test_size = 0.1, 
                                                            random_state = 2018)

hist = model.fit(x = train_vol, y = train_seg, batch_size = 16, epochs = 50, validation_data =(test_vol,test_seg))
model.save_weights('model/model_weights.h5')            
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
f = open('model/history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()
'''

def getMark():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95*180*35
    max_area = 1.05*180*35
    result = orig.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 10)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 10)
    return result        
            



with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

model.load_weights("model/model_weights.h5")
model._make_predict_function()   
print(model.summary())

lists = np.empty([1,128,128,1])
test = 'test.png'
img = cv2.imread(test,0)
img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
img = img.reshape(1,64,64,1)
img = (img-127.0)/127.0
preds = model.predict(img)
preds = preds[0]
print(preds.shape)
orig = cv2.imread(test,0)
orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
cv2.imwrite("test1.png",orig)

preds = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
cv2.imwrite("myimg.png",preds*255)
preds = getMark()
cv2.imshow('orig',orig)
cv2.imshow("ll",preds)
cv2.waitKey(0)












