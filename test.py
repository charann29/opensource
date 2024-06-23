# from skimage import io
# from skimage import data, color
# from skimage.transform import hough_circle, hough_circle_peaks
# from skimage.feature import canny
# from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from keras.utils.np_utils import to_categorical
# from keras.layers import  MaxPooling2D
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D
# from keras.models import Sequential
# from keras.models import model_from_json
# import pickle

# count = 0
# miss = []

# def getIrisFeatures(image):
#     global count
#     img = cv2.imread(image,0)
#     img = cv2.medianBlur(img,5)
#     cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,param1=63,param2=70,minRadius=0,maxRadius=0)
#     status = 'failed'
#     crop = None
#     if circles is not None:
#         height,width = img.shape
#         r = 0
#         mask = np.zeros((height,width), np.uint8)
#         for i in circles[0,:]:
#             cv2.circle(cimg,(i[0],i[1]),int(i[2]),(0,0,0))
#             cv2.circle(mask,(i[0],i[1]),int(i[2]),(255,255,255),thickness=0)
#             blank_image = cimg[:int(i[1]),:int(i[1])]

#             masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)
#             _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
#             contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#             x,y,w,h = cv2.boundingRect(contours[0][0])
#             crop = img[y:y+h,x:x+w]
#             cv2.imwrite('test.png',crop)
#             crop = cv2.imread('test.png')
#             r = i[2]
#             status = 'success'
    
#     return status,crop

# path = 'CASIA1'

# labels = []
# X_train = []
# Y_train = []

# def getID(name):
#     index = 0
#     for i in range(len(labels)):
#         if labels[i] == name:
#             index = i
#             break
#     return index        
    

# for root, dirs, directory in os.walk(path):
#     for j in range(len(directory)):
#         name = os.path.basename(root)
#         if name not in labels:
#             labels.append(name)
# print(labels)

# for root, dirs, directory in os.walk(path):
#     for j in range(len(directory)):
#         name = os.path.basename(root)
#         if 'Thumbs.db' not in directory[j]:
#             status,img = getIrisFeatures(root+"/"+directory[j])
#             if status =='success':
#                 img = cv2.resize(img, (64,64))
#                 im2arr = np.array(img)
#                 im2arr = im2arr.reshape(64,64,3)
#                 X_train.append(im2arr)
#                 ids = getID(name)
#                 Y_train.append(int(name)-1)
#                 print(str(ids)+" "+str(name))
        


# X_train = np.asarray(X_train)
# Y_train = np.asarray(Y_train)
# print(Y_train)

# X_train = X_train.astype('float32')
# X_train = X_train/255
    
# test = X_train[3]
# cv2.imshow("aa",test)
# cv2.waitKey(0)
# indices = np.arange(X_train.shape[0])
# np.random.shuffle(indices)
# X_train = X_train[indices]
# Y_train = Y_train[indices]
# Y_train = to_categorical(Y_train)
# np.save('model/X.txt',X_train)
# np.save('model/Y.txt',Y_train)

# X_train = np.load('model/X.txt.npy')
# Y_train = np.load('model/Y.txt.npy')
# print(Y_train)
# if os.path.exists('model/model.json'):
#     with open('model/model.json', "r") as json_file:
#         loaded_model_json = json_file.read()
#         classifier = model_from_json(loaded_model_json)
#     classifier.load_weights("model/model_weights.h5")
#     classifier._make_predict_function()   
#     print(classifier.summary())
#     f = open('model/history.pckl', 'rb')
#     data = pickle.load(f)
#     f.close()
#     acc = data['accuracy']
#     accuracy = acc[9] * 100
#     print("Training Model Accuracy = "+str(accuracy))
# else:
#     classifier = Sequential()
#     classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
#     classifier.add(MaxPooling2D(pool_size = (2, 2)))
#     classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#     classifier.add(MaxPooling2D(pool_size = (2, 2)))
#     classifier.add(Flatten())
#     classifier.add(Dense(output_dim = 256, activation = 'relu'))
#     classifier.add(Dense(output_dim = 108, activation = 'softmax'))
#     print(classifier.summary())
#     classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#     hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
#     classifier.save_weights('model/model_weights.h5')            
#     model_json = classifier.to_json()
#     with open("model/model.json", "w") as json_file:
#         json_file.write(model_json)
#     f = open('model/history.pckl', 'wb')
#     pickle.dump(hist.history, f)
#     f.close()
#     f = open('model/history.pckl', 'rb')
#     data = pickle.load(f)
#     f.close()
#     acc = data['accuracy']
#     accuracy = acc[9] * 100
#     print("Training Model Accuracy = "+str(accuracy))


import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from keras.models import Sequential, model_from_json
import pickle

def getIrisFeatures(image):
    img = cv2.imread(image, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
    if circles is not None:
        mask = np.zeros_like(img)
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=-1)
            break  # Use only the first circle detected
        masked_data = cv2.bitwise_and(img, img, mask=mask)
        x, y, w, h = cv2.boundingRect(mask)
        crop = masked_data[y:y+h, x:x+w]
        return 'success', cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return 'failed', None

def getID(name, labels):
    if name not in labels:
        labels.append(name)
    return labels.index(name)

path = 'CASIA1'
labels = []
X_train = []
Y_train = []

for root, dirs, files in os.walk(path):
    for file in files:
        if 'Thumbs.db' not in file:
            status, img = getIrisFeatures(os.path.join(root, file))
            if status == 'success':
                img = cv2.resize(img, (64, 64))
                X_train.append(img)
                ids = getID(os.path.basename(root), labels)
                Y_train.append(ids)

X_train = np.asarray(X_train).astype('float32') / 255.0
Y_train = to_categorical(np.asarray(Y_train), num_classes=len(labels))

os.makedirs('model', exist_ok=True)
np.save('model/X.txt', X_train)
np.save('model/Y.txt', Y_train)

if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("model/model_weights.h5")
    model._make_predict_function()
    with open('model/history.pckl', 'rb') as f:
        data = pickle.load(f)
    accuracy = data['accuracy'][-1] * 100
    print(f"Training Model Accuracy = {accuracy:.2f}%")
else:
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
    model.save_weights('model/model_weights.h5')
    with open("model/model.json", "w") as json_file:
        json_file.write(model.to_json())
    with open('model/history.pckl', 'wb') as f:
        pickle.dump(hist.history, f)
    accuracy = hist.history['accuracy'][-1] * 100
    print(f"Training Model Accuracy = {accuracy:.2f}%")
