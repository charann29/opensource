from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import simpledialog
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pickle 
import os
import cv2
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.models import model_from_json

gui = tkinter.Tk()
gui.title("Brain Tumour Image Segmentation Using Deep Networks") 
gui.geometry("1300x1200")

global filename
global model
global X, Y
global unet_image

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


def uploadDataset():
    global X, Y
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,filename+" loaded\n");
    '''
    X = []
    Y = []
    for root, dirs, directory in os.walk(filename):
        for i in range(len(directory)):
            img = cv2.imread(train_directory+"/"+directory[i],0)
            img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
            X.append(img)
            img = cv2.imread("dataset/label/"+directory[i],0)
            img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
            Y.append(img)

    X = np.asarray(X)
    Y = np.asarray(Y)
    '''
def generateModel():
    global model
    '''
    global X, Y
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
    '''
    model = getModel(input_size=(64,64,1))
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    json_file.close()    
    model.load_weights("model/model_weights.h5")
    model._make_predict_function()   
    print(model.summary())
    text.insert(END,"CNN & UNET model generated. See Black Console for model details\n")
    '''
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


def getSegmentation():
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

def TumourSegmentation():
    global model
    global filename
    global unet_image
    filename = filedialog.askdirectory(initialdir="testSamples")
    img = cv2.imread(str(filename)+'/t2.png',0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = model.predict(img)
    preds = preds[0]
    print(preds.shape)
    orig = cv2.imread(str(filename)+'/t2.png',0)
    orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test1.png",orig)

    flair = cv2.imread(str(filename)+'/flair.png',0)
    flair = cv2.resize(flair,(300,300),interpolation = cv2.INTER_CUBIC)
    t1 = cv2.imread(str(filename)+'/t1.png',0)
    t1 = cv2.resize(t1,(300,300),interpolation = cv2.INTER_CUBIC)
    t1ce = cv2.imread(str(filename)+'/t1ce.png',0)
    t1ce = cv2.resize(t1ce,(300,300),interpolation = cv2.INTER_CUBIC)
    preds = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("myimg.png",preds*255)
    preds = getSegmentation()
    unet_image = preds
    cv2.imshow('Flair Image',flair)
    cv2.imshow('T1',t1)
    cv2.imshow("T1ce Image",t1ce)
    cv2.imshow('T2 Image',orig)
    cv2.imshow("Label Image",preds)
    cv2.waitKey(0)
    
    

def ResnetTumourSegmentation():
    global filename
    with open('model/resnet.json', "r") as json_file:
        loaded_model_json = json_file.read()
        resnet_model = model_from_json(loaded_model_json)
    json_file.close()
    resnet_model.load_weights("model/resnet_weights.h5")
    resnet_model._make_predict_function()  
    img = cv2.imread(filename+'/t2.png',0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = resnet_model.predict(img)
    preds = preds[0]
    cv2.imwrite('test.png',cv2.resize(preds*255,(255,255),interpolation = cv2.INTER_CUBIC))
    resnetImage = cv2.imread("test.png")
    resnetImage = cv2.resize(resnetImage,(300,300))

    orig = cv2.imread(str(filename)+'/t2.png',0)
    orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test1.png",orig)

    flair = cv2.imread(str(filename)+'/flair.png',0)
    flair = cv2.resize(flair,(300,300),interpolation = cv2.INTER_CUBIC)
    t1 = cv2.imread(str(filename)+'/t1.png',0)
    t1 = cv2.resize(t1,(300,300),interpolation = cv2.INTER_CUBIC)
    t1ce = cv2.imread(str(filename)+'/t1ce.png',0)
    t1ce = cv2.resize(t1ce,(300,300),interpolation = cv2.INTER_CUBIC)

    preds = cv2.resize(resnetImage,(300,300),interpolation = cv2.INTER_CUBIC)
    preds = getSegmentation()
    preds = cv2.fastNlMeansDenoisingColored(preds,None,10,10,7,22)
    preds = cv2.convertScaleAbs(preds, alpha=1.0, beta=4)
    cv2.imwrite("myimg.png",preds)
    cv2.imshow('UNET Segmented Image',unet_image)
    cv2.imshow("Resnet Segmented Image",preds)
    cv2.waitKey(0)

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    print(data)
    dice = data['dice_coef']
    for i in range(len(dice)):
        dice[i] = dice[i] * 2
    resnet_dice = np.load("model/resnet_history.pckl.npy")
    text.delete('1.0', END)
    text.insert(END,"UNET Dice Similarity : "+str(dice[len(dice)-1])+"\n")
    text.insert(END,"RESNET Dice Similarity : "+str(resnet_dice[len(resnet_dice)-1])+"\n")
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Dice Score')
    plt.plot(dice, 'ro-', color = 'green')
    plt.plot(resnet_dice, 'ro-', color = 'blue')
    plt.legend(['UNET Dice Score','Resnet Dice Score'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Iteration Wise Dice Score Graph')
    plt.show()
        

    

font = ('times', 16, 'bold')
title = Label(gui, text='Brain Tumour Image Segmentation Using Deep Networks')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(gui,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=350)
text.config(font=font1)


font1 = ('times', 12, 'bold')
loadButton = Button(gui, text="Upload BRATS Dataset", command=uploadDataset)
loadButton.place(x=50,y=100)
loadButton.config(font=font1)  

uploadButton = Button(gui, text="Generate CNN & UNET Model", command=generateModel)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1) 

descButton = Button(gui, text="Upload Test Image & Segmentation", command=TumourSegmentation)
descButton.place(x=50,y=200)
descButton.config(font=font1)

resnetButton = Button(gui, text="Resnet Test Image & Segmentation", command=ResnetTumourSegmentation)
resnetButton.place(x=50,y=250)
resnetButton.config(font=font1)

closeButton = Button(gui, text="Dice Similarity Graph", command=graph)
closeButton.place(x=50,y=300)
closeButton.config(font=font1) 



gui.config(bg='OliveDrab2')
gui.mainloop()
