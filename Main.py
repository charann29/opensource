# # Original import
# # from keras.utils.np_utils import to_categorical

# # Updated import
# from tensorflow.keras.utils import to_categorical

# # Rest of your code
# # ...

# from tkinter import messagebox
# from tkinter import *
# from tkinter import simpledialog
# import tkinter
# from tkinter import filedialog
# from tkinter.filedialog import askopenfilename
# import numpy as np 
# import matplotlib.pyplot as plt
# import os
# # from keras.utils.np_utils import to_categorical
# from tensorflow.keras.utils import to_categorical

# from keras.layers import  MaxPooling2D
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D
# from keras.models import Sequential
# from keras.models import model_from_json
# import pickle
# import cv2
# from keras.preprocessing import image
# from skimage import data, color
# from skimage.transform import hough_circle, hough_circle_peaks
# from skimage.feature import canny
# from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte

# main = tkinter.Tk()
# main.title("Iris Recognition using Machine Learning Technique") #designing main screen
# main.geometry("1300x1200")

# global filename
# global model

# def getIrisFeatures(image):
#     global count
#     img = cv2.imread(image,0)
#     img = cv2.medianBlur(img,5)
#     cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,param1=63,param2=70,minRadius=0,maxRadius=0)
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
#             r = i[2]
#         cv2.imwrite("test.png",crop)
#     else:
#         count = count + 1
#         miss.append(image)
#     return cv2.imread("test.png")

# def uploadDataset():
#     global filename
#     filename = filedialog.askdirectory(initialdir=".")
#     text.delete('1.0', END)
#     text.insert(END,filename+" loaded\n\n");

# def loadModel():
#     global model
#     text.delete('1.0', END)
#     X_train = np.load('model/X.txt.npy')
#     Y_train = np.load('model/Y.txt.npy')
#     print(X_train.shape)
#     print(Y_train.shape)
#     text.insert(END,'Dataset contains total '+str(X_train.shape[0])+' iris images from '+str(Y_train.shape[1])+"\n")
#     if os.path.exists('model/model.json'):
#         with open('model/model.json', "r") as json_file:
#             loaded_model_json = json_file.read()
#             model = model_from_json(loaded_model_json)
#         model.load_weights("model/model_weights.h5")
#         model._make_predict_function()   
#         print(model.summary())
#         f = open('model/history.pckl', 'rb')
#         data = pickle.load(f)
#         f.close()
#         acc = data['accuracy']
#         accuracy = acc[59] * 100
#         text.insert(END,"CNN Model Prediction Accuracy = "+str(accuracy)+"\n\n")
#         text.insert(END,"See Black Console to view CNN layers\n")
#     else:
#         model = Sequential()
#         model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
#         model.add(MaxPooling2D(pool_size = (2, 2)))
#         model.add(Convolution2D(32, 3, 3, activation = 'relu'))
#         modeladd(MaxPooling2D(pool_size = (2, 2)))
#         model.add(Flatten())
#         model.add(Dense(output_dim = 256, activation = 'relu'))
#         model.add(Dense(output_dim = 108, activation = 'softmax'))
#         print(model.summary())
#         model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#         hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
#         model.save_weights('model/model_weights.h5')            
#         model_json = classifier.to_json()
#         with open("model/model.json", "w") as json_file:
#             json_file.write(model_json)
#         f = open('model/history.pckl', 'wb')
#         pickle.dump(hist.history, f)
#         f.close()
#         f = open('model/history.pckl', 'rb')
#         data = pickle.load(f)
#         f.close()
#         acc = data['accuracy']
#         accuracy = acc[59] * 100
#         text.insert(END,"CNN Model Prediction Accuracy = "+str(accuracy)+"\n\n")
#         text.insert(END,"See Black Console to view CNN layers\n")

# def predictChange():
#     filename = filedialog.askopenfilename(initialdir="testSamples")
#     image = getIrisFeatures(filename)
#     img = cv2.resize(image, (64,64))
#     im2arr = np.array(img)
#     im2arr = im2arr.reshape(1,64,64,3)
#     img = np.asarray(im2arr)
#     img = img.astype('float32')
#     img = img/255
#     preds = model.predict(img)
#     predict = np.argmax(preds) + 1
#     print(predict)
#     img = cv2.imread(filename)
#     img = cv2.resize(img, (600,400))
#     img1 = cv2.imread('test.png')
#     img1 = cv2.resize(img1, (400,200))
#     cv2.putText(img, 'Person ID Predicted from Iris Recognition is : '+str(predict), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
#     cv2.imshow('Person ID Predicted from Iris Recognition is : '+str(predict), img)
#     cv2.imshow('Iris features extacted from image', img1)
#     cv2.waitKey(0)
    


# def graph():
#     f = open('model/history.pckl', 'rb')
#     data = pickle.load(f)
#     f.close()

#     accuracy = data['accuracy']
#     loss = data['loss']
#     plt.figure(figsize=(10,6))
#     plt.grid(True)
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy/Loss')
#     plt.plot(loss, 'ro-', color = 'red')
#     plt.plot(accuracy, 'ro-', color = 'green')
#     plt.legend(['Loss', 'Accuracy'], loc='upper left')
#     #plt.xticks(wordloss.index)
#     plt.title('GoogLeNet Accuracy & Loss Graph')
#     plt.show()

# def close():
#     main.destroy()
    
# font = ('times', 16, 'bold')
# title = Label(main, text='Iris Recognition using Machine Learning Technique')
# title.config(bg='goldenrod2', fg='black')  
# title.config(font=font)           
# title.config(height=3, width=120)       
# title.place(x=0,y=5)

# font1 = ('times', 12, 'bold')
# text=Text(main,height=20,width=150)
# scroll=Scrollbar(text)
# text.configure(yscrollcommand=scroll.set)
# text.place(x=50,y=120)
# text.config(font=font1)


# font1 = ('times', 13, 'bold')
# uploadButton = Button(main, text="Upload Iris Dataset", command=uploadDataset, bg='#ffb3fe')
# uploadButton.place(x=50,y=550)
# uploadButton.config(font=font1)  

# modelButton = Button(main, text="Generate & Load CNN Model", command=loadModel, bg='#ffb3fe')
# modelButton.place(x=240,y=550)
# modelButton.config(font=font1) 

# graphButton = Button(main, text="Accuracy & Loss Graph", command=graph, bg='#ffb3fe')
# graphButton.place(x=505,y=550)
# graphButton.config(font=font1) 

# predictButton = Button(main, text="Upload Iris Test Image & Recognize", command=predictChange, bg='#ffb3fe')
# predictButton.place(x=730,y=550)
# predictButton.config(font=font1) 

# exitButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
# exitButton.place(x=1050,y=550)
# exitButton.config(font=font1) 


# main.config(bg='SpringGreen2')
# main.mainloop()
# from tkinter import messagebox
# from tkinter import *
# from tkinter import simpledialog
# import tkinter
# from tkinter import filedialog
# from tkinter.filedialog import askopenfilename
# import numpy as np 
# import matplotlib.pyplot as plt
# import os
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D  # Use Conv2D instead of Convolution2D
# from tensorflow.keras.models import Sequential, model_from_json
# import pickle
# import cv2
# from skimage import data, color
# from skimage.transform import hough_circle, hough_circle_peaks
# from skimage.feature import canny
# from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte

# main = tkinter.Tk()
# main.title("Iris Recognition using Machine Learning Technique")
# main.geometry("1300x1200")

# global filename
# global model

# def getIrisFeatures(image):
#     global count
#     img = cv2.imread(image, 0)
#     if img is None:
#         messagebox.showerror("Error", "Image not found or unable to read")
#         return None
#     img = cv2.medianBlur(img, 5)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
#     if circles is not None:
#         height, width = img.shape
#         r = 0
#         mask = np.zeros((height, width), np.uint8)
#         for i in circles[0, :]:
#             cv2.circle(cimg, (i[0], i[1]), int(i[2]), (0, 0, 0))
#             cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=0)
#             masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)
#             _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
#             contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             x, y, w, h = cv2.boundingRect(contours[0][0])
#             crop = img[y:y+h, x:x+w]
#             r = i[2]
#         cv2.imwrite("test.png", crop)
#     else:
#         count = count + 1
#         miss.append(image)
#         messagebox.showwarning("Warning", "No eye iris found")
#         return None
#     return cv2.imread("test.png")

# def uploadDataset():
#     global filename
#     filename = filedialog.askdirectory(initialdir=".")
#     text.delete('1.0', END)
#     text.insert(END, filename + " loaded\n\n")

# def loadModel():
#     global model
#     text.delete('1.0', END)
#     try:
#         X_train = np.load('model/X.txt.npy')
#         Y_train = np.load('model/Y.txt.npy')
#         text.insert(END, 'Dataset contains total ' + str(X_train.shape[0]) + ' iris images from ' + str(Y_train.shape[1]) + "\n")
#     except FileNotFoundError:
#         messagebox.showerror("Error", "Training data not found")
#         return

#     if os.path.exists('model/model.json'):
#         with open('model/model.json', "r") as json_file:
#             loaded_model_json = json_file.read()
#             model = model_from_json(loaded_model_json)
#         model.load_weights("model/model_weights.h5")
#         print(model.summary())
#         try:
#             with open('model/history.pckl', 'rb') as f:
#                 data = pickle.load(f)
#             acc = data['accuracy']
#             accuracy = acc[59] * 100
#             text.insert(END, "CNN Model Prediction Accuracy = " + str(accuracy) + "\n\n")
#             text.insert(END, "See Black Console to view CNN layers\n")
#         except FileNotFoundError:
#             messagebox.showerror("Error", "History file not found")
#     else:
#         model = Sequential()
#         model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(32, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(256, activation='relu'))
#         model.add(Dense(108, activation='softmax'))
#         print(model.summary())
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
#         model.save_weights('model/model_weights.h5')
#         model_json = model.to_json()
#         with open("model/model.json", "w") as json_file:
#             json_file.write(model_json)
#         with open('model/history.pckl', 'wb') as f:
#             pickle.dump(hist.history, f)
#         with open('model/history.pckl', 'rb') as f:
#             data = pickle.load(f)
#         acc = data['accuracy']
#         accuracy = acc[59] * 100
#         text.insert(END, "CNN Model Prediction Accuracy = " + str(accuracy) + "\n\n")
#         text.insert(END, "See Black Console to view CNN layers\n")

# def predictChange():
#     filename = filedialog.askopenfilename(initialdir="testSamples")
#     image = getIrisFeatures(filename)
#     if image is None:
#         return
#     img = cv2.resize(image, (64, 64))
#     im2arr = np.array(img)
#     im2arr = im2arr.reshape(1, 64, 64, 3)
#     img = np.asarray(im2arr)
#     img = img.astype('float32')
#     img = img / 255
#     preds = model.predict(img)
#     predict = np.argmax(preds) + 1
#     print(predict)
#     img = cv2.imread(filename)
#     img = cv2.resize(img, (600, 400))
#     img1 = cv2.imread('test.png')
#     img1 = cv2.resize(img1, (400, 200))
#     cv2.putText(img, 'Person ID Predicted from Iris Recognition is : ' + str(predict), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.imshow('Person ID Predicted from Iris Recognition is : ' + str(predict), img)
#     cv2.imshow('Iris features extracted from image', img1)
#     cv2.waitKey(0)

# def graph():
#     try:
#         with open('model/history.pckl', 'rb') as f:
#             data = pickle.load(f)
#     except FileNotFoundError:
#         messagebox.showerror("Error", "History file not found")
#         return

#     accuracy = data['accuracy']
#     loss = data['loss']
#     plt.figure(figsize=(10, 6))
#     plt.grid(True)
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy/Loss')
#     plt.plot(loss, 'ro-', color='red')
#     plt.plot(accuracy, 'ro-', color='green')
#     plt.legend(['Loss', 'Accuracy'], loc='upper left')
#     plt.title('GoogLeNet Accuracy & Loss Graph')
#     plt.show()

# def close():
#     main.destroy()

# font = ('times', 16, 'bold')
# title = Label(main, text='Iris Recognition using Machine Learning Technique')
# title.config(bg='goldenrod2', fg='black')
# title.config(font=font)
# title.config(height=3, width=120)
# title.place(x=0, y=5)

# font1 = ('times', 12, 'bold')
# text = Text(main, height=20, width=150)
# scroll = Scrollbar(text)
# text.configure(yscrollcommand=scroll.set)
# text.place(x=50, y=120)
# text.config(font=font1)

# font1 = ('times', 13, 'bold')
# uploadButton = Button(main, text="Upload Iris Dataset", command=uploadDataset, bg='#ffb3fe')
# uploadButton.place(x=50, y=550)
# uploadButton.config(font=font1)

# modelButton = Button(main, text="Generate & Load CNN Model", command=loadModel, bg='#ffb3fe')
# modelButton.place(x=240, y=550)
# modelButton.config(font=font1)

# graphButton = Button(main, text="Accuracy & Loss Graph", command=graph, bg='#ffb3fe')
# graphButton.place(x=505, y=550)
# graphButton.config(font=font1)

# predictButton = Button(main, text="Upload Iris Test Image & Recognize", command=predictChange, bg='#ffb3fe')
# predictButton.place(x=730, y=550)
# predictButton.config(font=font1)

# exitButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
# exitButton.place(x=1050, y=550)
# exitButton.config(font=font1)

# main.config(bg='SpringGreen2')
# main.mainloop()
# from tkinter import messagebox, filedialog, simpledialog, Text, Label, Button, Scrollbar
# import tkinter as tk
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D
# from tensorflow.keras.models import Sequential, model_from_json
# import pickle
# import cv2

# # Initialize main window
# main = tk.Tk()
# main.title("Iris Recognition using Machine Learning Technique")
# main.geometry("1300x1200")

# global filename
# global model

# # Function to extract iris features
# def getIrisFeatures(image):
#     img = cv2.imread(image, 0)
#     if img is None:
#         messagebox.showerror("Error", "Image not found or unable to read")
#         return None
#     img = cv2.medianBlur(img, 5)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
#     if circles is not None:
#         height, width = img.shape
#         r = 0
#         mask = np.zeros((height, width), np.uint8)
#         for i in circles[0, :]:
#             cv2.circle(cimg, (i[0], i[1]), int(i[2]), (0, 0, 0))
#             cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=0)
#             masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)
#             _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
#             contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             x, y, w, h = cv2.boundingRect(contours[0][0])
#             crop = img[y:y+h, x:x+w]
#             r = i[2]
#         cv2.imwrite("test.png", crop)
#     else:
#         messagebox.showwarning("Warning", "No circles detected")
#         return None
#     return cv2.imread("test.png")

# # Function to upload dataset
# def uploadDataset():
#     global filename
#     filename = filedialog.askdirectory(initialdir=".")
#     text.delete('1.0', tk.END)
#     text.insert(tk.END, filename + " loaded\n\n")

# # Function to load or generate model
# def loadModel():
#     global model
#     text.delete('1.0', tk.END)
#     try:
#         X_train = np.load('model/X.txt.npy')
#         Y_train = np.load('model/Y.txt.npy')
#         text.insert(tk.END, 'Dataset contains total ' + str(X_train.shape[0]) + ' iris images from ' + str(Y_train.shape[1]) + "\n")
#     except FileNotFoundError:
#         messagebox.showerror("Error", "Training data not found")
#         return

#     if os.path.exists('model/model.json'):
#         with open('model/model.json', "r") as json_file:
#             loaded_model_json = json_file.read()
#             model = model_from_json(loaded_model_json)
#         model.load_weights("model/model_weights.h5")
#         print(model.summary())
#         try:
#             with open('model/history.pckl', 'rb') as f:
#                 data = pickle.load(f)
#             acc = data['accuracy']
#             accuracy = acc[59] * 100
#             text.insert(tk.END, "CNN Model Prediction Accuracy = " + str(accuracy) + "\n\n")
#             text.insert(tk.END, "See Black Console to view CNN layers\n")
#         except FileNotFoundError:
#             messagebox.showerror("Error", "History file not found")
#     else:
#         model = Sequential()
#         model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(32, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(256, activation='relu'))
#         model.add(Dense(108, activation='softmax'))
#         print(model.summary())
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
#         model.save_weights('model/model_weights.h5')
#         model_json = model.to_json()
#         with open("model/model.json", "w") as json_file:
#             json_file.write(model_json)
#         with open('model/history.pckl', 'wb') as f:
#             pickle.dump(hist.history, f)
#         with open('model/history.pckl', 'rb') as f:
#             data = pickle.load(f)
#         acc = data['accuracy']
#         accuracy = acc[59] * 100
#         text.insert(tk.END, "CNN Model Prediction Accuracy = " + str(accuracy) + "\n\n")
#         text.insert(tk.END, "See Black Console to view CNN layers\n")

# # Function to predict change
# def predictChange():
#     filename = filedialog.askopenfilename(initialdir="testSamples")
#     image = getIrisFeatures(filename)
#     if image is None:
#         return
#     img = cv2.resize(image, (64, 64))
#     im2arr = np.array(img)
#     im2arr = im2arr.reshape(1, 64, 64, 3)
#     img = np.asarray(im2arr)
#     img = img.astype('float32')
#     img = img / 255
#     preds = model.predict(img)
#     predict = np.argmax(preds) + 1
#     print(predict)
#     img = cv2.imread(filename)
#     img = cv2.resize(img, (600, 400))
#     img1 = cv2.imread('test.png')
#     img1 = cv2.resize(img1, (400, 200))
#     cv2.putText(img, 'Person ID Predicted from Iris Recognition is : ' + str(predict), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.imshow('Person ID Predicted from Iris Recognition is : ' + str(predict), img)
#     cv2.imshow('Iris features extracted from image', img1)
#     cv2.waitKey(0)

# # Function to display accuracy and loss graph
# def graph():
#     try:
#         with open('model/history.pckl', 'rb') as f:
#             data = pickle.load(f)
#     except FileNotFoundError:
#         messagebox.showerror("Error", "History file not found")
#         return

#     accuracy = data['accuracy']
#     loss = data['loss']
#     plt.figure(figsize=(10, 6))
#     plt.grid(True)
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy/Loss')
#     plt.plot(loss, 'ro-', color='red')
#     plt.plot(accuracy, 'ro-', color='green')
#     plt.legend(['Loss', 'Accuracy'], loc='upper left')
#     plt.title('CNN Accuracy & Loss Graph')
#     plt.show()

# # Function to close the application
# def close():
#     main.destroy()

# # Set up the GUI layout
# font = ('times', 16, 'bold')
# title = Label(main, text='Iris Recognition using Machine Learning Technique')
# title.config(bg='goldenrod2', fg='black')
# title.config(font=font)
# title.config(height=3, width=120)
# title.place(x=0, y=5)

# font1 = ('times', 12, 'bold')
# text = Text(main, height=20, width=150)
# scroll = Scrollbar(text)
# text.configure(yscrollcommand=scroll.set)
# text.place(x=50, y=120)
# text.config(font=font1)

# font1 = ('times', 13, 'bold')
# uploadButton = Button(main, text="Upload Iris Dataset", command=uploadDataset, bg='#ffb3fe')
# uploadButton.place(x=50, y=550)
# uploadButton.config(font=font1)

# modelButton = Button(main, text="Generate & Load CNN Model", command=loadModel, bg='#ffb3fe')
# modelButton.place(x=240, y=550)
# modelButton.config(font=font1)

# graphButton = Button(main, text="Accuracy & Loss Graph", command=graph, bg='#ffb3fe')
# graphButton.place(x=505, y=550)
# graphButton.config(font=font1)

# predictButton = Button(main, text="Upload Iris Test Image & Recognize", command=predictChange, bg='#ffb3fe')
# predictButton.place(x=730, y=550)
# predictButton.config(font=font1)

# exitButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
# exitButton.place(x=1050, y=550)
# exitButton.config(font=font1)

# main.config(bg='SpringGreen2')
# main.mainloop()

# from tkinter import messagebox, filedialog, simpledialog, Text, Label, Button, Scrollbar
# import tkinter as tk
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D
# from tensorflow.keras.models import Sequential, model_from_json
# import pickle
# import cv2

# # Initialize main window
# main = tk.Tk()
# main.title("Iris Recognition using Machine Learning Technique")
# main.geometry("1300x1200")

# global filename
# global model

# # Function to extract iris features
# def getIrisFeatures(image):
#     img = cv2.imread(image, 0)
#     if img is None:
#         messagebox.showerror("Error", "Image not found or unable to read")
#         return None
#     img = cv2.medianBlur(img, 5)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
#     if circles is not None:
#         height, width = img.shape
#         r = 0
#         mask = np.zeros((height, width), np.uint8)
#         for i in circles[0, :]:
#             cv2.circle(cimg, (i[0], i[1]), int(i[2]), (0, 0, 0))
#             cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=0)
#             masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)
#             _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
#             contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             x, y, w, h = cv2.boundingRect(contours[0][0])
#             crop = img[y:y+h, x:x+w]
#             r = i[2]
#         cv2.imwrite("test.png", crop)
#     else:
#         messagebox.showwarning("Warning", "No circles detected")
#         return None
#     return cv2.imread("test.png")

# # Function to upload dataset
# def uploadDataset():
#     global filename
#     filename = filedialog.askdirectory(initialdir=".")
#     text.delete('1.0', tk.END)
#     text.insert(tk.END, filename + " loaded\n\n")

# # Function to load or generate model
# def loadModel():
#     global model
#     text.delete('1.0', tk.END)
#     try:
#         X_train = np.load('model/X.txt.npy')
#         Y_train = np.load('model/Y.txt.npy')
#         text.insert(tk.END, 'Dataset contains total ' + str(X_train.shape[0]) + ' iris images from ' + str(Y_train.shape[1]) + "\n")
#     except FileNotFoundError:
#         messagebox.showerror("Error", "Training data not found")
#         return

#     if os.path.exists('model/model.json'):
#         with open('model/model.json', "r") as json_file:
#             loaded_model_json = json_file.read()
#             model = model_from_json(loaded_model_json)
#         model.load_weights("model/model_weights.h5")
#         print(model.summary())
#         try:
#             with open('model/history.pckl', 'rb') as f:
#                 data = pickle.load(f)
#             acc = data['accuracy']
#             accuracy = acc[59] * 100
#             text.insert(tk.END, "CNN Model Prediction Accuracy = " + str(accuracy) + "\n\n")
#             text.insert(tk.END, "See Black Console to view CNN layers\n")
#         except FileNotFoundError:
#             messagebox.showerror("Error", "History file not found")
#     else:
#         model = Sequential()
#         model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(32, (3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(256, activation='relu'))
#         model.add(Dense(108, activation='softmax'))
#         print(model.summary())
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
#         model.save_weights('model/model_weights.h5')
#         model_json = model.to_json()
#         with open("model/model.json", "w") as json_file:
#             json_file.write(model_json)
#         with open('model/history.pckl', 'wb') as f:
#             pickle.dump(hist.history, f)
#         with open('model/history.pckl', 'rb') as f:
#             data = pickle.load(f)
#         acc = data['accuracy']
#         accuracy = acc[59] * 100
#         text.insert(tk.END, "CNN Model Prediction Accuracy = " + str(accuracy) + "\n\n")
#         text.insert(tk.END, "See Black Console to view CNN layers\n")

# # Function to predict change
# def predictChange():
#     filename = filedialog.askopenfilename(initialdir="testSamples")
#     image = getIrisFeatures(filename)
#     if image is None:
#         return
#     img = cv2.resize(image, (64, 64))
#     im2arr = np.array(img)
#     im2arr = im2arr.reshape(1, 64, 64, 3)
#     img = np.asarray(im2arr)
#     img = img.astype('float32')
#     img = img / 255
#     preds = model.predict(img)
#     predict = np.argmax(preds) + 1
#     print(predict)
#     img = cv2.imread(filename)
#     img = cv2.resize(img, (600, 400))
#     img1 = cv2.imread('test.png')
#     img1 = cv2.resize(img1, (400, 200))
#     cv2.putText(img, 'Person ID Predicted from Iris Recognition is : ' + str(predict), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.imshow('Person ID Predicted from Iris Recognition is : ' + str(predict), img)
#     cv2.imshow('Iris features extracted from image', img1)
#     cv2.waitKey(0)

# # Function to display accuracy and loss graph
# def graph():
#     try:
#         with open('model/history.pckl', 'rb') as f:
#             data = pickle.load(f)
#     except FileNotFoundError:
#         messagebox.showerror("Error", "History file not found")
#         return

#     accuracy = data['accuracy']
#     loss = data['loss']
#     plt.figure(figsize=(10, 6))
#     plt.grid(True)
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy/Loss')
#     plt.plot(loss, 'ro-', color='red')
#     plt.plot(accuracy, 'ro-', color='green')
#     plt.legend(['Loss', 'Accuracy'], loc='upper left')
#     plt.title('CNN Accuracy & Loss Graph')
#     plt.show()

# # Function to close the application
# def close():
#     main.destroy()

# # Set up the GUI layout
# font = ('times', 16, 'bold')
# title = Label(main, text='Iris Recognition using Machine Learning Technique')
# title.config(bg='goldenrod2', fg='black')
# title.config(font=font)
# title.config(height=3, width=120)
# title.place(x=0, y=5)

# font1 = ('times', 12, 'bold')
# text = Text(main, height=20, width=150)
# scroll = Scrollbar(text)
# text.configure(yscrollcommand=scroll.set)
# text.place(x=50, y=120)
# text.config(font=font1)

# font1 = ('times', 13, 'bold')
# uploadButton = Button(main, text="Upload Iris Dataset", command=uploadDataset, bg='#ffb3fe')
# uploadButton.place(x=50, y=550)
# uploadButton.config(font=font1)

# modelButton = Button(main, text="Generate & Load CNN Model", command=loadModel, bg='#ffb3fe')
# modelButton.place(x=240, y=550)
# modelButton.config(font=font1)

# graphButton = Button(main, text="Accuracy & Loss Graph", command=graph, bg='#ffb3fe')
# graphButton.place(x=505, y=550)
# graphButton.config(font=font1)

# predictButton = Button(main, text="Upload Iris Test Image & Recognize", command=predictChange, bg='#ffb3fe')
# predictButton.place(x=730, y=550)
# predictButton.config(font=font1)

# exitButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
# exitButton.place(x=1050, y=550)
# exitButton.config(font=font1)

# main.config(bg='SpringGreen2')
# main.mainloop()



# import tkinter as tk
# from tkinter import filedialog, messagebox, Text, Label, Button, Scrollbar
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import model_from_json
# import pickle
# import cv2
# import os

# # Initialize main window
# main = tk.Tk()
# main.title("Iris Recognition using Machine Learning Technique")
# main.geometry("1300x1200")

# global filename
# global model

# def getIrisFeatures(image):
#     img = cv2.imread(image, 0)
#     if img is None:
#         messagebox.showerror("Error", "Image not found or unable to read")
#         return None
#     img = cv2.medianBlur(img, 5)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
#     if circles is not None:
#         mask = np.zeros_like(img)
#         for i in circles[0, :]:
#             cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=-1)
#             break  # Use only the first circle detected
#         masked_data = cv2.bitwise_and(img, img, mask=mask)
#         x, y, w, h = cv2.boundingRect(mask)
#         crop = masked_data[y:y+h, x:x+w]
#         cv2.imwrite("test.png", crop)
#         return cv2.imread("test.png")
#     else:
#         messagebox.showwarning("Warning", "No circles detected")
#         return None

# def uploadDataset():
#     global filename
#     filename = filedialog.askdirectory(initialdir=".")
#     if not filename:
#         return
#     text.delete('1.0', tk.END)
#     text.insert(tk.END, f"{filename} loaded\n\n")

# def loadModel():
#     global model
#     text.delete('1.0', tk.END)
#     try:
#         X_train = np.load('model/X.txt.npy')
#         Y_train = np.load('model/Y.txt.npy')
#         text.insert(tk.END, f'Dataset contains total {X_train.shape[0]} iris images from {Y_train.shape[1]} classes\n')
#     except FileNotFoundError:
#         messagebox.showerror("Error", "Training data not found")
#         return

#     if os.path.exists('model/model.json'):
#         with open('model/model.json', "r") as json_file:
#             model = model_from_json(json_file.read())
#         model.load_weights("model/model_weights.h5")
#         print(model.summary())
#         try:
#             with open('model/history.pckl', 'rb') as f:
#                 data = pickle.load(f)
#             accuracy = data['accuracy'][-1] * 100
#             text.insert(tk.END, f"CNN Model Prediction Accuracy = {accuracy:.2f}%\n\n")
#             text.insert(tk.END, "See Black Console to view CNN layers\n")
#         except FileNotFoundError:
#             messagebox.showerror("Error", "History file not found")
#     else:
#         messagebox.showerror("Error", "Model not found. Please train the model using train.py")

# def predictChange():
#     filename = filedialog.askopenfilename(initialdir="testSamples")
#     if not filename:
#         return
#     image = getIrisFeatures(filename)
#     if image is None:
#         return
#     img = cv2.resize(image, (64, 64))
#     img = np.expand_dims(img, axis=0).astype('float32') / 255.0
#     preds = model.predict(img)
#     predict = np.argmax(preds) + 1
#     print(predict)
#     img_display = cv2.imread(filename)
#     img_display = cv2.resize(img_display, (600, 400))
#     img1 = cv2.imread('test.png')
#     img1 = cv2.resize(img1, (400, 200))
#     cv2.putText(img_display, f'Person ID Predicted from Iris Recognition is: {predict}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.imshow(f'Person ID Predicted from Iris Recognition is: {predict}', img_display)
#     cv2.imshow('Iris features extracted from image', img1)
#     cv2.waitKey(0)

# def graph():
#     try:
#         with open('model/history.pckl', 'rb') as f:
#             data = pickle.load(f)
#     except FileNotFoundError:
#         messagebox.showerror("Error", "History file not found")
#         return

#     accuracy = data['accuracy']
#     loss = data['loss']
#     plt.figure(figsize=(10, 6))
#     plt.grid(True)
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy/Loss')
#     plt.plot(loss, 'ro-', color='red')
#     plt.plot(accuracy, 'ro-', color='green')
#     plt.legend(['Loss', 'Accuracy'], loc='upper left')
#     plt.title('CNN Accuracy & Loss Graph')
#     plt.show()

# def exit():
#     main.destroy()

# font1 = ('times', 14, 'bold')
# title = Label(main, text='Iris Recognition using Machine Learning Technique')
# title.config(bg='darkviolet', fg='gold')
# title.config(font=font1)
# title.config(height=3, width=120)
# title.place(x=5, y=5)

# font2 = ('times', 12, 'bold')
# uploadButton = Button(main, text="Upload CASIA Iris Image Dataset", command=uploadDataset)
# uploadButton.place(x=50, y=100)
# uploadButton.config(font=font2)

# pathlabel = Label(main)
# pathlabel.config(bg='darkviolet', fg='white')
# pathlabel.config(font=font2)
# pathlabel.place(x=50, y=150)

# generateButton = Button(main, text="Generate & Load CNN Model", command=loadModel)
# generateButton.place(x=50, y=200)
# generateButton.config(font=font2)

# predictButton = Button(main, text="Upload Test Image & Predict Person", command=predictChange)
# predictButton.place(x=50, y=250)
# predictButton.config(font=font2)

# graphButton = Button(main, text="CNN Accuracy & Loss Graph", command=graph)
# graphButton.place(x=50, y=300)
# graphButton.config(font=font2)

# exitButton = Button(main, text="Exit", command=exit)
# exitButton.place(x=50, y=350)
# exitButton.config(font=font2)

# font3 = ('times', 12, 'bold')
# text = Text(main, height=20, width=150)
# scroll = Scrollbar(text)
# text.configure(yscrollcommand=scroll.set)
# text.place(x=10, y=400)
# text.config(font=font3)

# main.config(bg='darkviolet')
# main.mainloop()



# import tkinter as tk
# from tkinter import filedialog, messagebox, Text, Label, Button, Scrollbar
# import numpy as np
# from tensorflow.keras.models import model_from_json
# import pickle
# import cv2
# import os

# # Initialize main window
# main = tk.Tk()
# main.title("Iris Recognition using Machine Learning Technique")
# main.geometry("1300x1200")

# global filename
# global model

# def getIrisFeatures(image):
#     img = cv2.imread(image, 0)
#     if img is None:
#         messagebox.showerror("Error", "Image not found or unable to read")
#         return None
#     img = cv2.medianBlur(img, 5)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
#     if circles is not None:
#         mask = np.zeros_like(img)
#         for i in circles[0, :]:
#             cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=-1)
#             break  # Use only the first circle detected
#         masked_data = cv2.bitwise_and(img, img, mask=mask)
#         x, y, w, h = cv2.boundingRect(mask)
#         crop = masked_data[y:y+h, x:x+w]
#         cv2.imwrite("test.png", crop)
#         return cv2.imread("test.png")
#     else:
#         messagebox.showwarning("Warning", "No circles detected")
#         return None

# def uploadDataset():
#     global filename
#     filename = filedialog.askdirectory(initialdir=".")
#     if not filename:
#         return
#     text.delete('1.0', tk.END)
#     text.insert(tk.END, f"{filename} loaded\n\n")

# def loadModel():
#     global model
#     text.delete('1.0', tk.END)
#     try:
#         X_train = np.load('model/X.txt.npy')
#         Y_train = np.load('model/Y.txt.npy')
#         text.insert(tk.END, f'Dataset contains total {X_train.shape[0]} iris images from {Y_train.shape[1]} classes\n')
#     except FileNotFoundError:
#         messagebox.showerror("Error", "Training data not found")
#         return

#     if os.path.exists('model/model.json'):
#         with open('model/model.json', "r") as json_file:
#             model = model_from_json(json_file.read())
#         model.load_weights("model/model_weights.h5")
#         print(model.summary())
#         try:
#             with open('model/history.pckl', 'rb') as f:
#                 data = pickle.load(f)
#             accuracy = data['accuracy'][-1] * 100
#             text.insert(tk.END, f"CNN Model Prediction Accuracy = {accuracy:.2f}%\n\n")
#             text.insert(tk.END, "See Black Console to view CNN layers\n")
#         except FileNotFoundError:
#             messagebox.showerror("Error", "History file not found")
#     else:
#         messagebox.showerror("Error", "Model not found. Please train the model using train.py")

# def predictChange():
#     filename = filedialog.askopenfilename(initialdir="testSamples")
#     if not filename:
#         return
#     image = getIrisFeatures(filename)
#     if image is None:
#         return
#     img = cv2.resize(image, (64, 64))
#     img = np.expand_dims(img, axis=0).astype('float32') / 255.0
#     preds = model.predict(img)
#     predict = np.argmax(preds) + 1
#     print(predict)
#     img_display = cv2.imread(filename)
#     img_display = cv2.resize(img_display, (600, 400))
#     img1 = cv2.imread('test.png')
#     img1 = cv2.resize(img1, (400, 200))
#     cv2.putText(img_display, f'Person ID Predicted from Iris Recognition is: {predict}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.imshow(f'Person ID Predicted from Iris Recognition is: {predict}', img_display)
#     cv2.imshow('Iris features extracted from image', img1)
#     cv2.waitKey(0)

# def exit():
#     main.destroy()

# font1 = ('times', 14, 'bold')
# title = Label(main, text='Iris Recognition using Machine Learning Technique')
# title.config(bg='darkviolet', fg='gold')
# title.config(font=font1)
# title.config(height=3, width=120)
# title.place(x=5, y=5)

# font2 = ('times', 12, 'bold')
# uploadButton = Button(main, text="Upload CASIA Iris Image Dataset", command=uploadDataset)
# uploadButton.place(x=50, y=100)
# uploadButton.config(font=font2)

# pathlabel = Label(main)
# pathlabel.config(bg='darkviolet', fg='white')
# pathlabel.config(font=font2)
# pathlabel.place(x=50, y=150)

# predictButton = Button(main, text="Upload Test Image & Predict Person", command=predictChange)
# predictButton.place(x=50, y=200)
# predictButton.config(font=font2)

# exitButton = Button(main, text="Exit", command=exit)
# exitButton.place(x=50, y=250)
# exitButton.config(font=font2)

# font3 = ('times', 12, 'bold')
# text = Text(main, height=20, width=150)
# scroll = Scrollbar(text)
# text.configure(yscrollcommand=scroll.set)
# text.place(x=10, y=300)
# text.config(font=font3)

# main.config(bg='darkviolet')
# main.mainloop()

# import tkinter as tk
# from tkinter import filedialog, messagebox, Text, Label, Button, Scrollbar
# import numpy as np
# from tensorflow.keras.models import model_from_json
# import pickle
# import cv2
# import os

# # Initialize main window
# main = tk.Tk()
# main.title("Iris Recognition using Machine Learning Technique")
# main.geometry("1300x1200")

# global filename
# global model

# def getIrisFeatures(image):
#     img = cv2.imread(image, 0)
#     if img is None:
#         messagebox.showerror("Error", "Image not found or unable to read")
#         return None
#     img = cv2.medianBlur(img, 5)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
#     if circles is not None:
#         mask = np.zeros_like(img)
#         for i in circles[0, :]:
#             cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=-1)
#             break  # Use only the first circle detected
#         masked_data = cv2.bitwise_and(img, img, mask=mask)
#         x, y, w, h = cv2.boundingRect(mask)
#         crop = masked_data[y:y+h, x:x+w]
#         cv2.imwrite("test.png", crop)
#         return cv2.imread("test.png")
#     else:
#         messagebox.showwarning("Warning", "No eye iris is found")
#         return None

# def uploadDataset():
#     global filename
#     filename = filedialog.askdirectory(initialdir=".")
#     if not filename:
#         return
#     text.delete('1.0', tk.END)
#     text.insert(tk.END, f"{filename} loaded\n\n")

# def loadModel():
#     global model
#     text.delete('1.0', tk.END)
#     try:
#         X_train = np.load('model/X.txt.npy')
#         Y_train = np.load('model/Y.txt.npy')
#         text.insert(tk.END, f'Dataset contains total {X_train.shape[0]} iris images from {Y_train.shape[1]} classes\n')
#     except FileNotFoundError:
#         messagebox.showerror("Error", "Training data not found")
#         return

#     if os.path.exists('model/model.json'):
#         with open('model/model.json', "r") as json_file:
#             model = model_from_json(json_file.read())
#         model.load_weights("model/model_weights.h5")
#         print(model.summary())
#         try:
#             with open('model/history.pckl', 'rb') as f:
#                 data = pickle.load(f)
#             accuracy = data['accuracy'][-1] * 100
#             text.insert(tk.END, f"CNN Model Prediction Accuracy = {accuracy:.2f}%\n\n")
#             text.insert(tk.END, "See Black Console to view CNN layers\n")
#         except FileNotFoundError:
#             messagebox.showerror("Error", "History file not found")
#     else:
#         messagebox.showerror("Error", "Model not found. Please train the model using train.py")

# def predictChange():
#     filename = filedialog.askopenfilename(initialdir="testSamples")
#     if not filename:
#         return
#     image = getIrisFeatures(filename)
#     if image is None:
#         return
#     img = cv2.resize(image, (64, 64))
#     img = np.expand_dims(img, axis=0).astype('float32') / 255.0
#     preds = model.predict(img)
#     predict = np.argmax(preds) + 1
#     messagebox.showinfo("Prediction", f'Iris found! Person ID predicted: {predict}')
#     img_display = cv2.imread(filename)
#     img_display = cv2.resize(img_display, (600, 400))
#     img1 = cv2.imread('test.png')
#     img1 = cv2.resize(img1, (400, 200))
#     cv2.putText(img_display, f'Person ID Predicted from Iris Recognition is: {predict}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.imshow(f'Person ID Predicted from Iris Recognition is: {predict}', img_display)
#     cv2.imshow('Iris features extracted from image', img1)
#     cv2.waitKey(0)

# def exit():
#     main.destroy()

# font1 = ('times', 14, 'bold')
# title = Label(main, text='Iris Recognition using Machine Learning Technique')
# title.config(bg='darkviolet', fg='gold')
# title.config(font=font1)
# title.config(height=3, width=120)
# title.place(x=5, y=5)

# font2 = ('times', 12, 'bold')
# uploadButton = Button(main, text="Upload CASIA Iris Image Dataset", command=uploadDataset)
# uploadButton.place(x=50, y=100)
# uploadButton.config(font=font2)

# pathlabel = Label(main)
# pathlabel.config(bg='darkviolet', fg='white')
# pathlabel.config(font=font2)
# pathlabel.place(x=50, y=150)

# predictButton = Button(main, text="Upload Test Image & Predict Person", command=predictChange)
# predictButton.place(x=50, y=200)
# predictButton.config(font=font2)

# exitButton = Button(main, text="Exit", command=exit)
# exitButton.place(x=50, y=250)
# exitButton.config(font=font2)

# font3 = ('times', 12, 'bold')
# text = Text(main, height=20, width=150)
# scroll = Scrollbar(text)
# text.configure(yscrollcommand=scroll.set)
# text.place(x=10, y=300)
# text.config(font=font3)

# main.config(bg='darkviolet')
# main.mainloop()


import tkinter as tk
from tkinter import filedialog, messagebox, Text, Label, Button, Scrollbar
import numpy as np
from tensorflow.keras.models import model_from_json
import pickle
import cv2
import os

# Initialize main window
main = tk.Tk()
main.title("Iris Recognition using Machine Learning Technique")
main.geometry("1300x1200")

global filename
global model

def getIrisFeatures(image):
    img = cv2.imread(image, 0)
    if img is None:
        messagebox.showerror("Error", "Image not found or unable to read")
        return None
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
        cv2.imwrite("test.png", crop)
        return cv2.imread("test.png")
    else:
        messagebox.showwarning("Warning", "No eye iris is found")
        return None

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    if not filename:
        return
    text.delete('1.0', tk.END)
    text.insert(tk.END, f"{filename} loaded\n\n")

def loadModel():
    global model
    text.delete('1.0', tk.END)
    try:
        X_train = np.load('model/X.txt.npy')
        Y_train = np.load('model/Y.txt.npy')
        text.insert(tk.END, f'Dataset contains total {X_train.shape[0]} iris images from {Y_train.shape[1]} classes\n')
    except FileNotFoundError:
        messagebox.showerror("Error", "Training data not found")
        return

    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            model = model_from_json(json_file.read())
        model.load_weights("model/model_weights.h5")
        print(model.summary())
        try:
            with open('model/history.pckl', 'rb') as f:
                data = pickle.load(f)
            accuracy = data['accuracy'][-1] * 100
            text.insert(tk.END, f"CNN Model Prediction Accuracy = {accuracy:.2f}%\n\n")
            text.insert(tk.END, "See Black Console to view CNN layers\n")
        except FileNotFoundError:
            messagebox.showerror("Error", "History file not found")
    else:
        messagebox.showerror("Error", "Model not found. Please train the model using train.py")

def predictChange():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    if not filename:
        return
    image = getIrisFeatures(filename)
    if image is None:
        return
    img = cv2.resize(image, (64, 64))
    img = np.expand_dims(img, axis=0).astype('float32') / 255.0
    preds = model.predict(img)
    predict = np.argmax(preds) + 1
    messagebox.showinfo("Prediction", f'Iris found! Person ID predicted: {predict}')
    img_display = cv2.imread(filename)
    img_display = cv2.resize(img_display, (600, 400))
    img1 = cv2.imread('test.png')
    img1 = cv2.resize(img1, (400, 200))
    cv2.putText(img_display, f'Person ID Predicted from Iris Recognition is: {predict}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow(f'Person ID Predicted from Iris Recognition is: {predict}', img_display)
    cv2.imshow('Iris features extracted from image', img1)
    cv2.waitKey(0)

def exit():
    main.destroy()

font1 = ('times', 14, 'bold')
title = Label(main, text='Iris Recognition using Machine Learning Technique')
title.config(bg='darkviolet', fg='gold')
title.config(font=font1)
title.config(height=3, width=120)
title.place(x=5, y=5)

font2 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload CASIA Iris Image Dataset", command=uploadDataset)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font2)

pathlabel = Label(main)
pathlabel.config(bg='darkviolet', fg='white')
pathlabel.config(font=font2)
pathlabel.place(x=50, y=150)

predictButton = Button(main, text="Upload Test Image & Predict Person", command=predictChange)
predictButton.place(x=50, y=200)
predictButton.config(font=font2)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50, y=250)
exitButton.config(font=font2)

font3 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=300)
text.config(font=font3)

main.config(bg='darkviolet')
main.mainloop()
