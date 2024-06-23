# Import required Libraries
import tkinter as tk
from tkinter import *
from PIL import Image
from PIL import ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

# Build the convolution network architecture
face_model = Sequential()
face_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
face_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
face_model.add(MaxPooling2D(pool_size=(2, 2)))
face_model.add(Dropout(0.25))
face_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
face_model.add(MaxPooling2D(pool_size=(2, 2)))
face_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
face_model.add(MaxPooling2D(pool_size=(2, 2)))
face_model.add(Dropout(0.25))
face_model.add(Flatten())
face_model.add(Dense(1024, activation='relu'))
face_model.add(Dropout(0.5))
face_model.add(Dense(7, activation='softmax'))

# Load the saved weights
face_model.load_weights('recognition_model.h5')

# Disable OpenCL
cv2.ocl.setUseOpenCL(False)

# Create Datasets Dictionaries
facial_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
emojis_dict = {0:"emojis/angry.png", 1:"emojis/disgusted.png", 2:"emojis/fearful.png", 3:"emojis/happy.png", 4:"emojis/neutral.png", 5:"emojis/sad.png", 6:"emojis/surprised.png"}

# Global variables
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]

# Function to get face captured and recognize emotion
def Capture_Image():
    global cap1
    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("cant open the camera1")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(600,500))
    # It will detect the face in the video and bound it with a rectangular box
    bound_box = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    n_faces = bound_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in n_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_frame = gray_frame[y:y + h, x:x + w]
        crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)
        prediction = face_model.predict(crop_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, facial_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex

    if flag1 is None:
        print ("Error!")

    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB) #to store the image
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, Capture_Image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

# Function for showing Emoji According to Facial Expression
def Get_Emoji():
    frame2=cv2.imread(emojis_dict[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=facial_dict[show_text[0]],font=('arial',45,'bold'))
    lmain2.configure(image=imgtk2)
    lmain2.after(10, Get_Emoji)

# GUI Window to show captured image with emoji
if __name__ == '__main__':
    root=tk.Tk()
    heading = Label(root,bg='black')
    heading.pack()
    heading2=Label(root,text="Emojify",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')#to label the output
    heading2.pack()
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)
    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    root.title("Emojify")
    root.geometry("1400x900+100+10")
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    Capture_Image()
    Get_Emoji()
    root.mainloop()
