import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import os

cascade_file = r'C:\Users\home\Downloads\Emoji_Based_HumanReactions-main\Emoji_Based_HumanReactions-main\haarcascade_frontalface_default.xml'

network_model = Sequential()

network_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
network_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
network_model.add(MaxPooling2D(pool_size=(2, 2)))
network_model.add(Dropout(0.25))
network_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
network_model.add(MaxPooling2D(pool_size=(2, 2)))
network_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
network_model.add(MaxPooling2D(pool_size=(2, 2)))
network_model.add(Dropout(0.25))
network_model.add(Flatten())
network_model.add(Dense(1024, activation='relu'))
network_model.add(Dropout(0.5))
network_model.add(Dense(7, activation='softmax'))

network_model.load_weights('model.h5')

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dict = {0:cur_path +"/emojis/angry.png",1:cur_path +"/emojis/disgusted.png",2:cur_path +"/emojis/fearful.png",3:cur_path +"/emojis/happy.png",4:cur_path +"/emojis/neutral.png",5:cur_path +"/emojis/sad.png",6:cur_path +"/emojis/surpriced.png"}
#emoji_dict = ["/emoji/angry.png", "/emoji/disgusted.png", "/emoji/fearful.png", "/emoji/happy.png", "/emoji/neutral.png", "/emoji/sad.png", "/emoji/surpriced.png"]
print(emoji_dict)
global frame # Bliver ikke brugt
frame = np.zeros((480, 640, 3), dtype=np.uint8) # Bliver ikke brugt
global capture # Bliver ikke brugt 
show_text = [0] # Bliver ikke brugt 

def show_camera():
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        ok, camera_frame = capture.read()

        if not ok:
            continue
            
        camera_frame = cv2.resize(camera_frame, (600, 500))

        bounding_box = cv2.CascadeClassifier(cascade_file)

        gray_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
        
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5) #Scale reducer image size, neighbors = quality of deceted
        for (x, y, w, h) in num_faces:
            cv2.rectangle(camera_frame, (x, y-50), (x+w, y+h+10), (0,128,0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = network_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(camera_frame, emotion_dict[maxindex], (x+20, y-60), font, 1, (0,0,255), 2, cv2.LINE_AA)
            show_emoji_by_index(maxindex)

        frame = camera_frame.copy()
        print("--frame---")
        pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
        cv2.waitKey(125)
        print("---wait---")
        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def show_emoji_by_index(index):
    emoji_frame = cv2.imread(emoji_dict[index])
    img2 = Image.fromarray(emoji_frame)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    emoji_label.imgtk2 = imgtk2
    emoji_text.configure(text=emotion_dict[index], font=('arial', 40, 'bold'))
    emoji_label.configure(image=imgtk2)

if __name__ == '__main__':
    print("hek")
    root = tk.Tk()
    heading2 = Label(root, text="Python Project", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
    heading2.pack()
    camera_label = tk.Label(master=root, padx=50, bd=10)
    emoji_label = tk.Label(master=root, bd=10)
    emoji_text = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    camera_label.pack(side=LEFT)
    camera_label.place(x=100, y=300)
    emoji_text.pack()
    emoji_text.place(x=1300, y=180)
    emoji_label.pack(side=RIGHT)
    emoji_label.place(x=700, y=300)
    root.title("Python Project")
    root.geometry("1400x1000+100+10")
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="blue", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)
    print("-------------------------")
    show_camera()
    #show_emoji()
    #threading.Thread(target=show_camera).start()
    #threading.Thread(target=show_emoji).start()
    #root.mainloop()
