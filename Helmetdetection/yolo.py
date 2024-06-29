import numpy as np
import cv2 as cv
import subprocess
import time
import os
from yoloDetection import detectObject, displayImage
import sys


global class_labels
global cnn_model
global cnn_layer_names

def loadLibraries(): #function to load yolov3 model weight and class labels
        global class_labels
        global cnn_model
        global cnn_layer_names
        class_labels = open('yolov3model/yolov3-labels').read().strip().split('\n') #reading labels from yolov3 model
        print(str(class_labels)+" == "+str(len(class_labels)))
        cnn_model = cv.dnn.readNetFromDarknet('yolov3model/yolov3.cfg', 'yolov3model/yolov3.weights') #reading model
        cnn_layer_names = cnn_model.getLayerNames() #getting layers from cnn model
        cnn_layer_names = [cnn_layer_names[i - 1] for i in cnn_model.getUnconnectedOutLayers()] #assigning all layers

def detectFromImage(imagename): #function to detect object from images
        indexno=0
        #random colors to assign unique color to each label
        label_colors = (0,255,0)#np.random.randint(0,255,size=(len(class_labels),3),dtype='uint8')
        try:
                image = cv.imread(imagename) #image reading
                image_height, image_width = image.shape[:2] #converting image to two dimensional array
        except:
                raise 'Invalid image path'
        finally:
                image, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels,indexno)#calling detection function
                displayImage(image,0)#display image with detected objects label

def detectFromVideo(videoFile): #function to read objects from video
        
        #random colors to assign unique color to each label
        label_colors = (0,255,0)#np.random.randint(0,255,size=(len(class_labels),3),dtype='uint8')
        indexno = 0
        try:
                
                video = cv.VideoCapture(videoFile)
                frame_height, frame_width = None, None  #reading video from given path
                video_writer = None
        except:
                raise 'Unable to load video'
        finally:
                while True:
                        frame_grabbed, frames = video.read() #taking each frame from video
                        #print(frame_grabbed)
                        if not frame_grabbed: #condition to check whether video loaded or not
                                break
                        if frame_width is None or frame_height is None:
                                frame_height, frame_width = frames.shape[:2] #detecting object from frame
                        frames, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, frame_height, frame_width, frames, label_colors, class_labels,indexno)
                        #displayImage(frames,index)
                        #indexno = indexno + 1
                        print(indexno)
                        if indexno == 5:
                            video.release()    
                            break

        print ("Releasing resources")
        #video_writer.release()
        video.release()


if __name__ == '__main__':
        loadLibraries()
        print("sample commands to run code with image or video")
        print("python yolo.py image input_image_path")
        print("python yolo.py video input_video_path")
        if len(sys.argv) == 3:
                if sys.argv[1] == 'image':
                        detectFromImage(sys.argv[2])
                elif sys.argv[1] == 'video':
                        detectFromVideo(sys.argv[2])
                else:
                        print("invalid input")
        else:
                print("follow sample command to run code")

                
	#video_path = None
	#video_output_path = "out.avi"
