import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os

def detectObject(CNNnet, total_layer_names, image_height, image_width, image, name_colors, class_labels,indexno,  
            Boundingboxes=None, confidence_value=None, class_ids=None, ids=None, detect=True):
    
    if detect:
        blob_object = cv.dnn.blobFromImage(image,1/255.0,(416, 416),swapRB=True,crop=False)
        CNNnet.setInput(blob_object)
        cnn_outs_layer = CNNnet.forward(total_layer_names)
        Boundingboxes, confidence_value, class_ids = listBoundingBoxes(cnn_outs_layer, image_height, image_width, 0.5)
        ids = cv.dnn.NMSBoxes(Boundingboxes, confidence_value, 0.5, 0.3)
        if Boundingboxes is None or confidence_value is None or ids is None or class_ids is None:
           raise '[ERROR] unable to draw boxes.'
        image,option = labelsBoundingBoxes(image, Boundingboxes, confidence_value, class_ids, ids, name_colors, class_labels,indexno)

    return image,option


def labelsBoundingBoxes(image, Boundingbox, conf_thr, classID, ids, color_names, predicted_labels,indexno):
    option = 0
    if len(ids) > 0:
        for i in ids.flatten():
            # draw boxes
            xx, yy = Boundingbox[i][0], Boundingbox[i][1]
            width, height = Boundingbox[i][2], Boundingbox[i][3]
            
            class_color = (0,255,0)#[int(color) for color in color_names[classID[i]]]

            cv.rectangle(image, (xx, yy), (xx+width, yy+height), class_color, 2)
            print(classID[i])
            if classID[i] <= 1:
                text_label = "{}: {:4f}".format(predicted_labels[classID[i]], conf_thr[i])
                #displayImage(image,indexno)
                cv.putText(image, text_label, (xx, yy-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
                option = 1

    return image,option


def listBoundingBoxes(image, image_height, image_width, threshold_conf):
    box_array = []
    confidence_array = []
    class_ids_array = []

    for img in image:
        for obj_detection in img:
            detection_scores = obj_detection[5:]
            class_id = np.argmax(detection_scores)
            confidence_value = detection_scores[class_id]
            if confidence_value > threshold_conf and class_id <= 1:
                Boundbox = obj_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                center_X, center_Y, box_width, box_height = Boundbox.astype('int')

                xx = int(center_X - (box_width / 2))
                yy = int(center_Y - (box_height / 2))

                box_array.append([xx, yy, int(box_width), int(box_height)])
                confidence_array.append(float(confidence_value))
                class_ids_array.append(class_id)

    return box_array, confidence_array, class_ids_array

def displayImage(image,index):
    #cv.imwrite('bikes/'+str(index)+'.jpg',image)
    #index = index + 1
    cv.imshow("Final Image", image)
    cv.waitKey(0)


