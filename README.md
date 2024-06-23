# Project Emojify
------------------------------------------------------------------------

## About
Nowadays, Humans are using several emojis or avatars to show moods or feelings in the online mode of communication which is not basically face to face. Emojis act as nonverbal cues for humans. Without them, online chatting and other areas seem emotionless which is not favorable for humans due to the fact that Humans are Social Animals. They have become a crucial part of emotion recognition, online chatting, brand emotion, product review, and a lot more.

------------------------------------------------------------------------

## Goal
The main goal of this project is to learn about human facial expressions and then show accurate Avatar or Emoji similar to facial expression.

------------------------------------------------------------------------

## About the Datasets
FER-2013 dataset ( Facial Expression Recognition) from Kaggle.The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image. The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

Haarcascades classifiers are used in this project for the detection of the face which is taken from OpenCV. Due to Cuda drivers, haarcascades for Cuda have been used in this project. Normal haarcascades
can also be used depending upon the building system.

## Caragory of facial emotions:

- 0:angry
- 1:disgust
- 2:fear
- 3:happy
- 4:neutral
- 5:sad
- 6:surprise

## Facial Expressions used for detection:
- 0:Angry
- 1:Disgusted
- 2:Fearful
- 3:Happy
- 4:Neutral
- 5:Sad
- 6:Surprised

------------------------------------------------------------------------

## CNN to Recognize Facial Emotion
With advancements in Computer vision in AI and Deep learning in ML, it is now possible to detect human emotions from images. This project classifies Human facial expressions to filter and map corresponding emojis or avatars. This project creates a Convolution Neural Network (CNN) architecture and feed the FER2013 dataset to the model so that it can recognize emotion from images. This project will create and build the CNN model using the Keras layers in various steps.

To build the network we use two dense layers, one flatten layer and four conv2D layers. We will use the Softmax equation to generate the model output.

------------------------------------------------------------------------

## Requirements of this project
Basic Requirments:
- FER2013 datasets from Kaggle.
- Knowledge of python and AI&ML.
- Knowledge of Jupyter from Anaconda.
- Emojis/Avatars

Required Libraries:
- Tensorflow
- Keras
- OpenCV
- h5py
- Numpy
- Tkinter

------------------------------------------------------------------------

## Additional Info:
This project has been built using tensorflow-gpu and Ubuntu 22.04 as host OS using Jupyter Notebook. Errors can arise due to incompatibility, therefore several changes can be made according to the IDE and host OS.

![alt text](<WhatsApp Image 2024-06-23 at 9.58.44 PM.jpeg>)
![alt text](<WhatsApp Image 2024-06-23 at 9.58.45 PM.jpeg>)