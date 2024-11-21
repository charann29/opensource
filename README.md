This project implements a Real/Fake Logo detection system using deep learning. The goal is to train a Convolutional Neural Network (CNN) to distinguish between real and fake logos. The dataset consists of images of both genuine and fake logos, and the model is trained to classify these images into their respective categories.

Loading and Preprocessing Images
The script first loads and preprocesses the images from the specified paths. Images are resized to a common size and converted to NumPy arrays. The Inception V3 preprocessing function is applied to the image arrays.

Concatenating Data
Fake and genuine data from the training and test sets are concatenated to create the final training and test sets. Labels are also assigned (0 for fake, 1 for genuine).

Splitting the Dataset
The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

Defining the CNN Model
A simple CNN model is defined using the Keras Sequential API. It consists of convolutional layers, max-pooling layers, and dense layers.

Compiling the Model
The model is compiled with the Adam optimizer and binary cross-entropy loss, as it is a binary classification problem. The accuracy metric is used for evaluation.

Training the Model
The model is trained on the training set with 10 epochs and a batch size of 32. Validation data is used to monitor the model's performance during training.

Evaluating the Model
The trained model is evaluated on the test set, and accuracy along with the confusion matrix is printed.a variety of Python libraries for different tasks. Here's a breakdown of the libraries used and their purposes:

Installed via pip install yacs.
Likely used for configuration management.
pycocotools:

from pycocotools.coco import COCO
Used for handling COCO dataset annotations.

import numpy as np
Used for numerical operations.

import skimage.io as io
Used for image input and output operations.

import matplotlib.pyplot as plt
Used for plotting and visualization.

import os
Used for interacting with the operating system, such as file path manipulations.

import torch
Used for deep learning tasks, such as loading and running models.

from pathlib import Path
Used for object-oriented filesystem paths.

import config
Used for configuration management (custom module specific to the project).
torchvision.transforms:

import torchvision.transforms as T
Used for image transformations and preprocessing.
PIL (Python Imaging Library):

from PIL import Image
Used for opening and manipulating images.

import json
Used for handling JSON data, such as loading annotation files.
These libraries collectively support the workflow for downloading datasets, preprocessing annotations, visualizing data, setting up and training models, and performing inference with trained models.

![WhatsApp Image 2024-06-20 at 21 32 17_6c8e4f86](https://github.com/koushi4500/koushik-demo/assets/161230937/cd3643bd-c5c0-4ae6-9e72-227f9e8018d9)


![Uploading WhatsApp Image 2024-06-20 at 21.32.55_c5604f72.jpg…]()

