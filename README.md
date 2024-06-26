
# Fake Logo Dectection

This project implements a Real/Fake Logo detection system using deep learning. The goal is to train a Convolutional Neural Network (CNN) to distinguish between real and fake logos. The dataset consists of images of both genuine and fake logos, and the model is trained to classify these images into their respective categories.



## Table of contents
 Loading and Preprocessing Images, Concatenating Data, Splitting the Dataset, Defining the CNN Model, Compiling the Model, Training the Model, Evaluating the Model, Installation, Usage, Contributing, License, Loading and Preprocessing Images.


## Description
The script first loads and preprocesses the images from the specified paths. Images are resized to a common size and converted to NumPy arrays. The Inception V3 preprocessing function is applied to the image arrays.

Concatenating Data:
Fake and genuine data from the training and test sets are concatenated to create the final training and test sets. Labels are also assigned (0 for fake, 1 for genuine).

Splitting the Dataset:
The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

Defining the CNN Model:
A simple CNN model is defined using the Keras Sequential API. It consists of convolutional layers, max-pooling layers, and dense layers.

Compiling the Model:
The model is compiled with the Adam optimizer and binary cross-entropy loss, as it is a binary classification problem. The accuracy metric is used for evaluation.

Training the Model:
The model is trained on the training set with 10 epochs and a batch size of 32. Validation data is used to monitor the model's performance during training.

Evaluating the Model:
The trained model is evaluated on the test set, and accuracy along with the confusion matrix is printed.

## Installation

Install necssery dependences by running

```bash
pip install -r requirements.txt
```
## libraries used
yacs: Configuration management,

pycocotools: Handling COCO dataset annotations

numpy: Numerical operations

skimage.io: Image input and output operations

matplotlib: Plotting and visualization

os: Interacting with the operating system

torch: Deep learning tasks

pathlib: Object-oriented filesystem paths

torchvision.transforms: Image transformations and preprocessing

PIL: Opening and manipulating images

json: Handling JSON data

## Usage
Load and preprocess images: Prepare your dataset by loading and preprocessing images.

Concatenate data: Combine fake and genuine images to create training and test sets.

Split the dataset: Use train_test_split to divide the dataset into training and testing sets.

Define and compile the model: Create and compile the CNN model using Keras.

Train the model: Train the model on the training set, monitoring its performance with validation data.

Evaluate the model: Evaluate the trained model on the test set and print the accuracy and confusion matrix.
## Screenshots/Running Tests
![WhatsApp Image 2024-06-20 at 21 32 15_b4f09e6c](https://github.com/koushi4500/koushik-demo/assets/161230937/689a573d-b6ad-49a1-aaf2-95ab95a8bc80)

![WhatsApp Image 2024-06-20 at 21 32 53_50eb7d43](https://github.com/koushi4500/koushik-demo/assets/161230937/92147119-1870-45f0-a33c-0668b976ba76)







