# Emoji_Based_HumanReactions

## 1. Motivation

This Project detects the human reaction by using the CNN model and create  emoji Happy,Sad,Netural,Fearful,Disgust and Angry 

## 2. Dataset Used:

https://www.kaggle.com/datasets/msambare/fer2013


Dataset from https://www.kaggle.com/datasets/msambare/fer2013 contains the data with data Image Properties: 48 x 48 pixels (2304 bytes) labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

## 3. Why FER-2013 DataSet

Fer2013 is a challenging dataset. The images are not aligned and some of them are uncorrectly labeled as we can see from the following images. Moreover, some samples do not contain faces.

  ![image](https://user-images.githubusercontent.com/120184924/206932932-304c7cb8-0903-478a-a61e-90b93c9474ea.png)


This makes the classification harder because the model have to generalize well and be robust to incorrect data. The best accuracy results obtained on this dataset, as far as I know, is 93.6% described in this paper: [Facial Expression Recognition using Convolutional Neural Networks: State of the Art, Pramerdorfer & al. 2016]

# 4. How to Use ?
## 4.1 Install Dependencies:

1. Pandas
2. Numpy
3. CV2
4. datetime
5. Tensorflow
6. Tkinter
7. OS
8. Seaborn
9. Matplotlib

Anaconda and jupyter notebook is a better way to install dependencies 


## 4.2 Download the data:

1. Download the Face Landmarks model with the Fer2013 dataset.
     - [Kaggle FER-2013 Data for the facial datasets ](https://www.kaggle.com/datasets/msambare/fer2013)
 2. Unzip the downloaded files 

## 4.3 Train the model

1. Choose your parameters in 'parameters.py'
2. Launch training:

## 4.4 Optimize Hyperparameters 


1. optimize the dependednt parameters 
2. Created sam.py and the developed Webapp using Tkinter.
3. When the program is executed it opens the webcam and detects the human face and gives the output as emoji

![WhatsApp Image 2024-06-20 at 9 47 47 PM](https://github.com/VamshiKrishna660/First-Repo/assets/146640385/fbafc3b7-fcd0-4c02-a9e3-248f04f034ae)





