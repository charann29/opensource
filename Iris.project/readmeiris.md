*Iris Recognition using Machine Learning Technique*

This project focuses on iris recognition using machine learning techniques. The application utilizes a Convolutional Neural Network (CNN) model to identify individuals based on their iris patterns.

*Description*

Iris recognition is a biometric identification method that uses pattern-recognition techniques based on high-resolution images of the irises of an individual's eyes. This project leverages machine learning to perform iris recognition. The main features include:

- Uploading a dataset of iris images.
  
- Loading a pre-trained CNN model.
  
- Predicting the identity of a person based on a provided test iris image.
  

*How To Run*

-Clone the repository & run the application main.py

`python main.py`

*Features*


- Upload Dataset

   Click on the "Upload CASIA Iris Image Dataset" button to load the dataset directory. The application will display the number of images and classes loaded.( For example , if you are using this model for office biometric upload your staff's iris as dataset)
   Below is the interface in which we have uploaded iris datasets.

  ![WhatsApp Image 2024-06-23 at 22 29 18_077c0e95](https://github.com/SaiSruthisri/Sruthi_cmr_opensource/assets/148372065/d414cc9a-2080-47c2-a61f-ad560cdeabb4)


- Load Model

  The application automatically loads the CNN model architecture and weights if present in the model directory.

- Predict Person

  Click on the "Upload Test Image & Predict Person" button to select a test image.*If the Iris matches with the dataset no messages will be produced*.Below is the collage image of our input image & output recived for an clear & existing iris image👇👇
  
  
![WhatsApp Image 2024-06-23 at 23 45 59_cc1a521d](https://github.com/SaiSruthisri/Sruthi_cmr_opensource/assets/148372065/d8772c68-6054-4128-a234-3217a99ef379)

  
 But if any unclear iris or other images are uploaded then we get warning message as below "No eye iris is found". 👇👇
 

  ![WhatsApp Image 2024-06-23 at 22 32 33_ae9deded](https://github.com/SaiSruthisri/Sruthi_cmr_opensource/assets/148372065/5c693e33-387f-4048-a372-20fb946cb9bf)


*Troubleshooting*

- Model Not Found : Ensure the model.json and model_weights.h5 files are present in the model directory.

- Training Data Not Found : Ensure the X.txt.npy and Y.txt.npy files are present in the model directory.

- No Iris Found : Ensure the test image is clear and contains a visible iris.

*Acknowledgments*

- CASIA Iris Image Database for providing the dataset used in this project.

- TensorFlow and Keras for providing the machine learning framework.

- Tkinter for the GUI components.