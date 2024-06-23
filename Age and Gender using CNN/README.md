# Age and Gender Detection

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a system for detecting the age and gender of a person from their photo using deep learning models in OpenCV.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Libraries and Their Usage](#libraries-and-their-usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

This project leverages deep learning models to identify the age and gender of a person from a given image. The system uses OpenCV's DNN module to load pre-trained models for face detection, age prediction, and gender prediction.

<h2>Examples :</h2>
<p><b>NOTE:- I downloaded the images from Google,if you have any query or problem i can remove them, i just used it for Educational purpose.</b></p>

    >python detect.py --image girl1.jpg
    Gender: Female
    Age: 25-32 years
    
<img src="Example/Detecting age and gender girl1.png">

    >python detect.py --image girl2.jpg
    Gender: Female
    Age: 8-12 years
    
<img src="Example/Detecting age and gender girl2.png">

    >python detect.py --image kid1.jpg
    Gender: Male
    Age: 4-6 years    
    
<img src="Example/Detecting age and gender kid1.png">

    >python detect.py --image kid2.jpg
    Gender: Female
    Age: 4-6 years  
    
<img src="Example/Detecting age and gender kid2.png">

    >python detect.py --image man1.jpg
    Gender: Male
    Age: 38-43 years
    
<img src="Example/Detecting age and gender man1.png">

    >python detect.py --image man2.jpg
    Gender: Male
    Age: 25-32 years
    
<img src="Example/Detecting age and gender man2.png">

    >python detect.py --image woman1.jpg
    Gender: Female
    Age: 38-43 years
    
<img src="Example/Detecting age and gender woman1.png">
              


## Dataset

The project utilizes pre-trained models provided by OpenCV for face detection, age estimation, and gender classification. These models are trained on diverse datasets to achieve robust performance.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/your-username/age-gender-detection.git
    cd age-gender-detection
    ```

2. **Install Dependencies**:

    ```bash
    pip install opencv-python argparse
    ```

3. **Download the Required Models**:

    - `opencv_face_detector.pbtxt` and `opencv_face_detector_uint8.pb` for face detection.
    - `age_deploy.prototxt` and `age_net.caffemodel` for age estimation.
    - `gender_deploy.prototxt` and `gender_net.caffemodel` for gender classification.

   These can be downloaded from [OpenCV's GitHub repository](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) and other sources.

4. **Save the Models in the Project Directory**:
   
    Ensure the models are placed in the same directory as the `main.py` script or update the paths accordingly.

## Usage

To run the age and gender detection system, use the following command:

```bash
python main.py --image /path/to/your/image.jpg
```
If you want to use a live camera feed, just run:
```bash
python main.py
```
Replace /path/to/your/image.jpg with the path to your image file.

## Libraries and Their Usage
- OpenCV (cv2): Used for image processing and handling the camera input. It provides tools for reading images, capturing video from the camera, converting images to different color spaces, and drawing on images.
    ```bash
    import cv2
    ```
- argparse: Used for parsing command-line arguments. This allows the script to accept image file paths or default to the webcam.
    ```bash
    import argparse
    ```
- Math: Provides mathematical functions. Used here for image processing tasks.
    ```bash
    import math
    ```
- DNN Module in OpenCV: Handles loading and running deep learning models for face detection, age estimation, and gender classification.
    ```bash
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections=net.forward()
    ```

## Results
The system detects the face in the image and prints the estimated age and predicted gender. It also overlays this information on the image in real-time.

## Acknowledgments
- The face detection model (haarcascade_frontalface_default.xml) is a part of the OpenCV library.
- The age and gender models are provided by OpenCV's DNN module and are trained on diverse datasets.
