# Facial Expressions Recognition

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a Facial Expressions Recognition system using a Convolutional Neural Network (CNN). The system can identify seven different expressions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Model](#running-the-model)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

Facial expression recognition is a significant area in computer vision with applications in various fields such as human-computer interaction, behavior analysis, and emotion AI. This project uses a deep learning approach to detect faces and classify their expressions in real-time.

## Dataset

The dataset used for this project consists of images of faces with different expressions. Each image is converted to grayscale and resized to 48x48 pixels. The dataset is organized into training and validation sets.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/shayan-tej/facial-expressions-recognition.git
    cd facial-expression-recognition
    ```

2. **Install Dependencies**

    ```bash
    pip install numpy pandas matplotlib seaborn opencv-python keras
    ```

## Training the Model

1. **Open the Notebook**:

    ```bash
    jupyter notebook emotion-classification-cnn-model.ipynb
    ```

2. **Follow the Steps**:
- Load and preprocess the dataset.
- Define and compile the CNN model.
- Train the model using the training set.
- Save the trained model as model.h5.

## Running the Model

You can run the real-time facial expression recognition using the `main.py` script:

   ```bash
   python main.py
   ```

## Results

After training the model, you can visualize the training performance:

- Loss and Accuracy Curves:

    The notebook plots the training and validation loss and accuracy over epochs, helping to understand how well the model is learning.

## Acknowledgments

- The `haarcascade_frontalface_default.xml` file is a part of the OpenCV library used for face detection.
- The CNN model is built and trained using Keras with TensorFlow backend.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

### Instructions:

1. Replace `"https://github.com/your-username/facial-expressions-recognition.git"` with the actual URL of your GitHub repository.
2. Ensure `model.h5` and `haarcascade_frontalface_default.xml` are included in your repository, or provide instructions on how to obtain them.
3. If your project uses a different license, update the `License` section accordingly.

This `README.md` provides clear and concise instructions and descriptions, making it easier for users to understand, set up, and use the project.
