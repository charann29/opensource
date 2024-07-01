
# **Object Detection And Tracking Using Deep Learning**

This project implements object detection and tracking using the MobileNet-SSD (Single Shot MultiBox Detector) algorithm. The application is designed to identify and follow objects across video frames, leveraging the efficiency and speed of the MobileNet architecture for real-time performance. The project supports both pre-recorded video files and live webcam feeds. The front-end part of the application is built using Tkinter

## Features

- **Real-time object detection**: Efficient and fast object detection using MobileNet-SSD.
- **Video input options**: Supports both pre-recorded video files and live webcam feeds.
- **User-friendly interface**: Simple GUI built with Tkinter.


## Installation

### Prerequisites

Ensure you have the following installed:
 - Python 3.x
 - OpenCV
 - Tkinter (comes pre-installed with Python standard library)
 - NumPy
 - imutils
    

## Run Locally

1. **Install Visual Studio Code and Python:**
    - [Visual Studio Code](https://code.visualstudio.com/)
    - [Python](https://www.python.org/)

2. **Install the Python extension for VS Code:**
    Open VS Code, go to the Extensions view, and search for "Python" by Microsoft. Install it.

3. **Create a Virtual Environment:**
    Open a terminal in VS Code (Terminal > New Terminal) and run:
    ```sh
    python -m venv venv
    ```

4. **Activate the Virtual Environment:**
    - **Windows:**
      ```sh
      .\venv\Scripts\activate
      ```
    - **macOS/Linux:**
      ```sh
      source venv/bin/activate
      ```

5. **Install Required Packages:**
    Create a `requirements.txt` file in your project directory with the following content:
    ```plaintext
    numpy
    opencv-python
    imutlis
    tkinter
    ```
    Then, install the packages using pip:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

1. **Prepare the Object Detection Code:**
    Create a Python file, e.g., `app.py`, Which is present in the repository.

2. **Run the Application:**
    In the VS Code terminal, ensure your virtual environment is activated and run:
    ```sh
    python app.py
    ```

## Files

- `deploy.prototxt`: The configuration file for the MobileNet-SSD model.
- `mobilenet_iter_73000.caffemodel`: The pre-trained MobileNet-SSD model weights.
- `app.py`: The main Python script for object detection.

## Notes

- Adjust the paths to your `deploy.prototxt` and `mobilenet_iter_73000.caffemodel` files as needed.
- Ensure your webcam is properly connected and accessible by OpenCV.


## Screenshots
![Screenshot 2024-06-20 224839](https://github.com/Bhanuprakash842/cmr_opensource/assets/127642353/cb8e1443-8756-4644-a217-62e308227651)


https://github.com/Bhanuprakash842/cmr_opensource/assets/127642353/2717aa0c-87fc-4d87-898a-204d9340c945



## Acknowledgements

 - [OpenCV](https://opencv.org/)
 - [Tkinter](https://wiki.python.org/moin/TkInter)
 - [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)

