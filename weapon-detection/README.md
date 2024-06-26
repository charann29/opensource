Weapon Detection Using YOLOv3
=============================

Project Overview
----------------

This project utilizes the YOLOv3 (You Only Look Once) algorithm for real-time weapon detection. The code supports two modes of operation:

1.  **Webcam-based weapon detection**: Uses the webcam to detect weapons in real-time.
2.  **Video file-based weapon detection**: Uses a pre-recorded video file to detect weapons.

Features
--------

-   Real-time detection of weapons using either a webcam or a video file.
-   Display of bounding boxes and labels for detected objects.
-   Customizable detection thresholds.

Technologies and Libraries Used
-------------------------------

-   **Python (version 3.x)**
-   **OpenCV**: Library for computer vision tasks.
-   **NumPy**: Library for numerical computations.
-   **YOLOv3**: Pre-trained deep learning model for object detection.

Getting Started
---------------

### Prerequisites

Make sure you have the following installed:

-   Python 3.x
-   OpenCV
-   NumPy

You can install the required libraries using pip:

`pip install opencv-python numpy`



### Setup

1.  **Clone the repository and navigate to the project directory**:

    ```bash
    git clone <repository_url>
    cd <repository_name>

2.  **Download YOLOv3 weights and configuration files**:

    Download the YOLOv3 weights and configuration files from the official YOLO website or any other reliable source. Place these files in the project directory.

### Running the Code

Depending on your use case, run the appropriate script:

1.  **Webcam-based weapon detection**:

    ```bash
    python weapon_detection1.py

2.  **Video file-based weapon detection**:

    ```bash
    python weapon_detection.py

### Sample Output

The following image demonstrates the output of the YOLOv3 model, showing bounding boxes and labels for detected objects:

![Sample Output](https://github.com/GaviniShasank/cmr_opensource/assets/130299325/7d7e3f96-2c43-4b0e-a6f4-9a0ec38037ae)

By following the steps outlined above, you can execute the code and perform real-time weapon detection using either a webcam or a video file.

Customization
-------------

You can customize the detection thresholds and other parameters within the script to suit your specific requirements. Detailed comments within the code will guide you through making these adjustments.

Contact
-------

For further assistance or inquiries, please reach out via the repository's contact information.
