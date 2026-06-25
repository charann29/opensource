Helmet Detection and Automatic License Plate Recognition (ALPR)
===============================================================

This project demonstrates the use of machine learning for helmet detection and automatic license plate recognition (ALPR). It utilizes computer vision techniques and machine learning models to achieve these functionalities.

Features
--------

-   *Helmet Detection:*
    -   Uses a trained machine learning model to detect whether a person is wearing a helmet or not.
    -   Provides bounding boxes around detected helmets in images or videos.
-   *Automatic License Plate Recognition (ALPR):*
    -   Recognizes license plates from images or videos.
    -   Utilizes optical character recognition (OCR) techniques to extract alphanumeric characters from license plates.

Installation
------------

To install and run this project locally, follow these steps:

1.  *Clone the repository:*

    bash

    Copy code

    `git clone https://github.com/your-username/helmet-alpr.git
    cd helmet-alpr`

2.  *Set up environment:*

    -   Create a virtual environment (optional but recommended).

    -   Install dependencies:

        bash

        Copy code

        pip install -r requirements.txt

3.  *Download pretrained models (if necessary):*

    -   click the link at the end to download

Usage
-----

### Helmet Detection

To use helmet detection, run:

bash

Copy code

python helmet_detection.py --input path/to/image_or_video --output path/to/save/result

Replace path/to/image_or_video with the path to your input image or video file. The output will be saved at path/to/save/result.

### Automatic License Plate Recognition (ALPR)

To perform ALPR, run:

bash

Copy code

python alpr.py --input path/to/image_or_video --output path/to/save/result

Replace path/to/image_or_video with the path to your input image or video file. The recognized license plates and their bounding boxes will be saved at path/to/save/result.

Contributing
------------

Contributions are welcome! Please fork the repository and submit pull requests.

This project combines two advanced computer vision applications: Helmet Detection using Machine Learning and Automatic License Plate Recognition (ALPR). The Helmet Detection component utilizes deep learning models to detect the presence of helmets in images or video streams, providing real-time detection and bounding box visualization. This is particularly useful for safety monitoring in construction sites, industrial environments, and sports events. On the other hand, the ALPR module employs image processing techniques and optical character recognition (OCR) to accurately extract license plate numbers from vehicles in images or videos.
-----------------
yolo weights :-
https://drive.google.com/drive/folders/1uoNASEOVJ2rP9HxuyIiiNZt682rDV7Pg?usp=drive_link

**Here is the output with some images**

![WhatsApp Image 2024-06-29 at 23 00 20_08cd3b0e](https://github.com/charann29/cmr_opensource/assets/169017734/f72ddcef-84bb-433f-bdf7-221f9db99d04)

![WhatsApp Image 2024-06-29 at 23 00 18_3c0c5352](https://github.com/charann29/cmr_opensource/assets/169017734/dcc233bc-de2b-42c8-997b-ef16ff50a4e3)

![WhatsApp Image 2024-06-29 at 23 00 19_50a6aea7](https://github.com/charann29/cmr_opensource/assets/169017734/e78c7bff-095e-496a-af9e-05bef66e7050)
