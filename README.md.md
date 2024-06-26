# Automated Segmentation of Brain Tumor from Multimodal MR Images

## ABSTRACT
Automated segmentation of brain tumors from multimodal MR images is pivotal for the analysis and monitoring of disease progression. As gliomas are malignant and heterogeneous, efficient and accurate segmentation techniques are used for the successful delineation of tumors into intra-tumoral classes. Deep learning algorithms outperform on tasks of semantic segmentation as opposed to more conventional, context-based computer vision approaches. Extensively used for biomedical image segmentation, Convolutional Neural Networks (CNNs) have significantly improved the state-of-the-art accuracy on the task of brain tumor segmentation. In this paper, we propose an ensemble of two segmentation networks: a 3D CNN and a U-Net, in a significant yet straightforward combinative technique that results in better and accurate predictions. Both models were trained separately on the BraTS-19 challenge dataset and evaluated to yield segmentation maps which considerably differed from each other in terms of segmented tumor sub-regions and were ensembled variably to achieve the final prediction. The suggested ensemble achieved dice scores of 0.750, 0.906, and 0.846 for enhancing tumor, whole tumor, and tumor core, respectively, on the validation set, performing favorably in comparison to the state-of-the-art architectures currently available.

## INDEX TERMS
Deep learning, BraTS, medical imaging, segmentation, U-Net, CNN, ensembling.

## I. INTRODUCTION
Accurate segmentation of tumors through medical images is of particular importance as it provides essential information for the analysis and diagnosis of cancer as well as for mapping out treatment options and monitoring the progression of the disease. Brain tumors are one of the fatal cancers worldwide and are categorized, depending upon their origin, into primary and secondary tumor types. The most common histological form of primary brain cancer is the glioma, which originates from the brain glial cells and attributes towards 80% of all malignant brain tumors. Gliomas can be of the slow-progressing low-grade (LGG) subtype with a better patient prognosis or are the more aggressive and infiltrative high-grade glioma (HGG) or glioblastoma, which require immediate treatment. These tumors are associated with substantial morbidity, where the median survival for a patient with glioblastoma is only about 14 months with a 5-year survival rate near zero despite maximal surgical and medical therapy. A timely diagnosis, therefore, becomes imperative for effective treatment of the patients.

Magnetic Resonance Imaging (MRI) is a preferred technique widely employed by radiologists for the evaluation and assessment of brain tumors. It provides several complementary 3D MRI modalities acquired based on the degree of excitation and repetition times, i.e. T1-weighted, post-contrast T1-weighted (T1ce), T2-weighted and Fluid-Attenuated Inversion Recovery (FLAIR). The highlighted subregions of the tumor across different intensities of these sequences, such as the whole tumor (the entire tumor inclusive of infiltrative edema), is more prominent in FLAIR and T2 modalities. In contrast, T1 and T1ce images show the tumor core exclusive of peritumoral edema. This allows for the combinative use of these scans and the complementary information they deliver towards the detection of different tumor subregions.

The Multimodal Brain Tumor Segmentation Challenge (BraTS) is a platform to evaluate the development of machine learning models for the task of tumor segmentation, by facilitating the participants with an extensive dataset.


## LIBRARIES USED
TensorFlow: For building and training the deep learning models.
Keras: High-level neural networks API, running on top of TensorFlow.
OpenCV: For image processing tasks.
NumPy: For numerical computations.
Pandas: For data manipulation and analysis.
Matplotlib: For plotting and visualization.
Imutils: For image processing convenience functions.


## DATASET USED

The dataset used in this project is the BraTS-19 dataset, which includes multimodal MRI scans of brain tumors. The dataset consists of four types of MRI images for each patient: T1-weighted, post-contrast T1-weighted (T1ce), T2-weighted, and FLAIR images. The images are preprocessed using the crop_brain_contour function to focus on the brain region.

## MODEL ARCHITECTURE

The project employs an ensemble of two segmentation networks: a 3D CNN and a U-Net. Both models are trained separately and their outputs are combined to improve the final segmentation accuracy.

## 3D CNN
A convolutional neural network that processes 3D MRI volumes.
Includes layers such as Conv3D, MaxPooling3D, and Dense.
## U-Net
A type of convolutional network designed for biomedical image segmentation.
Consists of an encoder (downsampling path) and a decoder (upsampling path) with skip connections between corresponding layers.

## HOW TO RUN
STEP1:Mount Google Drive: Ensure you have your data stored in your Google Drive.
STEP2:Install Required Libraries:
STEP3:import necessary libarries
STEP4:Define Utility Functions
step5:use streamlit for frontend implementation


## USE CASES OF THIS PROJECT

Medical Diagnosis and Planning
Monitoring Disease Progression
Research and Development
Educational Tool
Automated Workflows in Healthcare
Integration with PACS Systems
Telemedicine and Remote Diagnostics






## Outputs: 
<br><br>
![alt text](<Screenshot 2024-06-24 184159.png>)
![alt text](<Screenshot 2024-06-24 184219.png>)
