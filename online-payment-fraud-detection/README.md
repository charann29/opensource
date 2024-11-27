# Online Payment Fraud Detection System

This project is an online payment fraud detection system built using Python, a machine learning model (Decision Tree Classifier), and a user-friendly interface using Streamlit. The system predicts whether a transaction is fraudulent or not based on transaction details such as the type of transaction, amount, and balances.

## Project Overview

The main goal of this project is to detect potentially fraudulent transactions in real-time. The system allows users to input transaction details, and based on the trained model, it predicts whether the transaction is likely fraudulent or not.

### Features
- **Transaction Type Selection**: Choose the type of transaction (e.g., CASH_OUT, PAYMENT, etc.).
- **Input Fields**: Enter the amount, the original balance before the transaction, and the new balance after the transaction.
- **Prediction**: The system predicts whether a transaction is fraudulent or not using a pre-trained Decision Tree Classifier.
- **User-Friendly Interface**: Built using Streamlit for easy interaction.

## Dataset

The dataset used to train the fraud detection model is `onlinefraud.csv`. The dataset contains multiple features such as transaction type, amount, old balance, new balance, etc., which are used to train the machine learning model.

## Installation

To run this project locally, follow these steps:

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/sharath4444/online-payment-fraud-detection.git
cd online-payment-fraud-detection
```

### Install the Required Packages

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

## File Structure

```
online-payment-fraud-detection/
│
├── app.py                # Streamlit app for fraud detection
├── fraud_detection_model.pkl  # Pre-trained machine learning model (Decision Tree Classifier)
├── onlinefraud.csv        # Dataset used for training the model
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── demo.gif               # Demo of the app (optional)
```

## How to Use the Application

1. Run the app using the command: `streamlit run app.py`
2. Enter the transaction details:
   - Select the transaction type (CASH_OUT, PAYMENT, etc.).
   - Enter the transaction amount.
   - Input the original balance before the transaction and the new balance after.
3. Click the "Predict" button to get the result:
   - **Fraud** or **No Fraud** based on the model's prediction.

## Model Training

The machine learning model used is a **Decision Tree Classifier**. It was trained on the dataset provided, using features such as:
- Transaction type
- Transaction amount
- Old balance before the transaction
- New balance after the transaction

The trained model is saved as `fraud_detection_model.pkl` and loaded in the Streamlit app for real-time predictions.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Any contributions that improve this project are highly appreciated!

---

### Author

Developed by [sharath](https://github.com/sharath4444).

## Demo

![Demo GIF](demo.gif)  <!-- If you have a demo gif or screenshot -->
![demo png](https://github.com/user-attachments/assets/4d298c01-f978-484c-a427-6a61191d9761)
