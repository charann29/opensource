# Parkinson Disease Detection

## Overview

Parkinson Disease Detection is a web application developed using Streamlit, designed to predict whether a person may have Parkinson's disease based on vocal measurements. The application leverages a pre-trained machine learning model (XGBoost) to provide predictions in real-time or in batch mode using CSV files.

## Features

- **Single Prediction:**
  Users can input individual vocal measurement features to predict the likelihood of Parkinson's disease.

- **Batch Prediction:**
  Supports batch prediction by uploading a CSV file containing multiple sets of vocal measurements. It predicts the likelihood of Parkinson's disease for each record in the CSV.

## Files

- `app.py`: Main script containing the Streamlit application logic.
- `parkinson_model.pkl`: Pre-trained XGBoost model file used for predictions.
- `batch_data.csv`: Sample CSV file for testing batch predictions.

## Usage

1. **Installation:**
   Clone the repository and install necessary dependencies.

2. **Running the Application:**
   Execute the Streamlit application by running `streamlit run app.py` in your terminal. Open the provided URL in your web browser.

3. **Single Prediction:**
   - Enter values for each vocal measurement feature.
   - Click the "Predict" button to see the prediction result.

4. **Batch Prediction:**
   - Click the "Predict Batch" button.
   - Upload a CSV file with multiple sets of vocal measurements.
   - View predictions for each record in the CSV.

## Modifications

- **Model Update:**
  Replace `parkinson_model.pkl` with an updated model file if necessary. Ensure compatibility with the input features defined in `app.py`.

- **Feature Expansion:**
  Modify `app.py` to include additional features for prediction if required by your dataset.

## Notes

- Ensure all necessary Python packages are installed (`streamlit`, `pandas`, `numpy`, `xgboost`).
- The sample batch data (`batch_data.csv`) is provided for demonstration purposes and should be replaced with actual data for meaningful predictions.

## Contact Information

For any questions or issues, please contact [Your Name] at [Your Email].

