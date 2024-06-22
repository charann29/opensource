# -- coding: utf-8 --

import pandas as pd
import streamlit as st

# Load the CSV file using pandas
heart_disease_data = pd.read_csv(r'C:\Users\91767\Documents\GitHub\MultiDiseasePrediction-using-ml\heart_disease_model.csv')

# Display the title
st.title("Heart Disease Prediction")

# Sidebar for navigation
with st.sidebar:
    st.header("Sidebar Example")
    st.write("You can add input widgets here.")

# Display the dataframe in the main page
st.write("Heart Disease Data")
st.dataframe(heart_disease_data)

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    age = st.text_input('Age')

with col2:
    sex = st.text_input('Sex')

with col3:
    cp = st.text_input('Chest Pain types')

with col1:
    trestbps = st.text_input('Resting Blood Pressure')

with col2:
    chol = st.text_input('Serum Cholestoral in mg/dl')

with col3:
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

with col1:
    restecg = st.text_input('Resting Electrocardiographic results')

with col2:
    thalach = st.text_input('Maximum Heart Rate achieved')

with col3:
    exang = st.text_input('Exercise Induced Angina')

with col1:
    oldpeak = st.text_input('ST depression induced by exercise')

with col2:
    slope = st.text_input('Slope of the peak exercise ST segment')

with col3:
    ca = st.text_input('Major vessels colored by flourosopy')

with col1:
    thal = st.text_input('Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

# Code for Prediction (assuming you will use the loaded data for some prediction logic)
heart_diagnosis = ''

# Creating a button for Prediction
if st.button('Heart Disease Test Result'):
    # Ensure inputs are in the correct format
    try:
        input_data = [
            float(age),
            float(sex),
            float(cp),
            float(trestbps),
            float(chol),
            float(fbs),
            float(restecg),
            float(thalach),
            float(exang),
            float(oldpeak),
            float(slope),
            float(ca),
            float(thal)
        ]

        # Placeholder for prediction logic
        # For demonstration, assuming heart_disease_data has some model-like logic
        # You should replace this with your actual prediction code
        heart_prediction = [1]  # Dummy prediction logic

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

        st.success(heart_diagnosis)
    except ValueError:
        st.error('Please enter valid input values.')