#COMPLEX UI

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the trained model
model_path = 'parkinson_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Title of the web app
st.title("Parkinson Disease Detection")

# Instructions
st.write("Enter the details to predict if a person has Parkinson's disease.")

# Get user input
def user_input_features():
    # Create input fields for each feature your model requires
    mdvp_fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0)
    mdvp_fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0)
    mdvp_flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0)
    mdvp_jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0)
    mdvp_jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0)
    mdvp_rap = st.number_input('MDVP:RAP', min_value=0.0)
    mdvp_ppq = st.number_input('MDVP:PPQ', min_value=0.0)
    jitter_ddp = st.number_input('Jitter:DDP', min_value=0.0)
    mdvp_shimmer = st.number_input('MDVP:Shimmer', min_value=0.0)
    mdvp_shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0)
    shimmer_apq3 = st.number_input('Shimmer:APQ3', min_value=0.0)
    shimmer_apq5 = st.number_input('Shimmer:APQ5', min_value=0.0)
    mdvp_apq = st.number_input('MDVP:APQ', min_value=0.0)
    shimmer_dda = st.number_input('Shimmer:DDA', min_value=0.0)
    nhr = st.number_input('NHR', min_value=0.0)
    hnr = st.number_input('HNR', min_value=0.0)
    rpde = st.number_input('RPDE', min_value=0.0)
    dfa = st.number_input('DFA', min_value=0.0)
    spread1 = st.number_input('spread1', min_value=0.0)
    spread2 = st.number_input('spread2', min_value=0.0)
    d2 = st.number_input('D2', min_value=0.0)
    ppe = st.number_input('PPE', min_value=0.0)

    # Create a dictionary of inputs
    data = {
        'MDVP:Fo(Hz)': mdvp_fo,
        'MDVP:Fhi(Hz)': mdvp_fhi,
        'MDVP:Flo(Hz)': mdvp_flo,
        'MDVP:Jitter(%)': mdvp_jitter_percent,
        'MDVP:Jitter(Abs)': mdvp_jitter_abs,
        'MDVP:RAP': mdvp_rap,
        'MDVP:PPQ': mdvp_ppq,
        'Jitter:DDP': jitter_ddp,
        'MDVP:Shimmer': mdvp_shimmer,
        'MDVP:Shimmer(dB)': mdvp_shimmer_db,
        'Shimmer:APQ3': shimmer_apq3,
        'Shimmer:APQ5': shimmer_apq5,
        'MDVP:APQ': mdvp_apq,
        'Shimmer:DDA': shimmer_dda,
        'NHR': nhr,
        'HNR': hnr,
        'RPDE': rpde,
        'DFA': dfa,
        'spread1': spread1,
        'spread2': spread2,
        'D2': d2,
        'PPE': ppe
    }
    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input data
input_df = user_input_features()

# Display the input data
st.subheader('User Input:')
st.write(input_df)

# Predict button for single prediction
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display result
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.write("The model predicts that the person has Parkinson's disease.")
    else:
        st.write("The model predicts that the person does not have Parkinson's disease.")

# Function to create a sample batch CSV file
def generate_batch_csv():
    columns_needed = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                      'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                      'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                      'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                      'spread1', 'spread2', 'D2', 'PPE']

    # Create a sample DataFrame with dummy values (modify as per your requirement)
    batch_data = pd.DataFrame(np.random.randn(5, len(columns_needed)), columns=columns_needed)
    
    # Save to CSV file
    batch_data.to_csv('batch_data.csv', index=False)
    return batch_data

# Generate batch CSV data
batch_data = generate_batch_csv()

# Display sample batch data
st.subheader('Sample Batch Data:')
st.write(batch_data)

# Predict button for batch prediction
if st.button('Predict Batch'):
    # Load the batch data
    batch_data = pd.read_csv('batch_data.csv')
    
    # Make predictions
    predictions = model.predict(batch_data)
    
    # Display batch predictions
    st.subheader('Batch Predictions:')
    st.write(predictions)
