import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import os

# File path to the dataset
DATA_PATH = 'salaries.csv'

# Check if the model file exists
MODEL_PATH = 'salary_model.pkl'
model_exists = os.path.isfile(MODEL_PATH)

# Function to train and save the model
def train_model():
    df = pd.read_csv(DATA_PATH)
    X = df[['years_of_experience', 'education_level', 'location']]
    y = df['salary']

    # Define preprocessing steps for numeric and categorical columns
    numeric_features = ['years_of_experience']
    categorical_features = ['education_level', 'location']

    # Use OneHotEncoder with handle_unknown='ignore' to handle unknown categories
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

# Train the model if it doesn't exist
if not model_exists:
    train_model()

# Load the trained model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Define the Streamlit app
st.title('Software Developer Salary Predictor')

# Input features
years_of_experience = st.number_input('Years of Experience', min_value=0, max_value=50, value=0)
education_level_options = ['Bachelors', 'Masters', 'PhD']  # Remove 'High School' option
education_level = st.selectbox('Education Level', education_level_options)
location_options = ['San Francisco', 'New York', 'Austin', 'Remote']  # Define location options
location = st.selectbox('Location', location_options)

# Prediction
if st.button('Predict Salary'):
    input_data = pd.DataFrame({
        'years_of_experience': [years_of_experience],
        'education_level': [education_level],
        'location': [location]
    })

    # Ensure input_data has correct column names and types
    input_data['education_level'] = input_data['education_level'].astype(str)
    input_data['location'] = input_data['location'].astype(str)

    # Load the encoder used during training
    df_train = pd.read_csv(DATA_PATH)
    categorical_features = ['education_level', 'location']
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(df_train[categorical_features])

    # Apply the same preprocessing steps as during training
    input_data_encoded = encoder.transform(input_data)

    # Predict using the loaded model
    predicted_salary = model.predict(input_data_encoded)

    # Display the predicted salary
    st.write(f'Predicted Salary: ${predicted_salary[0]:,.2f}')
