import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('salary_data.csv')  # Ensure you have the dataset in the project directory
    return data

# Build the Streamlit app
st.title("Salary Prediction Web App")

# Load and display the data
data = load_data()
st.write("## Data Overview")
st.write(data.head())

# Split the data into training and testing sets
X = data[['YearsExperience']]
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Display metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
st.write("## Model Performance")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Mean Absolute Error: {mae}")

# Allow user input for predictions
st.write("## Predict Salary")
years_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
predicted_salary = model.predict(np.array([[years_experience]]))
st.write(f"Predicted Salary: {predicted_salary[0]:.2f}")
