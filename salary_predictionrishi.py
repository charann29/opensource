import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate a synthetic dataset for illustration
def generate_data():
    np.random.seed(42)
    experience = np.random.randint(1, 21, 100)
    education_level = np.random.randint(1, 4, 100)  # 1: Bachelor's, 2: Master's, 3: PhD
    salary = (experience * 2000) + (education_level * 5000) + np.random.normal(0, 5000, 100)
    return pd.DataFrame({'Experience': experience, 'Education Level': education_level, 'Salary': salary})

# Load data
data = generate_data()

# Split data
X = data[['Experience', 'Education Level']]
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit app
st.title('Salary Prediction')

st.write('This is a simple example of a salary prediction model using a linear regression algorithm.')

# Input features
st.sidebar.header('Input Features')
experience = st.sidebar.slider('Years of Experience', min_value=1, max_value=20, value=5, step=1)
education_level = st.sidebar.selectbox('Education Level', options=[1, 2, 3], format_func=lambda x: {1: "Bachelor's", 2: "Master's", 3: "PhD"}[x])

# Predict salary
input_features = pd.DataFrame({'Experience': [experience], 'Education Level': [education_level]})
predicted_salary = model.predict(input_features)[0]

# Display results
st.write(f'### Predicted Salary: ${predicted_salary:,.2f}')
st.write('#### Model Performance')
st.write(f'Mean Squared Error: ${mse:,.2f}')
