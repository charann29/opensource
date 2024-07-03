import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '2018 Cases against Police Personnels.csv'
data = pd.read_csv(file_path)

# Basic data cleaning and preprocessing
data = data.fillna(0)

# Convert categorical columns to numerical values if needed
data['State/UT'] = data['State/UT'].astype('category').cat.codes

# Define features and target variable
X = data.drop(columns=['S. No.', 'Category', 'State/UT'])
y = data['Number of Cases - Registered']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title('Crime Prediction and Analysis')

# Dataset Overview Section
st.header('Dataset Overview')
st.write('Explore the dataset used for analysis.')
st.write(data.head())

# Summary Statistics Section
st.header('Summary Statistics')
st.write('Get an overview of statistical measures of the dataset.')
st.write(data.describe())

# Correlation Heatmap Section
st.header('Correlation Heatmap')
st.write('Visualize the correlations between numerical features.')
numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
st.pyplot(fig)

# Model Performance Section
st.header('Model Performance')
st.write('Assess how well the Linear Regression model performs.')
st.write('- Mean Squared Error:', mse)
st.write('- R^2 Score:', r2)

# Predictions Section
st.header('Predictions')
st.write('Explore the predictions made by the model.')
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write(predictions_df)

# Visualization of Predictions Section
st.header('Visualization of Predictions')
st.write('Visualize how well the model predictions match the actual values.')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, color='blue')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted')
st.pyplot(fig)

# Footer
st.write("""
This Streamlit application provides an analysis of crime data and predictions using a Linear Regression model.
""")
