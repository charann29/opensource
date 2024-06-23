import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

# Define the file path
file_path = r"C:\Users\USER\Documents\GitHub\cmr_opensource__\house_rent_prediction\House_Rent_Dataset.csv"

# Print the file path
print(f"File path: {file_path}")

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"File not found: {file_path}")
    exit()

# Load the dataset
try:
    df = pd.read_csv(file_path)
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Drop unnecessary columns
columns_to_drop = ["Posted On", "Floor", "Area Locality", "Point of Contact", "Area Type", "Tenant Preferred"]
for col in columns_to_drop:
    if col in df.columns:
        df.pop(col)
    else:
        print(f"Column not found in dataframe: {col}")

# Replace categorical data with numerical values
city_mapping = {"Mumbai": 5, "Bangalore": 4, "Hyderabad": 3, "Delhi": 2, "Chennai": 1, "Kolkata": 0}
furnishing_mapping = {"Furnished": 2, "Semi-Furnished": 1, "Unfurnished": 0}

df['City'] = df['City'].replace(city_mapping)
df['Furnishing Status'] = df['Furnishing Status'].replace(furnishing_mapping)

# Extract target variable
if "Rent" in df.columns:
    target = df.pop("Rent")
else:
    print("Target column 'Rent' not found in dataframe.")
    exit()

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2)

# Train the model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
print("Model trained successfully.")

# Serialize the model to a file
model_file_path = 'model1.pkl'
with open(model_file_path, 'wb') as model_file:
    pickle.dump(lin_reg, model_file)
    print(f"Model saved to {model_file_path}")
