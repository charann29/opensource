import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load your dataset
# data = pd.read_csv('path_to_your_rental_data.csv')

# For demonstration, let's create a DataFrame manually
data = pd.DataFrame({
    'Size': [500, 600, 700, 800, 900, 1000], # in square feet
    'Bedrooms': [1, 1, 2, 2, 3, 3],
    'Location': [5, 5, 5, 4, 4, 3], # some numeric representation of location
    'RentalPrice': [1000, 1200, 1400, 1600, 1800, 2000] # in dollars
})

# Prepare the data for training
X = data.iloc[:, :-1].values # all columns except the last one
y = data.iloc[:, -1].values # the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict rental prices
y_pred = model.predict(X_test)

# Visualize the results (optional)
# plt.scatter(X_train[:,0], y_train) # Plotting only with respect to 'Size'
# plt.plot(X_train[:,0], model.predict(X_train), color='blue')
# plt.title('Rental Price vs Size (Training set)')
# plt.xlabel('Size (sq ft)')
# plt.ylabel('Rental Price ($)')
# plt.show()

# Use the model to predict rental prices based on new data
new_property_details = [[750, 2, 4]] # Example: 750 sq ft, 2 bedrooms, location score of 4
predicted_rental_price = model.predict(new_property_details)
print(f"The predicted rental price is: ${predicted_rental_price[0]:.2f}")
