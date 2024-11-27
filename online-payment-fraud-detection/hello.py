#%%
import pandas as pd
import numpy as np
data = pd.read_csv("onlinefraud.csv")
data.head()
# %%
data.isnull().sum()
# %%
# Exploring transaction type
data.type.value_counts()
# %%
type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()
# %%
# %%
print(data.dtypes)

# %%
data_numeric = data.select_dtypes(include=[float, int])

# %%
data_numeric = data_numeric.dropna()  # Drop rows with missing values
# or
data_numeric = data_numeric.fillna(0)  # Replace missing values with 0 or another value

# %%
correlation = data_numeric.corr()
print(correlation["isFraud"].sort_values(ascending=False))

# %%
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
data.head()
# %%
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])
# %%
# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)
# %%
# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))
# %%
features = np.array([[4, 9000.60, 9000.60, 50000.0]])
print(model.predict(features))


# %%
# After training your model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Your existing model training code
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Save the trained model using pickle
import pickle

with open("fraud_detection_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as fraud_detection_model.pkl")

# %%
