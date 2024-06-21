# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Reading the dataset
dataframe = pd.read_csv('dataset.csv')
print(dataframe.head())

# Plotting a box plot for the 'Insulin' column
sns.boxplot(x=dataframe["Insulin"])
plt.show()

# Checking for null values
print(dataframe.isnull().sum())

# Displaying the correlation matrix
print(dataframe.corr())

# Plotting the heatmap for correlation matrix
f, ax = plt.subplots(figsize=[20,15])
sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Plotting pie chart and count plot for 'Outcome' column
f, ax = plt.subplots(1, 2, figsize=(18,8))
dataframe['Outcome'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Target')
ax[0].set_ylabel('')
sns.countplot('Outcome', data=dataframe, ax=ax[1])
ax[1].set_title('Outcome')
plt.show()

# Creating feature set (X) and labels (y)
y = dataframe["Outcome"]
X = dataframe.drop(["Outcome"], axis=1)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Classifier Accuracy: {accuracy_score(y_test, y_pred_rf)}")

# Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print(f"Gradient Boosting Classifier Accuracy: {accuracy_score(y_test, y_pred_gb)}")

# Support Vector Classifier
svc = SVC(gamma='auto')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print(f"Support Vector Classifier Accuracy: {accuracy_score(y_test, y_pred_svc)}")
