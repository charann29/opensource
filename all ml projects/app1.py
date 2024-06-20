# Importing required Libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Giving Title
st.title("KNN Classification on Inbuilt and Kaggle Datasets")

# Select box for dataset
data_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine", "Diabetes", "Digits", "Salary", "Naive Bayes Classification", "Car Evaluation"))

# Load the dataset
def load_dataset(Data):
    if Data == "Iris":
        return datasets.load_iris()
    elif Data == "Wine":
        return datasets.load_wine()
    elif Data == "Breast Cancer":
        return datasets.load_breast_cancer()
    elif Data == "Diabetes":
        return datasets.load_diabetes()
    elif Data == "Digits":
        return datasets.load_digits()
    elif Data == "Salary":
        return pd.read_csv("Salary_dataset.csv")
    elif Data == "Naive Bayes Classification":
        return pd.read_csv("Naive-Bayes-Classification-Data.csv")
    else:
        return pd.read_csv("car_evaluation.csv")

# Call function to load the dataset
data = load_dataset(data_name)

# Split between input and output
def Input_output(data, data_name):
    if data_name == "Salary":
        X, Y = data['YearsExperience'].to_numpy().reshape(-1, 1), data['Salary'].to_numpy().reshape(-1, 1)
    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data['diabetes']
    elif data_name == "Car Evaluation":
        df = data
        le = LabelEncoder()
        for i in df.columns:
            df[i] = le.fit_transform(df[i])
        X, Y = df.drop(['unacc'], axis=1), df['unacc']
    else:
        X = data.data
        Y = data.target
    return X, Y

# Calling Function to get Input and Output
X, Y = Input_output(data, data_name)

# Add Parameters for KNN
def add_parameter_knn():
    params = dict()
    k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20, key="k_n_slider")
    params['K'] = k_n
    weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
    params['weights'] = weights_custom
    return params

# Calling Function to get Parameters for KNN
params = add_parameter_knn()

# Write dataset information
def info(data_name):
    if data_name not in ["Diabetes", "Salary", "Naive Bayes Classification", "Car Evaluation"]:
        st.write(f"## Classification {data_name} Dataset")
        st.write('Shape of Dataset is: ', X.shape)
        st.write('Number of classes: ', len(np.unique(Y)))
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": data.target_names})
        st.write('Values and Name of Classes')
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
    elif data_name in ["Diabetes", "Salary"]:
        st.write(f"## Regression {data_name} Dataset")
        st.write('Shape of Dataset is: ', X.shape)
    else:
        st.write(f"## Classification {data_name} Dataset")
        st.write('Shape of Dataset is: ', X.shape)
        st.write('Number of classes: ', len(np.unique(Y)))
        if data_name == "Naive Bayes Classification":
            df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": ['Not Diabetic', 'Diabetic']})
        else:
            df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": ['Unacceptable', 'Acceptable', 'Good Condition', 'Very Good Condition']})
        st.write('Values and Name of Classes')
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

# Calling function to print Dataset Information
info(data_name)

# Build and train KNN model
model = KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
model.fit(x_train, y_train)

# Find the predicted values
predict = model.predict(x_test)

# Evaluate the model
st.write("Training Accuracy is:", model.score(x_train, y_train) * 100)
st.write("Testing Accuracy is:", accuracy_score(y_test, predict) * 100)

# Plotting Dataset
pca = PCA(2)
if data_name != "Salary":
    X = pca.fit_transform(X)

fig = plt.figure()
def plot_data(data_name):
    if data_name in ["Diabetes", "Salary"]:
        sns.scatterplot(x=X[:, 0], y=Y, hue=Y, palette='viridis', alpha=0.4)
        plt.plot(x_test, predict, color="red")
        plt.title("Scatter Plot of Dataset")
    else:
        colors = ['purple', 'green', 'yellow', 'red']
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), alpha=0.4)
        plt.legend(shadow=True)
        plt.title("Scatter Plot of Dataset With Target Classes")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

# Calling Function to plot data
plot_data(data_name)
st.pyplot(fig)