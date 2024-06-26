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

st.title("KNN Classifier on Inbuilt and Kaggle Datasets")

data_name=st.sidebar.selectbox("Select Dataset", ("Iris","Breast Cancer","Wine","Diabetes","Digits","Salary","Naive Bayes Classification","Car Evaluation"))

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
    elif Data == "Naive Bayes Classification" :
        return pd.read_csv("Naive-Bayes-Classification-Data.csv")
    else :
        return pd.read_csv("car_evaluation.csv")

data = load_dataset(data_name)

def Input_output(data,data_name):
    if data_name == "Salary":
        X, Y = data['YearsExperience'].to_numpy().reshape(-1, 1), data['Salary'].to_numpy().reshape(-1, 1)
    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data['diabetes']
    elif data_name == "Car Evaluation":
        df= data
        le = LabelEncoder()
        func = lambda i: le.fit(df[i]).transform(df[i])
        for i in df.columns:
            df[i] = func(i)
        X, Y = df.drop(['unacc'], axis=1), df['unacc']
    else :
        X = data.data
        Y = data.target
    return X,Y

X, Y = Input_output(data,data_name)

k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20,key="k_n_slider")
weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
algo_model = KNeighborsClassifier(n_neighbors=k_n, weights=weights_custom)
algo_model.fit(x_train,y_train)
predict=algo_model.predict(x_test)
st.write("Training Accuracy is:",algo_model.score(x_train,y_train)*100)
st.write("Testing Accuracy is:",accuracy_score(y_test,predict)*100)

pca=PCA(2)
X = pca.fit_transform(X)

fig = plt.figure()
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette="viridis", alpha=0.4)
plt.legend(shadow=True)
plt.title("Scatter Classification Plot of Dataset With Target Classes")
st.pyplot(fig)