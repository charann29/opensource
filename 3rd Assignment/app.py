import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

st.set_page_config(page_title="KNN Model", page_icon=":broken_heart:", layout="centered")

st.title("KNN on Inbuilt and Custom Datasets")

data_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine", "Diabetes", "Digits", "Salary", "Naive Bayes Classification", "Car Evaluation"))

def load_dataset(data_name):
    if data_name == "Iris":
        return datasets.load_iris()
    elif data_name == "Wine":
        return datasets.load_wine()
    elif data_name == "Breast Cancer":
        return datasets.load_breast_cancer()
    elif data_name == "Diabetes":
        return datasets.load_diabetes()
    elif data_name == "Digits":
        return datasets.load_digits()
    elif data_name == "Salary":
        return pd.read_csv("Salary_dataset.csv")
    elif data_name == "Naive Bayes Classification":
        return pd.read_csv("Naive-Bayes-Classification-Data.csv")
    else:
        return pd.read_csv("car_evaluation.csv")

data = load_dataset(data_name)

def input_output(data, data_name):
    if data_name == "Salary":
        X, Y = data['YearsExperience'].to_numpy().reshape(-1, 1), data['Salary']
    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data['diabetes']
    elif data_name == "Car Evaluation":
        df = data
        le = LabelEncoder()
        for col in df.columns:
            df[col] = le.fit_transform(df[col])
        X, Y = df.drop('unacc', axis=1), df['unacc']
    else:
        X, Y = data.data, data.target
    return X, Y

X, Y = input_output(data, data_name)

def add_parameter_knn():
    params = dict()
    k_n = st.sidebar.text_input('Number of Neighbors (K)', value='5')
    try:
        params['K'] = int(k_n)
    except ValueError:
        st.sidebar.error("Please enter a valid integer for the number of neighbors (K)")
        params['K'] = 5  # Default value
    weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
    params['weights'] = weights_custom
    return params

params = add_parameter_knn()

def model_knn(params):
    model = KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])
    return model

model = model_knn(params)

def info(data_name):
    st.write(f"## Classifier on {data_name} Dataset")
    st.write('Shape of Dataset:', X.shape)
    st.write('Number of classes:', len(np.unique(Y)))
    if data_name not in ["Salary", "Naive Bayes Classification", "Car Evaluation"]:
        df = pd.DataFrame({"Target Value": np.unique(Y), "Target Name": data.target_names})
        st.write('Values and Names of Classes:')
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

info(data_name)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
st.write(f'Accuracy: {accuracy:.2f}')
