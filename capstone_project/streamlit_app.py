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

st.title("KNN Classifier on Kaggle Sets")

data_name = st.sidebar.selectbox("Select Dataset", 
                                 ("Iris", "Breast Cancer", "Wine", "Diabetes", "Digits", "Salary", "Naive Bayes Classification", "Car Evaluation"))

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

data = load_dataset(data_name)

def Input_output(data, data_name):
    if data_name == "Salary":
        X, Y = data['YearsExperience'].to_numpy().reshape(-1, 1), data['Salary'].to_numpy().reshape(-1, 1)
    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data['diabetes']
    elif data_name == "Car Evaluation":
        df = data
        le = LabelEncoder()
        func = lambda i: le.fit(df[i]).transform(df[i])
        for i in df.columns:
            df[i] = func(i)
        X, Y = df.drop(['unacc'], axis=1), df['unacc']
    else:
        X = data.data
        Y = data.target
    return X, Y

X, Y = Input_output(data, data_name)

# Parameters for KNeighborsClassifier
k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20, key="k_n_slider")
weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))

# Create and train KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=k_n, weights=weights_custom)

def info(data_name):
    if data_name not in ["Diabetes", "Salary", "Naive Bayes Classification", "Car Evaluation"]:
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is: KNN CLASSIFIER')
        st.write('Shape of Dataset is:', X.shape)
        st.write('Number of classes:', len(np.unique(Y)))
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": data.target_names})
        st.write('Values and Name of Classes')
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")
    elif data_name == "Diabetes":
        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is: KNeighborsClassifier')
        st.write('Shape of Dataset is:', X.shape)
    elif data_name == 'Salary':
        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is: KNeighborsClassifier')
        st.write('Shape of Dataset is:', X.shape)
    elif data_name == "Naive Bayes Classification":
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is: KNeighborsClassifier')
        st.write('Shape of Dataset is:', X.shape)
        st.write('Number of classes:', len(np.unique(Y)))
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": ['Not Diabetic', 'Diabetic']})
        st.write('Values and Name of Classes')
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")
    else:
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is: KNeighborsClassifier')
        st.write('Shape of Dataset is:', X.shape)
        st.write('Number of classes:', len(np.unique(Y)))
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": ['Unacceptable', 'Acceptable', 'Good Condition', 'Very Good Condition']})
        st.write('Values and Name of Classes')
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")

info(data_name)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
model.fit(x_train, y_train)
predict = model.predict(x_test)

st.write("Training Accuracy is:", model.score(x_train, y_train) * 100)
st.write("Testing Accuracy is:", accuracy_score(y_test, predict) * 100)

pca = PCA(2)

if data_name != "Salary":
    X = pca.fit_transform(X)

fig, ax = plt.subplots()

def choice_classifier(data_name):
    scatter = None  # Initialize scatter variable
    
    if data_name == "Diabetes":
        scatter = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', alpha=0.8)
        ax.set_title("Scatter Classification Plot of Dataset")
    elif data_name == "Digits":
        colors = ['purple', 'green', 'yellow', 'red', 'black', 'cyan', 'pink', 'magenta', 'grey', 'teal']
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), ax=ax, alpha=0.4)
        ax.legend(data.target_names, shadow=True)
        ax.set_title("Scatter Classification Plot of Dataset With Target Classes")
    elif data_name == "Salary":
        sns.scatterplot(x=data['YearsExperience'], y=data['Salary'], data=data, ax=ax)
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel("Salary")
        ax.set_title("Scatter Classification Plot of Dataset")
    elif data_name == "Naive Bayes Classification":
        colors = ['purple', 'green']
        sns.scatterplot(x=data['glucose'], y=data['bloodpressure'], hue=Y, palette=sns.color_palette(colors), ax=ax, alpha=0.4)
        ax.legend(shadow=True)
        ax.set_xlabel('Glucose')
        ax.set_ylabel("Blood Pressure")
        ax.set_title("Scatter Classification Plot of Dataset With Target Classes")
    else:
        colors = ['purple', 'green', 'yellow', 'red']
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), ax=ax, alpha=0.4)
        ax.legend(shadow=True)
        ax.set_title("Scatter Classification Plot of Dataset With Target Classes")
    
    return scatter 

if data_name == 'Salary':
    sns.scatterplot(x=data['YearsExperience'], y=data['Salary'], data=data, ax=ax)
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel("Salary")
    ax.set_title("Scatter Plot of Dataset")
else:
    scatter = choice_classifier(data_name)

if data_name != "Salary" and data_name != "Naive Bayes Classification":
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

if data_name == "Naive Bayes Classification":
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

st.pyplot(fig)