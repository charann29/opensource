import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Giving Title
st.title("KNN Algorithm on Inbuilt and Custom Datasets")

# Now we are making a select box for dataset
data_name = st.sidebar.selectbox("Select Dataset", 
                                  ("Iris", "Breast Cancer", "Wine", "Diabetes", "Digits", "Salary", "Naive Bayes Classification", "Car Evaluation"))

# The Next is selecting algorithm type
algorithm_type = st.sidebar.selectbox("Select Algorithm Type", ("Classifier", "Regressor"))

# Now we need to load the dataset
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

# Now we need to call function to load the dataset
data = load_dataset(data_name)

# Now after this we need to split between input and output
def Input_output(data, data_name):
    if data_name == "Salary":
        X, Y = data['YearsExperience'].to_numpy().reshape(-1, 1), data['Salary'].to_numpy().reshape(-1, 1)
    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data['diabetes']
    elif data_name == "Car Evaluation":
        df = data
        le = LabelEncoder()
        for i in df.columns:
            df[i] = le.fit(df[i]).transform(df[i])
        X, Y = df.drop(['unacc'], axis=1), df['unacc']
    else:
        X = data.data
        Y = data.target
    return X, Y

# Calling Function to get Input and Output
X, Y = Input_output(data, data_name)

# Adding Parameters so that we can select from various parameters for KNN
def add_parameter_knn():
    params = dict()
    k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20)
    params['K'] = k_n
    weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
    params['weights'] = weights_custom
    return params

# Getting KNN parameters
params = add_parameter_knn()

# Now we will build ML Model for this dataset and calculate accuracy for that
def model_knn(algorithm_type, params):
    if algorithm_type == "Regressor":
        return KNeighborsRegressor(n_neighbors=params['K'], weights=params['weights'])
    else:
        return KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])

# Calling Function to create the model
algo_model = model_knn(algorithm_type, params)

# Now splitting into Testing and Training data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

# Training algorithm
algo_model.fit(x_train, y_train)

# Now we will find the predicted values
predict = algo_model.predict(x_test)

# Finding Accuracy or Error
if algorithm_type == "Regressor":
    st.write("Mean Squared error is:", mean_squared_error(y_test, predict))
    st.write("Mean Absolute error is:", mean_absolute_error(y_test, predict))
else:
    st.write("Training Accuracy is:", algo_model.score(x_train, y_train) * 100)
    st.write("Testing Accuracy is:", accuracy_score(y_test, predict) * 100)

# Plotting Dataset
fig = plt.figure()

def choice_plot(data_name, algorithm_type, X, Y, x_test=None, predict=None):
    if algorithm_type == "Regressor":
        plt.scatter(X[:, 0], Y, c=Y, cmap='viridis', alpha=0.4)
        if x_test is not None and predict is not None:
            plt.plot(x_test, predict, color="red")
            plt.legend(['Actual Values', 'Best Line or General formula'])
        plt.colorbar()
        plt.title("Scatter Plot of Dataset")
    else:
        if X.shape[1] < 2:
            st.write("Error: The dataset does not have enough dimensions for a scatter plot.")
            return
        colors = sns.color_palette("husl", len(np.unique(Y)))
        sns.scatterplot(x=X[:, 0], y=X[:,1] , hue=Y, palette=colors, alpha=0.4)
        plt.legend(shadow=True)
        plt.title("Scatter Plot of Dataset")

# Apply PCA only if there are more than 2 dimensions
if data_name != "Salary" and X.shape[1] > 1:
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
else:
    X_transformed = X

choice_plot(data_name, algorithm_type, X_transformed, Y, x_test, predict)

if algorithm_type == 'Regressor':
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

st.pyplot(fig)