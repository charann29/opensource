# Importing required Libraries
# Importing Numpy
import numpy as np
# To read csv file
import pandas as pd
# Importing datasets from sklearn
from sklearn import datasets
# For splitting between training and testing
from sklearn.model_selection import train_test_split
# Importing KNN algorithm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# Importing accuracy score and mean_squared_error
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
# Importing PCA for dimension reduction
from sklearn.decomposition import PCA
# For Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# For model deployment
import streamlit as st
# Importing Label Encoder
# For converting string to int
from sklearn.preprocessing import LabelEncoder

# Giving Title
st.title("KNN on Inbuilt and Custom Datasets")

# Now we are making a select box for dataset
data_name = st.sidebar.selectbox("Select Dataset",("Iris", "Breast Cancer", "Wine", "Diabetes", "Digits", "Salary", "Naive Bayes Classification", "Car Evaluation"))

# The Next is selecting regressor or classifier
# We will display this in the sidebar
algorithm_type = st.sidebar.selectbox("Select Algorithm Type",("Classifier", "Regressor"))

# Now we need to load the builtin dataset
# This is done using the load_dataset_name function
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
# Defining Function for Input and Output
def Input_output(data, data_name):
    if data_name == "Salary":
        X, Y = data['YearsExperience'].to_numpy().reshape(-1, 1), data['Salary'].to_numpy().reshape(-1, 1)
    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data['diabetes']
    elif data_name == "Car Evaluation":
        df = data
        # For converting string columns to numeric values
        le = LabelEncoder()
        # Function to convert string values to numeric values
        func = lambda i: le.fit(df[i]).transform(df[i])
        for i in df.columns:
            df[i] = func(i)
        X, Y = df.drop(['unacc'], axis=1), df['unacc']
    else:
        # We use data.data as we need to copy data to X which is Input
        X = data.data
        # Since this is built in dataset we can directly load output or target class by using data.target function
        Y = data.target
    return X, Y

# Calling Function to get Input and Output
X, Y = Input_output(data, data_name)

# Adding Parameters so that we can select from various parameters for KNN
def add_parameter_knn(algorithm_type):
    # Declaring a dictionary for storing parameters
    params = dict()
    # Adding number of Neighbour in KNN
    k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20, key="k_n_slider")
    # Adding in dictionary
    params['K'] = k_n
    # Adding weights
    weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
    # Adding to dictionary
    params['weights'] = weights_custom
    return params

# Adding Parameters for KNN
params = add_parameter_knn(algorithm_type)

# Now we will build ML Model for this dataset and calculate accuracy for that for classifier
def model_classifier_knn(params):
    return KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])

# Now we will build ML Model for this dataset and calculate accuracy for that for regressor
def model_regressor_knn(params):
    return KNeighborsRegressor(n_neighbors=params['K'], weights=params['weights'])

# Now we will write the dataset information
# Since diabetes is a regression dataset, it does not have classes
def info(data_name):
    if data_name not in ["Diabetes", "Salary", "Naive Bayes Classification", "Car Evaluation"]:
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : KNN {algorithm_type}')
        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)
        st.write('Number of classes: ', len(np.unique(Y)))
        # Making a dataframe to store target name and value
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": data.target_names})
        # Display the DataFrame without index labels
        st.write('Values and Name of Classes')
        # Display the DataFrame as a Markdown table
        # To successfully run this we need to install tabulate
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")
    elif data_name == "Diabetes":
        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is : KNN {algorithm_type}')
        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)
    elif data_name == 'Salary':
        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is : KNN {algorithm_type}')
        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)
    elif data_name == "Naive Bayes Classification":
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : KNN {algorithm_type}')
        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)
        st.write('Number of classes: ', len(np.unique(Y)))
        # Making a dataframe to store target name and value
        df = pd.DataFrame({"Target Value": list(np.unique(Y)),
                           "Target Name": ['Not Diabetic', 'Diabetic']})
        # Display the DataFrame without index labels
        st.write('Values and Name of Classes')
        # Display the DataFrame as a Markdown table
        # To successfully run this we need to install tabulate
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")
    else:
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : KNN {algorithm_type}')
        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)
        st.write('Number of classes: ', len(np.unique(Y)))
        # Making a dataframe to store target name and value
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": ['Unacceptable', 'Acceptable', 'Good Condition', 'Very Good Condition']})
        # Display the DataFrame without index labels
        st.write('Values and Name of Classes')
        # Display the DataFrame as a Markdown table
        # To successfully run this we need to install tabulate
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")

# Calling function to print Dataset Information
info(data_name)

# Now selecting classifier or regressor
# Calling Function based on regressor and classifier
if algorithm_type == "Regressor":
    algo_model = model_regressor_knn(params)
else:
    algo_model = model_classifier_knn(params)

# Now splitting into Testing and Training data
# It will split into 80 % training data and 20 % Testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

# Training algorithm
algo_model.fit(x_train, y_train)

# Now we will find the predicted values
predict = algo_model.predict(x_test)

# Finding Accuracy
# Evaluating/Testing the model
if algorithm_type != 'Regressor':
    # For all algorithm we will find accuracy
    st.write("Training Accuracy is:", algo_model.score(x_train, y_train) * 100)
    st.write("Testing Accuracy is:", accuracy_score(y_test, predict) * 100)
else:
    # Checking for Error
    # Error is less as accuracy is more
    # For regression we will find error
    st.write("Mean Squared error is:", mean_squared_error(y_test, predict))
    st.write("Mean Absolute error is:", mean_absolute_error(y_test, predict))

# Plotting Dataset
# Since there are many dimensions, first we will do Principle Component analysis to do dimension reduction and then plot
pca = PCA
