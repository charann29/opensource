# Importing required Libraries
# Importing Numpy
import numpy as np
# To read csv file
import pandas as pd
# Importing datasets from sklearn
from sklearn import datasets
# For splitting between training and testing
from sklearn.model_selection import train_test_split
# Importing Knn algorithm
from sklearn.neighbors import KNeighborsClassifier
# Importing accuracy score
from sklearn.metrics import accuracy_score
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
st.title("KNN Classifier on Inbuilt and Kaggle Datasets")

# Now we are making a select box for dataset
data_name=st.sidebar.selectbox("Select Dataset",
                  ("Iris","Breast Cancer","Wine","Diabetes","Digits","Salary","Naive Bayes Classification","Car Evaluation"))

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
    elif Data == "Naive Bayes Classification" :
        return pd.read_csv("Naive-Bayes-Classification-Data.csv")
    else :
        return pd.read_csv("car_evaluation.csv")

# Now we need to call function to load the dataset
data = load_dataset(data_name)

# Now after this we need to split between input and output

# Defining Function for Input and Output
def Input_output(data,data_name):

    if data_name == "Salary":
        X, Y = data['YearsExperience'].to_numpy().reshape(-1, 1), data['Salary'].to_numpy().reshape(-1, 1)

    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data['diabetes']

    elif data_name == "Car Evaluation":

        df= data

        # For converting string columns to numeric values
        le = LabelEncoder()

        # Function to convert string values to numeric values
        func = lambda i: le.fit(df[i]).transform(df[i])
        for i in df.columns:
            df[i] = func(i)

        X, Y = df.drop(['unacc'], axis=1), df['unacc']

    else :
        # We use data.data as we need to copy data to X which is Input
        X = data.data
        # Since this is built in dataset we can directly load output or target class by using data.target function
        Y = data.target

    return X,Y

# Calling Function to get Input and Output
X , Y = Input_output(data,data_name)

# Adding Parameters so that we can select from various parameters for KNN
def add_parameter_knn():
    # Declaring a dictionary for storing parameters
    params = dict()
    # Adding number of Neighbour in Classifier
    k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20)
    # Adding in dictionary
    params['K'] = k_n
    # Adding weights
    weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
    # Adding to dictionary
    params['weights'] = weights_custom
    return params

# Calling function to get KNN parameters
params = add_parameter_knn()

# Now we will build ML Model for this dataset and calculate accuracy for that for classifier
def model_knn(params):
    return KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])

# Now we will write the dataset information
# Since diabetes is a regression dataset, it does not have classes
def info(data_name):

    if data_name not in ["Diabetes","Salary","Naive Bayes Classification","Car Evaluation"]:
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : KNN Classifier')

        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)
        st.write('Number of classes: ', len(np.unique(Y)))
        # Making a dataframe to store target name and value

        df = pd.DataFrame({"Target Value" : list(np.unique(Y)),"Target Name" : data.target_names})
        # Display the DataFrame without index labels
        st.write('Values and Name of Classes')

        # Display the DataFrame as a Markdown table
        # To successfully run this we need to install tabulate
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")

    elif data_name == "Diabetes":

        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is : KNN Classifier')

        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)

    elif data_name == 'Salary':

        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is : KNN Classifier')

        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)

    elif data_name == "Naive Bayes Classification":

        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : KNN Classifier')

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
        st.write(f'Algorithm is : KNN Classifier')

        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)
        st.write('Number of classes: ', len(np.unique(Y)))
        # Making a dataframe to store target name and value

        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": ['Unacceptable','Acceptable','Good Condition','Very Good Condition']})

        # Display the DataFrame without index labels
        st.write('Values and Name of Classes')

        # Display the DataFrame as a Markdown table
        # To successfully run this we need to install tabulate
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")

# Calling function to print Dataset Information
info(data_name)

# Now selecting KNN Classifier
algo_model = model_knn(params)

# Now splitting into Testing and Training data
# It will split into 80 % training data and 20 % Testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

# Training algorithm
algo_model.fit(x_train,y_train)

# Now we will find the predicted values
predict=algo_model.predict(x_test)

# Finding Accuracy
# Evaluating/Testing the model
st.write("Training Accuracy is:",algo_model.score(x_train,y_train)*100)
st.write("Testing Accuracy is:",accuracy_score(y_test,predict)*100)

# Plotting Dataset
# Since there are many dimensions, first we will do Principle Component analysis to do dimension reduction and then plot
pca=PCA(2)

# Salary and Naive bayes classification data does not need pca
if data_name != "Salary":
    X = pca.fit_transform(X)

# Plotting
fig = plt.figure()

# Now while plotting we have to show target variables for datasets
# Now since diabetes is regression dataset it dosen't have target variables
# So we have to apply condition and plot the graph according to the dataset
# Seaborn is used as matplotlib does not display all label names

def choice_classifier(data_name):

    # Plotting Regression Plot for dataset diabetes
    # Since this is a regression dataset we show regression line as well
    if data_name == "Diabetes":
        plt.scatter(X[:, 0], X[:,1], c=Y, cmap='viridis', alpha=0.8)
        plt.title("Scatter Classification Plot of Dataset")
        plt.colorbar()

    # Plotting for digits
    # Since this dataset has many classes/target values we can plot it using seaborn
    # Also viridis will be ignored here and it will plot by default according to its own settings
    # But we can set Color palette according to our requirements
    # We need not to give data argument else it gives error
    # Hue paramter is given to show target variables
