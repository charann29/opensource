# In this program we will apply various ML algorithms to the built in datasets in scikit-learn

# Importing required Libraries
# Importing Numpy
import numpy as np
# To read csv file
import pandas as pd
# Importing datasets from sklearn
from sklearn import datasets
# For splitting between training and testing
from sklearn.model_selection import train_test_split
# Importing Algorithm for Simple Vector Machine
from sklearn.svm import SVC, SVR
# Importing Knn algorithm
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Importing  Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# Importing Random Forest Classifer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# Importing Naive Bayes algorithm
from sklearn.naive_bayes import GaussianNB
# Importing Linear and Logistic Regression
from sklearn.linear_model import LinearRegression,LogisticRegression
# Importing accuracy score and mean_squared_error
from sklearn.metrics import mean_squared_error, accuracy_score,mean_absolute_error
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
st.title("ML Algorithms on Inbuilt and Kaggle Datasets")

# Now we are making a select box for dataset
data_name=st.sidebar.selectbox("Select Dataset",
                  ("Iris","Breast Cancer","Wine","Diabetes","Digits","Salary","Naive Bayes Classification","Car Evaluation"))

# The Next is selecting algorithm
# We will display this in the sidebar
algorithm=st.sidebar.selectbox("Select Supervised Learning Algorithm",
                     ("KNN","SVM","Decision Tree","Naive Bayes","Random Forest","Linear Regression","Logistic Regression"))

# The Next is selecting regressor or classifier
# We will display this in the sidebar
if algorithm != 'Linear Regression' and algorithm != 'Logistic Regression' and algorithm != "Naive Bayes":
    algorithm_type = st.sidebar.selectbox("Select Algorithm Type",
                        ("Classifier","Regressor"))
else:
    st.sidebar.write(f"In {algorithm} Classifier and Regressor dosen't exist separately")
    if algorithm == "Linear Regression":
        algorithm_type = "Regressor"
        st.sidebar.write("{} only does Regression".format(algorithm))
    else:
        algorithm_type = "Classifier"
        st.sidebar.write(f"{algorithm} only does Classification")

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

# Adding Parameters so that we can select from various parameters for classifier
def add_parameter_classifier_general(algorithm):

    # Declaring a dictionary for storing parameters
    params = dict()

    # Deciding parameters based on algorithm

    # Adding paramters for SVM
    if algorithm == 'SVM':

        # Adding regularization parameter from range 0.01 to 10.0
        c_regular = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
        # Kernel is the arguments in the ML model
        # Polynomial ,Linear, Sigmoid and Radial Basis Function are types of kernals which we can add
        kernel_custom = st.sidebar.selectbox('Kernel', ('linear', 'poly ', 'rbf', 'sigmoid'))
        # Adding in dictionary
        params['C'] = c_regular
        params['kernel'] = kernel_custom

    # Adding Parameters for KNN
    elif algorithm == 'KNN':

        # Adding number of Neighbour in Classifier
        k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20,key="k_n_slider")
        # Adding in dictionary
        params['K'] = k_n
        # Adding weights
        weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
        # Adding to dictionary
        params['weights'] = weights_custom

    # Adding Parameters for Naive Bayes
    # It doesn't have any paramter
    elif algorithm == 'Naive Bayes':
        st.sidebar.info("This is a simple Algorithm. It doesn't have Parameters for Hyper-tuning.")

    # Adding Parameters for Decision Tree
    elif algorithm == 'Decision Tree':

        # Taking max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)
        # Adding criterion
        # mse is for regression- It is used in DecisionTreeRegressor
	    # mse will give error in classifier so it is removed
        criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
        # Adding splitter
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        # Taking random state
        # Adding to dictionary
        params['max_depth'] = max_depth
        params['criterion'] = criterion
        params['splitter'] = splitter

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Adding Parameters for Random Forest
    elif algorithm == 'Random Forest':

        # Taking max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)
        # Adding number of estimators
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 90)
        # Adding criterion
        # mse is for regression- It is used in RandomForestRegressor
	    # mse will give error in classifier so it is removed
        criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy', 'log_loss'))
        # Adding to dictionary
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        params['criterion'] = criterion

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Adding Parameters for Logistic Regression
    else:

        # Adding regularization parameter from range 0.01 to 10.0
        c_regular = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
        params['C'] = c_regular
        # Taking fit_intercept
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ('True', 'False'))
        params['fit_intercept'] = bool(fit_intercept)
        # Taking Penalty only l2 and None is supported
        penalty = st.sidebar.selectbox("Penalty", ('l2', None))
        params['penalty'] = penalty
        # Taking n_jobs
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params['n_jobs'] = n_jobs

    return params

# Adding Parameters so that we can select from various parameters for regressor
def add_parameter_regressor(algorithm):

    # Declaring a dictionary for storing parameters
    params = dict()

    # Deciding parameters based on algorithm
    # Adding Parameters for Decision Tree
    if algorithm == 'Decision Tree':

        # Taking max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)
        # Adding criterion
        # mse is for regression- It is used in DecisionTreeRegressor
        criterion = st.sidebar.selectbox('Criterion', ('absolute_error', 'squared_error', 'poisson', 'friedman_mse'))
        # Adding splitter
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        # Taking random state
        # Adding to dictionary
        params['max_depth'] = max_depth
        params['criterion'] = criterion
        params['splitter'] = splitter

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Adding Parameters for Linear Regression
    elif algorithm == 'Linear Regression':

        # Taking fit_intercept
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ('True', 'False'))
        params['fit_intercept'] = bool(fit_intercept)
        # Normalize does not work in linear regression
        # Taking n_jobs
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params['n_jobs'] = n_jobs

    # Adding Parameters for Random Forest
    else:

        # Taking max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)
        # Adding number of estimators
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 90)
        # Adding criterion
        # mse is for regression- It is used in RandomForestRegressor
        criterion = st.sidebar.selectbox('Criterion', ('absolute_error', 'squared_error', 'poisson', 'friedman_mse'))
        # Adding to dictionary
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        params['criterion'] = criterion

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    return params

# Calling Function based on regressor and classifier
# Here since the parameters for regressor and classifier are same for some algorithm we can directly use this
# Because of this here except for this three algorithm we do not need to take parameters separately
if (algorithm_type == "Regressor") and (algorithm == 'Decision Tree' or algorithm == 'Random Forest' or algorithm_type == "Linear Regression"):
    params = add_parameter_regressor(algorithm)
else :
    params = add_parameter_classifier_general(algorithm)

# Now we will build ML Model for this dataset and calculate accuracy for that for classifier
def model_classifier(algorithm, params):

    if algorithm == 'KNN':
        return KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])

    elif algorithm == 'SVM':
        return SVC(C=params['C'], kernel=params['kernel'])

    elif algorithm == 'Decision Tree':
        return DecisionTreeClassifier(
            criterion=params['criterion'], splitter=params['splitter'],
            random_state=params['random_state'])

    elif algorithm == 'Naive Bayes':
        return GaussianNB()

    elif algorithm == 'Random Forest':
        return RandomForestClassifier(n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'],
                                      criterion=params['criterion'],
                                      random_state=params['random_state'])

    elif algorithm == 'Linear Regression':
        return LinearRegression(fit_intercept=params['fit_intercept'], n_jobs=params['n_jobs'])

    else:
        return LogisticRegression(fit_intercept=params['fit_intercept'],
                                  penalty=params['penalty'], C=params['C'], n_jobs=params['n_jobs'])


# Now we will build ML Model for this dataset and calculate accuracy for that for regressor
def model_regressor(algorithm, params):

    if algorithm == 'KNN':
        return KNeighborsRegressor(n_neighbors=params['K'], weights=params['weights'])

    elif algorithm == 'SVM':
        return SVR(C=params['C'], kernel=params['kernel'])

    elif algorithm == 'Decision Tree':
        return DecisionTreeRegressor(
            criterion=params['criterion'], splitter=params['splitter'],
            random_state=params['random_state'])

    elif algorithm == 'Random Forest':
        return RandomForestRegressor(n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'],
                                      criterion=params['criterion'],
                                      random_state=params['random_state'])

    else:
        return LinearRegression(fit_intercept=params['fit_intercept'], n_jobs=params['n_jobs'])

# Now we will write the dataset information
# Since diabetes is a regression dataset, it does not have classes
def info(data_name):

    if data_name not in ["Diabetes","Salary","Naive Bayes Classification","Car Evaluation"]:
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')

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
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')

        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)

    elif data_name == 'Salary':

        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')

        # Printing shape of data
        st.write('Shape of Dataset is: ', X.shape)

    elif data_name == "Naive Bayes Classification":

        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')

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
        st.write(f'Algorithm is : {algorithm}')

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

# Now selecting classifier or regressor
# Calling Function based on regressor and classifier
if algorithm_type == "Regressor":
    algo_model = model_regressor(algorithm,params)
else :
    algo_model = model_classifier(algorithm,params)

# Now splitting into Testing and Training data
# It will split into 80 % training data and 20 % Testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

# Training algorithm
algo_model.fit(x_train,y_train)

# Now we will find the predicted values
predict=algo_model.predict(x_test)

# Finding Accuracy
# Evaluating/Testing the model
if algorithm != 'Linear Regression' and algorithm_type != 'Regressor':
    # For all algorithm we will find accuracy
    st.write("Training Accuracy is:",algo_model.score(x_train,y_train)*100)
    st.write("Testing Accuracy is:",accuracy_score(y_test,predict)*100)
else:
    # Checking for Error
    # Error is less as accuracy is more
    # For linear regression we will find error
    st.write("Mean Squared error is:",mean_squared_error(y_test,predict))
    st.write("Mean Absolute error is:",mean_absolute_error(y_test,predict))

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
    elif data_name == "Digits":
        colors = ['purple', 'green', 'yellow', 'red', 'black', 'cyan', 'pink', 'magenta', 'grey', 'teal']
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), cmap="viridis", alpha=0.4)
        # Giving legend
        # If we try to show the class target name it will show in different color than the ones that are plotted
        plt.legend(data.target_names, shadow=True)
        # Giving Title
        plt.title("Scatter Classification Plot of Dataset With Target Classes")

    elif data_name == "Salary":
        sns.scatterplot(x=data['YearsExperience'],y=data['Salary'],data=data)
        plt.xlabel('Years of Experience')
        plt.ylabel("Salary")
        plt.title("Scatter Classification Plot of Dataset")

    elif data_name == "Naive Bayes Classification":
        colors = ['purple', 'green']
        sns.scatterplot(x=data['glucose'], y=data['bloodpressure'],data=data, hue=Y, palette=sns.color_palette(colors), alpha=0.4)
        plt.legend(shadow=True)
        plt.xlabel('Glucose')
        plt.ylabel("Blood Pressure")
        plt.title("Scatter Classification Plot of Dataset With Target Classes")


    # We cannot give data directly we have to specify the values for x and y
    else:
        colors = ['purple', 'green', 'yellow','red']
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), alpha=0.4)
        plt.legend(shadow=True)
        plt.title("Scatter Classification Plot of Dataset With Target Classes")


def choice_regressor(data_name):

    # Plotting Regression Plot for dataset diabetes
    # Since this is a regression dataset we show regression line as well
    if data_name == "Diabetes":
        plt.scatter(X[:, 0], Y, c=Y, cmap='viridis', alpha=0.4)
        plt.plot(x_test, predict, color="red")
        plt.title("Scatter Regression Plot of Dataset")
        plt.legend(['Actual Values', 'Best Line or General formula'])
        plt.colorbar()

    # Plotting for digits
    # Since this dataset has many classes/target values we can plot it using seaborn
    # Also viridis will be ignored here and it will plot by default according to its own settings
    # But we can set Color palette according to our requirements
    # We need not to give data argument else it gives error
    # Hue paramter is given to show target variables
    elif data_name == "Digits":
        colors = ['purple', 'green', 'yellow', 'red', 'black', 'cyan', 'pink', 'magenta', 'grey', 'teal']
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), cmap="viridis", alpha=0.4)
        plt.plot(x_test, predict, color="red")
        # Giving legend
        # If we try to show the class target name it will show in different color than the ones that are plotted
        plt.legend(data.target_names, shadow=True)
        # Giving Title
        plt.title("Scatter Plot of Dataset With Target Classes")

    elif data_name == "Salary":
        sns.scatterplot(x=data['YearsExperience'],y=data['Salary'],data=data)
        plt.plot(x_test, predict, color="red")
        plt.xlabel('Years of Experience')
        plt.ylabel("Salary")
        plt.legend(['Actual Values', 'Best Line or General formula'])
        plt.title("Scatter Regression Plot of Dataset")

    # We cannot give data directly we have to specify the values for x and y
    else:
        plt.scatter(X[:, 0], X[:, 1],cmap='viridis',c=Y, alpha=0.4)
        plt.plot(x_test, predict, color="red")
        plt.legend(['Actual Values', 'Best Line or General formula'])
        plt.colorbar()
        plt.title("Scatter Regression Plot of Dataset With Target Classes")

if algorithm_type == 'Regressor':
    choice_regressor(data_name)
else:
    # Calling Function
    choice_classifier(data_name)

if data_name != "Salary" and data_name != "Naive Bayes Classification":
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

# Since we have done pca in naive bayes classification data for plotting regression plot
if data_name == "Naive Bayes Classification" and algorithm_type == 'Regressor':
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
st.pyplot(fig)
