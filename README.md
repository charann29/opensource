
# ML Model Datasets Using Streamlits 
This repository contains my machine learning models implementation code using streamlit in the Python programming language.<br><br>
<a href = "https://ml-model-datasets-using-apps-3gy37ndiancjo2nmu36sls.streamlit.app/"><img width="960" title = "Website Image" alt="Website Image" src="https://github.com/madhurimarawat/ML-Model-Datasets-Using-Streamlits/assets/105432776/64584f95-f19c-426d-b647-a6310d0f0d2d"></a>
<br><br>
<a href = "https://ml-model-datasets-using-apps-3gy37ndiancjo2nmu36sls.streamlit.app/"><img width="960" title = "Website Image" alt="Website Image" src="https://github.com/madhurimarawat/ML-Model-Datasets-Using-Streamlits/assets/105432776/3c872711-3e6c-4d4a-a216-4cf5b7d98361"></a>


---
# Mode of Execution Used <img src="https://th.bing.com/th/id/R.c936445e15a65dfdba20a63e14e7df39?rik=fqWqO9kKIVlK7g&riu=http%3a%2f%2fassets.stickpng.com%2fimages%2f58481537cef1014c0b5e4968.png&ehk=dtrTKn1QsJ3%2b2TFlSfLR%2fxHdNYHdrqqCUUs8voipcI8%3d&risl=&pid=ImgRaw&r=0" title="PyCharm" alt="PyCharm" width="40" height="40">&nbsp;<img src="https://seeklogo.com/images/S/streamlit-logo-1A3B208AE4-seeklogo.com.png" title="Streamlit" alt="Streamlit" width="40" height="40">



# PyCharm and Streamlit Overview

## PyCharm

### Official Website

Visit the official website of PyCharm: [JetBrains PyCharm](https://www.jetbrains.com/pycharm/)


### Download

Download PyCharm according to your platform:
- [Linux](https://www.jetbrains.com/pycharm/download/#section=linux)
- [macOS](https://www.jetbrains.com/pycharm/download/#section=mac)
- [Windows](https://www.jetbrains.com/pycharm/download/#section=windows)

### Versions

1. **Community Version**
   - Free and open-source.
   - Available for download at the end of the PyCharm website.
   - Setup via the setup wizard.

2. **Professional Version**
   - Available at the top of the PyCharm website.
   - Download and follow setup instructions.
   - Choose between free trial or paid version.

### Using PyCharm

- **Virtual Environment**: Each project has its own virtual environment for installing libraries.
- **File Types**: Supports script files, text files, and Jupyter Notebooks.
- **Execution**: Press `Shift+F10` (Windows) to execute the current file.
- **Output**: Results appear in the Console; installations are managed in the terminal.

---

## Streamlit Server

### Overview

- **Streamlit** is a Python framework for deploying machine learning models and other Python projects.
- It simplifies frontend development by offering predefined functions for frontend components.

### Installation

To install Streamlit, run the following command in your terminal:

```pip install streamlit```


### Usage

- Deploy any machine learning model or Python project with ease.
- Streamlit handles the frontend so you can focus on the backend logic.
- It's user-friendly and ideal for quick prototyping and deployments.

---

## Running Project in Streamlit Server
<p>Make Sure all dependencies are already satisfied before running the app.</p>

1. We can Directly run streamlit app  with the following command-<br>
```
streamlit run app.py
```
where app.py is the name of file containing streamlit code.<br>

By default, streamlit will run on port 8501.<br>

Also we can execute multiple files simultaneously and it will be executed in next ports like 8502 and so on.<br>

2. Navigate to URL http://localhost:8501

You should be able to view the homepage of your app.

🌟 Project and Models will change but this process will remain the same for all Streamlit projects.<br>

## Deploying using Streamlit

1. Visit the official website of streamlit : <a href="https://streamlit.io/"><img src="https://seeklogo.com/images/S/streamlit-logo-1A3B208AE4-seeklogo.com.png" title="Streamlit" alt="Streamlit" width="40" height="40"></a> <br><br>
2. Now make an account with GitHub.<br><br>
3. Now add all the code in Github repository.<br><br>
4. Go to streamlit and there is an option for new deployment.<br><br>
5. Type your Github repository name and specify the file name. If you name your file as streamlit_app it will directly access it else you have to specify the path.<br><br>
6. Now also make sure you upload all your libraries and requirement name in a requirement.txt file.<br><br>
7. Version can also be mentioned like this python==3.9.<br><br>
8. When we mention version in the requirement file streamlit install all dependencies from there.<br><br>
9. If everything went well our app will be deployed on web and you can share the link and access the app from all browsers.

---


## About Project

- Complete Description about the project and resources used.
- Developed a Streamlit website allowing application of multiple supervised learning algorithms on various datasets.
- Implemented Data Visualization to illustrate the algorithms' performance on the datasets.
- Deployed the website using Streamlit.
- Visit the website: [ML Algorithms on Inbuilt and Kaggle Datasets](https://ml-model-datasets-using-apps-3gy37ndiancjo2nmu36sls.streamlit.app/)

---

## Algorithm Used

### Supervised Learning

- Basically supervised learning is when we teach or train the machine using data that is well-labelled.
- Which means some data is already tagged with the correct answer.
- After that, the machine is provided with a new set of examples (data) so that the supervised learning algorithm analyses the training data (set of training examples) and produces a correct outcome from labeled data.

#### i) K-Nearest Neighbors (KNN)

- K-Nearest Neighbours is one of the most basic yet essential classification algorithms in Machine Learning.
- It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining, and intrusion detection.
- In this algorithm, we identify categories based on neighbors.

#### ii) Support Vector Machines (SVM)

- The main idea behind SVMs is to find a hyperplane that maximally separates the different classes in the training data.
- This is done by finding the hyperplane that has the largest margin, which is defined as the distance between the hyperplane and the closest data points from each class.
- Once the hyperplane is determined, new data can be classified by determining on which side of the hyperplane it falls.
- SVMs are particularly useful when the data has many features, and/or when there is a clear margin of separation in the data.

#### iii) Naive Bayes Classifiers

- Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem.
- It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e., every pair of features being classified is independent of each other.
- The fundamental Naive Bayes assumption is that each feature makes an independent and equal contribution to the outcome.

#### iv) Decision Tree

- It builds a flowchart-like tree structure where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.
- It is constructed by recursively splitting the training data into subsets based on the values of the attributes until a stopping criterion is met, such as the maximum depth of the tree or the minimum number of samples required to split a node.
- The goal is to find the attribute that maximizes the information gain or the reduction in impurity after the split.

#### v) Random Forest

- It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.
- Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.
- The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.

#### vi) Linear Regression

- Regression: It predicts the continuous output variables based on the independent input variable, like the prediction of house prices based on different parameters such as house age, distance from the main road, location, area, etc.
- It computes the linear relationship between a dependent variable and one or more independent features.
- The goal of the algorithm is to find the best linear equation that can predict the value of the dependent variable based on the independent variables.

#### vii) Logistic Regression

- Logistic regression is a supervised machine learning algorithm mainly used for classification tasks where the goal is to predict the probability that an instance of belonging to a given class or not.
- It is a kind of statistical algorithm, which analyzes the relationship between a set of independent variables and the dependent binary variables.
- It is a powerful tool for decision-making, for example email spam or not.

---

## Dataset Used

### Iris Dataset

- Iris Dataset is a part of sklearn library.
- Sklearn comes loaded with datasets to practice machine learning techniques and iris is one of them.
- Iris has 4 numerical features and a tri class target variable.
- This dataset can be used for classification as well as clustering.
- In this dataset, there are 4 features sepal length, sepal width, petal length, and petal width and the target variable has 3 classes namely ‘setosa’, ‘versicolor’, and ‘virginica’.
- Objective for a multiclass classifier is to predict the target class given the values for the four features.
- Dataset is already cleaned, no preprocessing required.

### Breast Cancer Dataset

- The breast cancer dataset is a classification dataset that contains 569 samples of malignant and benign tumor cells.
- The samples are described by 30 features such as mean radius, texture, perimeter, area, smoothness, etc.
- The target variable has 2 classes namely ‘benign’ and ‘malignant’.
- Objective for a multiclass classifier is to predict the target class given the values for the features.
- Dataset is already cleaned, no preprocessing required.

### Wine Dataset

- The wine dataset is a classic and very easy multi-class classification dataset that is available in the sklearn library.
- It contains 178 samples of wine with 13 features and 3 classes.
- The goal is to predict the class of wine based on the features.
- Dataset is already cleaned, no preprocessing required.

### Digits Dataset

- The digits dataset is a classic multi-class classification dataset that is available in the sklearn library.
- It contains 1797 samples of digits with 10 classes.
- The goal is to predict the class of digit based on the features.
- Dataset is already cleaned, no preprocessing required.

### Diabetes Dataset

- The diabetes dataset is a regression dataset that is available in the sklearn library.
- It contains 442 samples and 10 classes.
- Dataset is already cleaned, no preprocessing required.

### Naive Bayes Classification Data

- Dataset is taken from: [Naive bayes classification data](https://www.kaggle.com/datasets/himanshunakrani/naive-bayes-classification-data)
- Contains diabetes data for classification.
- The dataset has 3 columns-glucose, blood pressure, and diabetes and 995 entries.
- Column glucose and blood pressure data is to classify whether the patient has diabetes or not.
- Dataset is already cleaned, no preprocessing required.

### Cars Evaluation Dataset

- Dataset is taken from: [Cars Evaluation Dataset](https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set)
- Contains information about cars with respect to features like Attribute Values:
  - buying v-high, high, med, low
  - maint v-high, high, med, low
  - doors 2, 3, 4, 5-more
  - persons 2, 4, more
  - lug_boot small, med, big
  - safety low, med, high
- Target categories are:
  - unacc 1210 (70.023 %)
  - acc 384 (22.222 %)
  - good 69 ( 3.993 %)
  - v-good 65 ( 3.762 %)
- Contains Values in string format.
- Dataset is not cleaned, preprocessing is required.

### Salary Dataset

- Dataset is taken from: [Salary Dataset](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression)
- Contains Salary data for Regression.
- The dataset has 2 columns-Years of Experience and Salary and 30 entries.
- Column Years of Experience is used to find regression for Salary.
- Dataset is already cleaned, no preprocessing required.

---


# Libraries Used 📚 💻

<p>Below is a short description of all the libraries used:</p>

To install a Python library, use the following command:

```pip install library_name```


- **NumPy (Numerical Python)**:
  NumPy enables efficient collection of mathematical functions to operate on arrays and matrices.

- **Pandas (Python Data Analysis)**:
  Pandas is primarily used for analyzing, cleaning, exploring, and manipulating data.

- **Matplotlib**:
  Matplotlib is a versatile data visualization and plotting library.

- **Scikit-learn**:
  Scikit-learn is a comprehensive machine learning library that provides tools for various machine learning algorithms such as classification, regression, clustering, etc.

- **Seaborn**:
  Seaborn is built on top of Matplotlib and offers enhanced statistical graphics for better data visualization.



# Resources

- [Classification versus Regression in Machine Learning](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)

This resource provides an in-depth explanation of the differences between classification and regression tasks in machine learning, detailing their purposes, methods, and use cases.

  # Code Imports

  
#### Importing Numpy

    import numpy as np

#### Why? Numpy is fundamental for numerical computations, providing support for arrays and matrices, and mathematical functions.

#### To read csv file

    import pandas as pd

#### Why? Pandas is used for data manipulation and analysis, handling numerical tables and time series, and reading various file formats.

#### Importing datasets from sklearn

    from sklearn import datasets

#### Why? Scikit-learn provides built-in datasets useful for practicing machine learning algorithms.

#### For splitting between training and testing

    from sklearn.model_selection import train_test_split

#### Why? This module splits datasets into training and testing sets, crucial for evaluating model performance.

#### Importing Algorithm for Support Vector Machine

    from sklearn.svm import SVC, SVR

#### Why? SVC and SVR are Support Vector Machine implementations for classification and regression tasks.

#### Importing K-nearest neighbors algorithm

    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#### Why? KNeighborsClassifier and KNeighborsRegressor are implementations of k-nearest neighbors for classification and regression tasks.

#### Importing Decision Tree algorithm

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

#### Why? DecisionTreeClassifier and DecisionTreeRegressor are implementations of decision tree algorithms.

#### Importing Random Forest Classifer

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#### Why? RandomForestClassifier and RandomForestRegressor are ensemble methods based on decision trees for classification and regression tasks.

#### Importing Naive Bayes algorithm

    from sklearn.naive_bayes import GaussianNB

#### Why? GaussianNB is a Naive Bayes classifier implementation suitable for classification tasks.

#### Importing Linear and Logistic Regression

    from sklearn.linear_model import LinearRegression, LogisticRegression

#### Why? LinearRegression models linear relationships between variables. LogisticRegression is for binary classification tasks.

#### Importing accuracy score and mean_squared_error

    from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error

#### Why? These metrics evaluate model performance: mean_squared_error for regression, accuracy_score for classification accuracy, and mean_absolute_error for regression.

#### Importing PCA for dimension reduction

    from sklearn.decomposition import PCA

#### Why? PCA reduces data dimensionality by projecting it onto a lower-dimensional space.

#### For Plotting

    import matplotlib.pyplot as plt
    import seaborn as sns

#### Why? Matplotlib and Seaborn are used for visualizing data, exploring patterns, and presenting results.

#### For model deployment

    import streamlit as st

#### Why? Streamlit is a framework for building web applications for machine learning and data science, used here for deploying models and creating interactive data applications.

#### Importing Label Encoder for converting string to int

    from sklearn.preprocessing import LabelEncoder

#### Why? LabelEncoder converts categorical labels to numerical labels, necessary for many machine learning algorithms.

