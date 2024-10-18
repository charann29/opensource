### **Title: Movie Success Rate Prediction Project**

### **Description:**
**This pull request introduces a new project aimed at predicting the success rate of movies based on various features such as genre, director, runtime, rating, votes, revenue, and metascore. The project leverages machine learning models to make predictions and is designed to assist movie directors and producers in evaluating the potential success of their movies before release.**

### **Key Features:**

### **Movie Success Rate Prediction:**
**Implements three different machine learning models: Naive Bayes, Logistic Regression, and Support Vector Machine (SVM) to predict the success rate of movies.
Utilizes a dataset of movies for training and testing the models.**

### **User Interface with Streamlit:**
**Provides a user-friendly interface built with Streamlit.
Allows users to input details of a new movie and predict its success rate.
Displays predictions as percentages for better clarity.**

### **Data Handling and Preprocessing:**
**Cleans and preprocesses the dataset to handle missing values and encode categorical variables.
Normalizes numerical features to improve model performance.**

### **Deployment:**
**Instructions included for setting up the project environment and running the Streamlit app.
Ensures the environment folder (env) and dataset (Movie-DataSet.csv) are ignored in version control to maintain a clean repository.**

### **Changes Made:**
**Added movie_success_prediction.py: Main script for the Streamlit app.
Added Movie-DataSet.csv: Dataset used for training and testing models.
Added .gitignore: To exclude the Python environment folder (env) and dataset from version control.**

### **How to Run:**
**Clone the repository and navigate to the project directory.
Ensure you have the required dependencies installed.
Run the Streamlit app with the command:
streamlit run movie_success_prediction.py**

### **UI Screenshots of My Project:**
![Movie-Success-Prediction-UI-SS-1](https://github.com/charann29/cmr_opensource/assets/147246984/445eccd4-33d2-4de5-846e-ee2324231673)

### **Tested it with 2 TestCases of Recent Movies and their one week collection to predict their Long Run Success Rate :**

**TEST - 1 (Dune : Part Two)**

![Movie-Success-Prediction-UI-SS-2](https://github.com/charann29/cmr_opensource/assets/147246984/4d3324a8-cf63-4997-a223-beae36bd0012)
![Movie-Success-Prediction-UI-SS-3](https://github.com/charann29/cmr_opensource/assets/147246984/36d35f91-40da-4069-a40a-f33e06d328d8)

**TEST - 2 (Oppenheimer)**

![Movie-Success-Prediction-UI-SS-4](https://github.com/charann29/cmr_opensource/assets/147246984/6dd40b8e-f73d-460c-8737-e43cb77d0417)
![Movie-Success-Prediction-UI-SS-5](https://github.com/charann29/cmr_opensource/assets/147246984/f414a73e-588b-424e-bfd7-56a316933a8a)


### **Future Improvements:**
**Expand the dataset with more movie features.
Explore additional machine learning models and techniques to enhance prediction accuracy.
Incorporate user feedback to improve the interface and functionality.
Note:
This project is intended for educational purposes and demonstrates the application of machine learning in predicting movie success rates. Further refinement and validation are necessary for production-level deployment.**

**Thank you for considering this pull request. I look forward to your feedback and suggestions.**