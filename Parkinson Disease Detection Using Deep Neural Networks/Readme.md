# Parkinson Disease Detection Using Deep Neural Networks

## Project Overview
This project aims to detect Parkinson's disease using machine learning and deep learning techniques. By leveraging various classifiers and a deep neural network, we strive to achieve high accuracy in predicting the presence of Parkinson's disease based on a given dataset.

## Dataset
The dataset used in this project is `parkinsons.data`, which contains features related to Parkinson's disease. The target variable indicates the presence or absence of the disease.

## Usage
1. Ensure the dataset `parkinsons.data` is in the project directory.
2. Run the Jupyter notebook `parkinson-prediction-classifiers-neuralnetwork.ipynb` to see the entire process of data preprocessing, model training, and evaluation.
3. Alternatively, you can run the script `app.py` to see the implementation in a script format.

## Libraries Used
- **Pandas**: Data manipulation and analysis.
- **Matplotlib**: Plotting and visualization.
- **NumPy**: Numerical computing.
- **Seaborn**: Statistical data visualization.
- **Scikit-learn (sklearn)**: Machine learning library for data preprocessing, model training, and evaluation.
  - **MinMaxScaler**: Feature scaling.
  - **train_test_split**: Splitting the dataset into training and testing sets.
  - **LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier**: Various classifiers.
  - **GridSearchCV, cross_val_score**: Hyperparameter tuning and cross-validation.
  - **classification_report, confusion_matrix, precision_score, recall_score, auc, roc_curve, accuracy_score, f1_score**: Model evaluation metrics.
  - **PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay**: Plotting evaluation metrics.
- **Imbalanced-learn (imblearn)**: Handling imbalanced datasets.
  - **SMOTE**: Synthetic Minority Over-sampling Technique.
- **Termcolor**: Text coloring in the terminal for better visualization of logs.
- **TensorFlow (tensorflow)**: Deep learning framework.
  - **Sequential, Dense, Activation, Dropout**: Building and training neural networks.

## Project Structure
- `parkinson-prediction-classifiers-neuralnetwork.ipynb`: Jupyter notebook containing the entire workflow from data preprocessing to model evaluation.
- `app.py`: Script form of the project for direct execution.
- `parkinsons.data`: Dataset used for training and evaluation.

## Project Workflow

### 1. Importing Libraries
The project starts by importing essential libraries required for data manipulation, visualization, model training, and evaluation. These libraries include Pandas for data handling, NumPy for numerical computations, Matplotlib and Seaborn for data visualization, Scikit-learn for machine learning models and evaluation metrics, Imbalanced-learn for handling imbalanced datasets, Termcolor for colored terminal outputs, and TensorFlow for building and training neural networks.

### 2. Loading the Data
The dataset, named `parkinsons.data`, is loaded into a Pandas DataFrame. This dataset includes features relevant to detecting Parkinson's disease.

### 3. Exploratory Data Analysis (EDA)
Initial data exploration is conducted to understand the dataset, identify any missing values, and visualize the data distribution. This step helps in gaining insights into the data and determining any necessary preprocessing steps.

### 4. Data Preprocessing
- **Feature Scaling**: The features in the dataset are scaled to a uniform range using MinMaxScaler. This ensures that all features contribute equally to the model training.
- **Handling Imbalanced Data**: Techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are used to address any imbalance in the dataset. This helps in preventing the model from being biased towards the majority class.

### 5. Splitting the Data
The dataset is divided into training and testing sets. This allows the model to be trained on one part of the data and evaluated on another, ensuring that the model's performance is tested on unseen data.

### 6. Building Machine Learning Models
Several machine learning classifiers are built and trained on the training data. These classifiers include Logistic Regression, Support Vector Machine (SVC), Decision Tree, Random Forest, and K-Nearest Neighbors. Hyperparameter tuning and cross-validation are performed to optimize the models' performance.

### 7. Model Evaluation
The performance of each machine learning model is evaluated using various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Visualization tools are used to plot confusion matrices, precision-recall curves, and ROC curves, providing a visual understanding of the models' performance.

### 8. Building a Deep Neural Network
A deep neural network is constructed using TensorFlow's Keras API. The neural network comprises multiple layers, including input layers, hidden layers with dense (fully connected) and dropout layers, and an output layer with an appropriate activation function. The model is compiled with an optimizer, loss function, and evaluation metrics, and then trained on the training data.

### 9. Evaluating the Neural Network
The performance of the neural network is evaluated on the testing data using the same metrics as the machine learning models. The results are compared to determine the best-performing model for detecting Parkinson's disease.

# Output
![alt text](<Screenshot 2024-06-21 092639.png>) 
![alt text](<Screenshot 2024-06-21 092609.png>)