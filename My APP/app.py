import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

# Set up Streamlit layout
st.set_page_config(layout="wide")

# Load data
@st.cache
def load_data():
    df = pd.read_csv("loan_data.csv")
    return df

df = load_data()

# Sidebar for data exploration and preprocessing
st.sidebar.title("Data Exploration and Preprocessing")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(df.head())

# Data preprocessing
st.sidebar.subheader("Data Preprocessing")
selected_columns = st.sidebar.multiselect("Select columns for preprocessing", df.columns)
if selected_columns:
    df = df[selected_columns]

# Encoding categorical variables if any
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Visualizations
st.sidebar.subheader("Data Visualizations")
if st.sidebar.checkbox("Show histograms and plots"):
    st.subheader("Histograms and Plots")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['fico'].loc[df['credit.policy']==1], bins=30, label='Credit.Policy=1', ax=ax, kde=True)
    sns.histplot(df['fico'].loc[df['credit.policy']==0], bins=30, label='Credit.Policy=0', ax=ax, kde=True)
    plt.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[df['not.fully.paid']==1]['fico'], bins=30, alpha=0.5, color='blue', label='not.fully.paid=1', ax=ax, kde=True)
    sns.histplot(df[df['not.fully.paid']==0]['fico'], bins=30, alpha=0.5, color='green', label='not.fully.paid=0', ax=ax, kde=True)
    plt.legend()
    st.pyplot(fig)

    st.subheader("Relationships and Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.jointplot(x='fico', y='int.rate', data=df, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(df.corr(), cmap='BuPu', annot=True, ax=ax)
    st.pyplot(fig)

# Model training and evaluation
st.sidebar.subheader("Model Training and Evaluation")
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Decision Tree Classifier
st.subheader("Decision Tree Classifier")
dt_clf = DecisionTreeClassifier()
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20]}
grid_search = GridSearchCV(dt_clf, param_grid, scoring='recall_weighted', cv=StratifiedKFold(n_splits=5), return_train_score=True)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
st.write("Best Parameters for Decision Tree Classifier:", best_params)

dt_clf = DecisionTreeClassifier(max_depth=best_params['max_depth'])
dt_clf.fit(X_train, y_train)
y_pred_test = dt_clf.predict(X_test)

st.subheader("Decision Tree Classifier Evaluation")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_test))
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred_test))
st.write("Test Accuracy Score:", accuracy_score(y_test, y_pred_test))

# Random Forest Classifier
st.subheader("Random Forest Classifier")
rf_clf = RandomForestClassifier(n_estimators=600)
rf_clf.fit(X_train, y_train)
y_pred_test_rf = rf_clf.predict(X_test)

st.subheader("Random Forest Classifier Evaluation")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_test_rf))
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred_test_rf))
st.write("Test Accuracy Score:", accuracy_score(y_test, y_pred_test_rf))

# Additional classifiers can be added similarly

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Created by BATCH-04")