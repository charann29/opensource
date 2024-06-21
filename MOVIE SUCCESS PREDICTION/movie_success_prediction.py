import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = 'IMDB-Movie-Data.csv'  # Ensure this file is in the same directory as your script
df = pd.read_csv(file_path)

# Clean and preprocess data
df_cleaned = df.dropna()

# Encode categorical variables
genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()

df_cleaned['Genre'] = genre_encoder.fit_transform(df_cleaned['Genre'])
df_cleaned['Director'] = director_encoder.fit_transform(df_cleaned['Director'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)', 'Metascore']
df_cleaned[numerical_features] = scaler.fit_transform(df_cleaned[numerical_features])

# Define features and target variable
features = ['Genre', 'Director', 'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)', 'Metascore']
X = df_cleaned[features]
y = df_cleaned['Success']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize machine learning models
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Streamlit app configuration
st.set_page_config(page_title='Movie Success Rate Prediction', layout='wide')

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        color: #008080;
        text-align: center;
        padding-bottom: 20px;
        text-shadow: 2px 2px #888888;
    }
    .button-wrapper {
        text-align: center;
        margin-top: 30px;
    }
    body {
        background-color: #f0f0f0;
    }
    .welcome-message {
        font-size: 18px;
        color: #333;
        text-align: center;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="title">Movie Success Rate Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="welcome-message">Welcome to the Movie Success Rate Prediction app. Enter details of your movie to predict its success rate.</p>', unsafe_allow_html=True)

# Input fields for movie details
col1, col2 = st.columns(2)
with col1:
    movie_name = st.text_input('Movie Name')
    genre = st.selectbox('Genre', df['Genre'].unique())
    director = st.selectbox('Director', df['Director'].unique())
with col2:
    runtime = st.number_input('Runtime (Minutes)', min_value=0, max_value=int(df['Runtime (Minutes)'].max()))
    rating = st.number_input('Rating', min_value=0.0, max_value=10.0)
    votes = st.number_input('Votes', min_value=0, max_value=int(df['Votes'].max()))
    revenue = st.number_input('Revenue (Millions)', min_value=0.0, max_value=float(df['Revenue (Millions)'].max()))
    metascore = st.number_input('Metascore', min_value=0, max_value=100)

# Prediction button
button_clicked = st.button('Predict Success Rate')

if button_clicked:
    with st.spinner('Predicting...'):
        # Prepare input data
        input_data = pd.DataFrame({
            'Genre': [genre],
            'Director': [director],
            'Runtime (Minutes)': [runtime],
            'Rating': [rating],
            'Votes': [votes],
            'Revenue (Millions)': [revenue],
            'Metascore': [metascore]
        })

        # Encode categorical variables and scale numerical features
        input_data['Genre'] = genre_encoder.transform(input_data['Genre'])
        input_data['Director'] = director_encoder.transform(input_data['Director'])
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Make predictions using trained models
        nb_pred = nb_model.predict_proba(input_data)[0][1]
        lr_pred = lr_model.predict_proba(input_data)[0][1]
        svm_pred = svm_model.predict_proba(input_data)[0][1]

        # Display prediction results
        st.subheader(f"Prediction Results for {movie_name}")
        st.write(f"Naive Bayes Prediction: {nb_pred * 100:.2f}%")
        st.write(f"Logistic Regression Prediction: {lr_pred * 100:.2f}%")
        st.write(f"SVM Prediction: {svm_pred * 100:.2f}%")

# Footer and acknowledgements
st.markdown("---")
st.markdown("Developed by CMRCET-CSM/AIM-SECTION-1(BATCH-16)")
