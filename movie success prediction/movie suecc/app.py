import streamlit as st
import pandas as pd
import requests
import pickle
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

# Load the saved models
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# API details
api_key = "c3b87632de85fe1569bd0b15603df2cc"
api_endpoint = "https://api.themoviedb.org/3"

# Fetch genres from the API
def fetch_genres():
    response = requests.get(f"{api_endpoint}/genre/movie/list", params={"api_key": api_key})
    response.raise_for_status()  # Check for HTTP errors
    genres = response.json().get('genres', [])
    genre_dict = {genre['id']: genre['name'] for genre in genres}
    return genre_dict

# Fetch popular movies from the API
def fetch_popular_movies():
    params = {
        "api_key": api_key,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "include_video": "false",
        "page": 1
    }
    response = requests.get(f"{api_endpoint}/discover/movie", params=params)
    response.raise_for_status()  # Check for HTTP errors
    movies = response.json().get('results', [])
    return movies
def preprocess_input(data, label_encoder):
    genre_encoded = label_encoder.transform([data['genre']])[0]
    return pd.DataFrame({'vote_average': [data['rating']], 'genre_encoded': [genre_encoded]})






# Fetch genres and popular movies
genres = fetch_genres()
popular_movies = fetch_popular_movies()

# Refit the label encoder with current genres
genre_names = list(genres.values())
genre_encoder = LabelEncoder()
genre_encoder.fit(genre_names)

# Streamlit app
st.title("Movie Success Prediction")

# Display some preloaded popular movies with posters
st.header("Popular Movies")
cols = st.columns(2)
for idx, movie in enumerate(popular_movies[:6]):  # Display only the first 6 popular movies, 2 per row
    with cols[idx % 2]:
        st.subheader(movie['title'])
        st.image(f"https://image.tmdb.org/t/p/w500{movie['poster_path']}")
        st.write(f"Rating: {movie['vote_average']}")
        genre_names = [genres[genre_id] for genre_id in movie['genre_ids'] if genre_id in genres]
        st.write(f"Genre: {', '.join(genre_names)}")

# User inputs
st.header("Predict Movie Success")
title = st.text_input("Title")
genre = st.selectbox("Genre", genre_names)
rating = st.slider("Rating", 0.0, 10.0, step=0.1)

# Prepare the input data
input_data = {
    'genre': genre,
    'rating': rating
}

# Preprocess the input data
try:
    processed_data = preprocess_input(input_data, genre_encoder)
except ValueError as e:
    st.error(f"Error in preprocessing input: {e}")

# Prediction
if st.button("Predict"):
    try:
        prediction = svm_model.predict(processed_data)
        result = "Success" if prediction[0] == 1 else "Failure"
        st.write(f"The predicted outcome for the movie '{title}' is: {result}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
