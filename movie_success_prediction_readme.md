# Movie Success Rate Prediction

This repository contains a Streamlit web application for predicting the success rate of movies based on various features. The application uses machine learning models to provide predictions on the likelihood of a movie's success.

## Features

- **Movie Details Input**: Allows users to input movie details such as name, genre, director, runtime, rating, votes, revenue, and metascore.
- **Prediction Models**: Utilizes three different machine learning models (Naive Bayes, Logistic Regression, SVM) to predict the success rate.
- **Interactive UI**: Built with Streamlit, providing an interactive user interface for easy input and real-time predictions.

## Installation

1. **Install dependencies**:
    Ensure you have Python installed. It's recommended to use a virtual environment.

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

2. **Download the dataset**:
    Ensure you have the `IMDB-Movie-Data.csv` file in the same directory as your script. You can download it from [here](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset).

## Usage

1. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

2. **Interact with the app**:
    Open your web browser and go to `http://localhost:8501`. Enter the movie details and click on the "Predict Success Rate" button to see the predictions.

## Files

- `movie_success_prediction.py`: Main application script.
- `IMDB-Movie-Data.csv`: Dataset file containing movie details.
- `requirements.txt`: Contains the list of dependencies.

## Models

The application uses three machine learning models to predict the success rate of a movie:
- **Naive Bayes**: Suitable for binary and multiclass classification.
- **Logistic Regression**: A statistical model for binary classification.
- **Support Vector Machine (SVM)**: Effective for high-dimensional spaces and classification problems.

## Acknowledgements

Developed by CMRCET-CSM/AIM-SECTION-1 (BATCH-16).
