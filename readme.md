# Movie Success Prediction App

This is a Streamlit web application that predicts the success of a movie based on its genre and rating. It uses a pre-trained SVM model for the prediction.

## Features
- Display popular movies with posters fetched from The Movie Database (TMDb) API.
- Allows users to input a movie title, select a genre, and set a rating.
- Predicts whether the movie will be successful based on the provided inputs.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Requests
- Scikit-learn
- Pickle

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/movie-success-prediction.git
    cd movie-success-prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have the SVM model file `svm_model.pkl` in the project directory. If not, train a model and save it using Pickle.

## Running the App

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to the provided URL (usually `http://localhost:8501`).

## API Key

The app uses The Movie Database (TMDb) API to fetch movie genres and popular movies. You need to have an API key from TMDb. 

Replace `c3b87632de85fe1569bd0b15603df2cc` in the `app.py` file with your actual TMDb API key:
```python
api_key = "c3b87632de85fe1569bd0b15603df2cc
"
