import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_model():
    model_path = 'saved_steps.pkl'
    try:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        st.error(f"File '{model_path}' not found. Please ensure the file exists in the correct directory.")
        return None
    except (ValueError, pickle.UnpicklingError) as e:
        st.error(f"Model file '{model_path}' is not compatible with the current scikit-learn version or corrupted. Error: {e}")
        return None

data = load_model()

if data is not None:
    regressor = data["model"]
    le_country = data["le_country"]
    le_education = data["le_education"]

    def show_predict_page():
        st.title("Software Developer Salary Prediction")

        st.write("""### We need some information to predict the salary""")

        countries = (
            "United States",
            "India",
            "United Kingdom",
            "Germany",
            "Canada",
            "Brazil",
            "France",
            "Spain",
            "Australia",
            "Netherlands",
            "Poland",
            "Italy",
            "Russian Federation",
            "Sweden",
        )

        education = (
            "Less than a Bachelors",
            "Bachelor’s degree",
            "Master’s degree",
            "Post grad",
        )

        country = st.selectbox("Country", countries)
        education = st.selectbox("Education Level", education)

        experience = st.slider("Years of Experience", 0, 50, 3)

        ok = st.button("Calculate Salary")
        if ok:
            X = np.array([[country, education, experience]])
            X[:, 0] = le_country.transform(X[:, 0])
            X[:, 1] = le_education.transform(X[:, 1])
            X = X.astype(float)

            salary = regressor.predict(X)
            st.subheader(f"The estimated salary is ${salary[0]:.2f}")
else:
    def show_predict_page():
        st.error("Model data could not be loaded. Please check the file path and try again.")
