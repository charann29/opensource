import streamlit as st
import pickle
import numpy as np

# Loading the model
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return prediction

def main():
    # Frontend elements of the web page
    st.title("Iris Flower Prediction")
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Creating input fields for user input
    sepal_length = st.text_input("Sepal Length", "Type Here")
    sepal_width = st.text_input("Sepal Width", "Type Here")
    petal_length = st.text_input("Petal Length", "Type Here")
    petal_width = st.text_input("Petal Width", "Type Here")

    # Converting user input into float and making prediction
    if st.button("Predict"):
        try:
            sepal_length = float(sepal_length)
            sepal_width = float(sepal_width)
            petal_length = float(petal_length)
            petal_width = float(petal_width)
            prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
            st.success(f"The predicted species is: {prediction}")
        except ValueError:
            st.error("Please enter valid numeric values.")

if __name__ == '__main__':
    main()
