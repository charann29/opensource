from sklearn.datasets import load_iris
import pandas as pd
import streamlit as st 
import pickle
import numpy as np

species = ['setosa', 'versicolor', 'virginica']
image = ['setosa.jpg', 'versicolor.jpg', 'virginica.jpg']
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def main():

    # Creating Sidebar for inputs
    st.sidebar.title("Inputs")
    sepal_length = st.sidebar.slider("sepal length (cm)",4.3,7.9,5.0)
    sepal_width = st.sidebar.slider("sepal width (cm)",2.0,4.4,3.6)
    petal_length = st.sidebar.slider("petal length (cm)",1.0,6.9,1.4)
    petal_width = st.sidebar.slider("petal width (cm)",0.1,2.5,0.2)

    # Getting Prediction from model
    inp = np.array([sepal_length,sepal_width,petal_length,petal_width])
    inp = np.expand_dims(inp,axis=0)
    prediction = model.predict_proba(inp)

    # Main page
    st.title("Iris Flower Classification")
    st.write("This app correctly classifies iris flower among 3 possible species")

    ## Show Results when prediction is done
    if prediction.any():
        st.write('''
        ## Results
        Following is the probability of each class
        ''')
        
        df = pd.DataFrame(prediction, index = ['result'], columns=species)
        st.dataframe(df)
        result = species[np.argmax(prediction)]
        st.write("**This flower belongs to " + result + " class**")
        st.image(image[np.argmax(prediction)])

    

if __name__ == "__main__":
    main()