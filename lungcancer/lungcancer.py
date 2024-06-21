import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# Function to load dataset
def load_dataset(uploaded_file):
    dataset = pd.read_csv(uploaded_file)
    return dataset

# Function to preprocess dataset
def preprocess_dataset(dataset):
    dataset.GENDER = dataset.GENDER.map({'M': 1, 'F': 2})
    dataset.LUNG_CANCER = dataset.LUNG_CANCER.map({"YES": 2, "NO": 1})
    return dataset

# Function to split dataset
def split_dataset(dataset):
    train_set = dataset[:250]
    test_set = dataset[250:300].reset_index(drop=True)
    output = dataset[300:].reset_index(drop=True)
    return train_set, test_set, output

# Function to train the model
def train_model(x_train, y_train):
    model = DecisionTreeClassifier(random_state=4)
    model.fit(x_train, y_train)
    return model

# Function to predict and evaluate the model
def predict_and_evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, mae, accuracy

# Streamlit app
def main():
    st.title("Lung Cancer Prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        dataset = load_dataset(uploaded_file)

        # Display dataset information
        st.write("Dataset Info:")
        st.write(dataset.info())
        
        st.write("Dataset Description:")
        st.write(dataset.describe())
        
        st.write("First few rows of the dataset:")
        st.write(dataset.head())

        # Preprocess dataset
        dataset = preprocess_dataset(dataset)
        st.write("Dataset after mapping categorical values:")
        st.write(dataset.head())

        # Split dataset
        train_set, test_set, output = split_dataset(dataset)
        
        st.write("Training Set Head:")
        st.write(train_set.head())
        
        st.write("Test Set Head:")
        st.write(test_set.head())
        
        st.write("Output Set Head:")
        st.write(output.head())

        # Define the features and target
        features = ["GENDER", "AGE", "SMOKING", "ANXIETY", "CHRONIC DISEASE", "CHEST PAIN", "ALCOHOL CONSUMING", "SHORTNESS OF BREATH"]
        x_train = train_set[features]
        y_train = train_set["LUNG_CANCER"]
        x_test = test_set[features]
        y_test = test_set["LUNG_CANCER"]

        st.write("Training Features Head:")
        st.write(x_train.head())
        
        st.write("Training Target Head:")
        st.write(y_train.head())
        
        st.write("Test Features Head:")
        st.write(x_test.head())
        
        st.write("Test Target Head:")
        st.write(y_test.head())

        # Train the Decision Tree Classifier
        model = train_model(x_train, y_train)

        # Predict on the test set
        y_pred, mae, accuracy = predict_and_evaluate(model, x_test, y_test)
        st.write("Errors:", mae)
        st.write("Accuracy:", accuracy)

        # Predict on the output set
        x_output = output[features]
        y_output = output["LUNG_CANCER"]
        pred_output = model.predict(x_output)
        pred_output = pd.Series(pred_output, name="LUNG_CANCER").map({2: "YES", 1: "NO"})

        st.write("Predicted Output:")
        st.write(pred_output.head())

        # Prepare the final output dataframe
        out = pd.DataFrame({
            "age": x_output.AGE,
            "predicted_lung_cancer": pred_output
        })
        st.write("Output DataFrame:")
        st.write(out.head())

        # Map the actual values in the output set for comparison
        y_out = y_output.map({2: "YES", 1: "NO"})
        bg = pd.DataFrame({"LUNG_CANCER": y_out})
        st.write("Actual Output DataFrame:")
        st.write(bg.head())

        # Save the output dataframe to a CSV file if needed
        if st.button("Save Output to CSV"):
            out.to_csv(r"C:\Users\navad\Downloads\App\predicted_output.csv", index=False)
            st.write("Output saved to predicted_output.csv")

if __name__ == "__main__":
    main()
