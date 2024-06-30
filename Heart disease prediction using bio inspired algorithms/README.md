# Heart Disease Prediction App

This is a Streamlit-based web application for predicting heart disease. The app takes user inputs through the sidebar and predicts whether the person has heart disease based on these inputs.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Python 3.6+
- Pandas
- Streamlit

You can install the necessary packages using pip:
```bash
pip install pandas streamlit
```
## Running the App
- Clone the repository or download the code.
- Navigate to the directory containing the code.
- Ensure that the heart_disease_model.csv file is in the same directory as the script.
- Run the Streamlit app using the following command:

## CSV File
The app expects a CSV file named heart_disease_model.csv containing the heart disease data. Make sure this file is located in the same directory as the Python script.

## Code Overview
The main parts of the code include:

- Loading Data: The CSV file is loaded using Pandas.
- Displaying Title: The title of the app is displayed using st.title().
- Sidebar: A sidebar is added for navigation and - input widgets using st.sidebar.
- Data Display: The heart disease data is displayed using st.dataframe().
- Input Fields: Multiple input fields are created for user input.
- Prediction Button: A button is provided to trigger the prediction logic.
- Prediction Logic: Dummy prediction logic is included to demonstrate functionality. Replace this with actual prediction code.
## Inputs
The app takes the following inputs from the user:

- Age
- Sex
- Chest Pain Types
- esting Blood Pressure
- Serum Cholesterol in mg/dl
- Fasting Blood Sugar > 120 mg/dl
- Resting Electrocardiographic Results
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- ST Depression Induced by Exercise
- Slope of the Peak Exercise ST Segment
- Major Vessels Colored by Fluoroscopy
- Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)
## Output
The app predicts whether the person has heart disease and displays the result as:

- "The person is having heart disease"
- "The person does not have any heart disease"
## Error Handling
If the user inputs invalid values, the app will display an error message asking the user to enter valid input values.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Streamlit for providing the framework to build this web app.
- The dataset used in this app for prediction.