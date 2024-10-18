import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("fraud_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app title and description
st.title("Online Payment Fraud Detection System")
st.markdown("""
This application predicts whether an online payment transaction is fraudulent based on transaction details.
Enter the transaction information below and click **Predict** to check for fraud.
""")

# Input section for transaction details
st.subheader("Enter Transaction Details")
transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"], help="Select the type of transaction.")
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f", help="Enter the transaction amount.")
oldbalanceOrg = st.number_input("Original Balance (Before Transaction)", min_value=0.0, format="%.2f", help="Enter the account balance before the transaction.")
newbalanceOrig = st.number_input("New Balance (After Transaction)", min_value=0.0, format="%.2f", help="Enter the account balance after the transaction.")

# Map transaction types to numeric values
transaction_map = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
transaction_type_num = transaction_map[transaction_type]

# Predict fraud when button is clicked
if st.button("Predict"):
    # Prepare the input features for prediction
    input_features = np.array([[transaction_type_num, amount, oldbalanceOrg, newbalanceOrig]])
    
    # Perform prediction
    prediction = model.predict(input_features)
    
    # Display the result
    if prediction[0] == "Fraud":
        st.error("⚠️ This transaction is predicted as **Fraudulent**!")
    else:
        st.success("✅ This transaction is predicted as **Not Fraudulent**.")

# Footer
st.markdown("""
---
**Note:** This prediction is based on the trained model and may not be 100% accurate. Use this information as a guide, not a decision-making tool.
""")
