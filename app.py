import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval Prediction App")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", value=5000)
coapplicant_income = st.number_input("Coapplicant Income", value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", value=100)
loan_amount_term = st.number_input("Loan Amount Term (in days)", value=360)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
self_employed = 1 if self_employed == "Yes" else 0
education = 0 if education == "Graduate" else 1
dependents = 3 if dependents == "3+" else int(dependents)
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Prepare input
input_data = np.array([[gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount,
                        loan_amount_term, credit_history, property_area]])

# Scale numerical features
input_data[:, [5, 6, 7, 8]] = scaler.transform(input_data[:, [5, 6, 7, 8]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    st.subheader(f"Loan Application Status: {result}")
