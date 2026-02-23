import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load model & scaler
# -----------------------------
model = joblib.load("knn_liver_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# App Title
# -----------------------------
st.title("🩺 Liver Disease Prediction App")
st.write("Enter patient clinical details:")

# -----------------------------
# Inputs (MATCH DATASET)
# -----------------------------
age = st.number_input("Age of the patient", 1, 100, 45)

gender = st.selectbox("Gender of the patient", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

tb = st.number_input("Total Bilirubin", 0.0, 30.0, 1.0)
db = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.5)
alp = st.number_input("Alkaline Phosphotase", 0.0, 2500.0, 200.0)
alt = st.number_input("Alamine Aminotransferase", 0.0, 2000.0, 30.0)
ast = st.number_input("Aspartate Aminotransferase", 0.0, 2000.0, 30.0)
tp = st.number_input("Total Proteins", 0.0, 10.0, 6.5)
alb = st.number_input("Albumin", 0.0, 6.0, 3.5)
ag = st.number_input("Albumin and Globulin Ratio", 0.0, 3.0, 1.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_data = np.array([[
        age,
        gender,
        tb,
        db,
        alp,
        alt,
        ast,
        tp,
        alb,
        ag
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Liver Disease")
    else:
        st.success("✅ Low Risk / No Liver Disease")

    st.write(f"Prediction Confidence: {np.max(probability)*100:.2f}%")