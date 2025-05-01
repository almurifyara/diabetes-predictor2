
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
model = joblib.load('stacking_model.pkl')

# Function to preprocess input
def preprocess_input(user_input):
    return np.array(user_input).reshape(1, -1)

# Streamlit interface
st.title("Diabetes Prediction")

age = st.number_input("Age", min_value=1, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking_history = st.selectbox("Smoking History", ["Never", "Former", "Current"])

glucose_level = st.number_input("Glucose Level")
blood_pressure = st.number_input("Blood Pressure")

user_input = [age, gender, smoking_history, glucose_level, blood_pressure]

preprocessed_input = preprocess_input(user_input)

if st.button("Predict"):
    prediction = model.predict(preprocessed_input)
    if prediction == 1:
        st.write("You are at risk of diabetes.")
    else:
        st.write("You are not at risk of diabetes.")
