import streamlit as st
import numpy as np

st.set_page_config(page_title="Chronic Heart Failure Detection")

st.title("Chronic Heart Failure Detection Using RFAM")
st.write("Enter patient details for prediction")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=45)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=30,
    max_value=200,
    value=80
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

oxygen_level = st.number_input(
    "Oxygen Level",
    min_value=50,
    max_value=100,
    value=95
)

# Prediction button
if st.button("Predict"):

    # Simple demo prediction logic
    # Replace with ML model later

    if heart_rate > 100 or oxygen_level < 90:
        st.error("High Risk of Chronic Heart Failure")
    else:
        st.success("Low Risk of Chronic Heart Failure")

    st.write("Prediction Completed Successfully")
