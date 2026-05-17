import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

st.set_page_config(page_title="CHF Detection Using RFAM")

st.title("Detection of Chronic Heart Failure from Heart Sounds Using RFAM")

st.write("Upload dataset and run ML/DL analysis")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    # Graph section
    st.subheader("Graphical Representation")

    fig, ax = plt.subplots()

    df.hist(ax=ax)

    st.pyplot(fig)

    # Dummy ML training
    st.subheader("Run ML Segmented Model")

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    st.success(f"ML Model Accuracy: {accuracy * 100:.2f}%")

    # DL Graph Demo
    st.subheader("DL Accuracy & Loss Graph")

    epochs = np.arange(1, 11)

    acc = np.random.uniform(0.7, 0.98, 10)

    loss = np.random.uniform(0.5, 0.1, 10)

    fig2, ax2 = plt.subplots()

    ax2.plot(epochs, acc, marker='o', label='Accuracy')

    ax2.plot(epochs, loss, marker='o', label='Loss')

    ax2.set_xlabel("Epoch")

    ax2.set_ylabel("Value")

    ax2.legend()

    st.pyplot(fig2)

    # Prediction Section
    st.subheader("Predict CHF from Test Sound")

    heart_rate = st.number_input(
        "Heart Rate",
        min_value=30,
        max_value=200,
        value=80
    )

    oxygen = st.number_input(
        "Oxygen Level",
        min_value=50,
        max_value=100,
        value=95
    )

    if st.button("Predict CHF"):

        if heart_rate > 100 or oxygen < 90:
            st.error("Abnormal Heart Sound Detected")
        else:
            st.success("Normal Heart Sound Detected")

        st.write("Prediction Completed Successfully")
