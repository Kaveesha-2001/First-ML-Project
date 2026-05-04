import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

# --- CUSTOM CSS FOR BEAUTIFUL INTERFACE ---
st.markdown("""
    <style>
    .stApp {
        background-color: #fff0f5 !important;
    }
    h1 {
        color: #d81b60 !important; /* Deep pink color */
        text-align: center;
    }
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #ffe4e1 !important;
    }
    .stButton>button {
        background-color: #ff69b4 !important; /* Hot pink button */
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.write("# 🌸 Iris Flower Prediction App")
st.write("Welcome! This app predicts the **Iris flower type** based on your inputs.")

# --- LOADING THE SVM MODEL ---
try:
    with open('SVM.pickle', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'SVM.pickle' is in the same folder.")

# --- SIDEBAR INPUTS ---
st.sidebar.header('🌷 Input Flower Features')


def user_input_features():
    # Creating sliders for the 4 features
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

    data = {
        'sepal-length': sepal_length,
        'sepal-width': sepal_width,
        'petal-length': petal_length,
        'petal-width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Get user input
df = user_input_features()

# Display user input on the main screen
st.subheader('Selected Input Parameters')
st.write(df)

# --- PREDICTION LOGIC ---
if st.button('🌸 Predict Flower Type'):
    prediction = model.predict(df)

    st.subheader('Prediction Result')
    result = prediction[0]

    # Displaying different flower type
    if result == 'Iris-setosa':
        st.success(f"The flower is **{result}** 🌷")
        st.image("images/setosa.jpg", caption="Iris Setosa", width=300)
    elif result == 'Iris-versicolor':
        st.success(f"The flower is **{result}** 🌻")
        st.image("images/versicolor.jpg", caption="Iris Versicolor", width=300)
    else:
        st.success(f"The flower is **{result}** 🌺")
        st.image("images/virginica.jpg", caption="Iris Virginica", width=300)
