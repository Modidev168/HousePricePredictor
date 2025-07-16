import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load data and model
data = pd.read_csv('Cleaned_data.csv')

# App Title
st.title("🏡 Bangalore House Price Predictor")

# Input Widgets
location = st.selectbox("📍 Select Location", sorted(data['location'].unique()))
bhk = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bath = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4, 5])
sqft = st.number_input("📏 Total Square Feet", min_value=200.0, step=10.0)

# Prediction Button
if st.button("Predict Price"):
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_df)[0] * 1e5  # model output is in lakhs

    st.success(f"💰 Estimated Price: ₹{np.round(prediction):,.0f}")
