import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load data and model
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

# Precompute average prices by location
location_avg = data.groupby('location')['price'].mean().to_dict()

# Title
st.title("🏡 Bangalore House Price Predictor")

# Sidebar Input Widgets
location = st.selectbox("📍 Select Location", sorted(data['location'].unique()))
bhk = st.selectbox("🛏️ BHK", list(range(1, 11)))
bath = st.selectbox("🛁 Bathrooms", list(range(1, 6)))
sqft = st.number_input("📏 Total Square Feet", min_value=200.0, step=10.0)

# Predict Button
if st.button("Predict Price"):
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    predicted_price_lakhs = pipe.predict(input_df)[0]
    predicted_price = predicted_price_lakhs * 1e5

    # Confidence Interval (±10%)
    lower = predicted_price * 0.9
    upper = predicted_price * 1.1

    st.subheader("💰 Predicted Price")
    st.success(f"Estimated Price: ₹{np.round(predicted_price):,.0f}")
    st.info(f"Confidence Range: ₹{np.round(lower):,.0f} - ₹{np.round(upper):,.0f}")

    # Feature 1: Comparison with average price in that location
    avg_loc_price = location_avg.get(location, 0) * 1e5
    st.markdown("### 📊 Price Comparison")
    st.write(f"📍 **Average price in {location}**: ₹{np.round(avg_loc_price):,.0f}")
    diff = predicted_price - avg_loc_price
    pct_diff = (diff / avg_loc_price) * 100 if avg_loc_price else 0
    if diff > 0:
        st.warning(f"⚠️ Predicted price is **₹{np.round(diff):,.0f} higher** than average (+{pct_diff:.1f}%)")
    else:
        st.success(f"✅ Predicted price is **₹{np.abs(np.round(diff)):,.0f} lower** than average ({pct_diff:.1f}%)")

    # Feature 1: Visualize distribution of prices in selected location
    st.markdown("### 📈 Price Distribution in Selected Location")
    loc_prices = data[data['location'] == location]['price'] * 1e5
    fig, ax = plt.subplots()
    ax.hist(loc_prices, bins=15, color='skyblue', edgecolor='black')
    ax.axvline(predicted_price, color='red', linestyle='--', label='Your Prediction')
    ax.set_title(f"Price Distribution in {location}")
    ax.set_xlabel("Price (in ₹)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
