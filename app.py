import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Set page config
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="centered"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #d43b3b;
        transform: scale(1.02);
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🍔 Food Delivery Time Prediction")
st.write("Predict your food delivery time based on distance, weather, and more!")

@st.cache_resource
def load_resources():
    # Load raw data to recreate encoders for missing ones
    try:
        df = pd.read_csv("Food_Delivery_Times.csv")
    except FileNotFoundError:
        st.error("Error: Food_Delivery_Times.csv not found. Cannot recreate encoders.")
        return None, None, None, None, None
    
    # Recreate encoders
    encoders = {}
    for col in ['Weather', 'Traffic_Level', 'Time_of_Day']:
        if col in df.columns:
            le = LabelEncoder()
            # Fill NaNs for fitting (same logic as notebook)
            df[col] = df[col].fillna(df[col].mode()[0])
            le.fit(df[col])
            encoders[col] = le
            
    # Load the saved encoder for Vehicle_Type (assuming it was the last one saved)
    try:
        with open('label_encoder.pkl', 'rb') as f:
            vehicle_encoder = pickle.load(f)
            encoders['Vehicle_Type'] = vehicle_encoder
    except Exception as e:
        st.error(f"Error loading label_encoder.pkl: {e}")
        # Fallback: recreate if pickle fails
        if 'Vehicle_Type' in df.columns:
             le = LabelEncoder()
             le.fit(df['Vehicle_Type'])
             encoders['Vehicle_Type'] = le
    
    # Load Model
    try:
        with open('best_random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

    return encoders, model

encoders, model = load_resources()

if encoders and model:
    with st.form("prediction_form"):
        st.subheader("Order Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
            preparation_time = st.number_input("Preparation Time (min)", min_value=1, max_value=120, value=15)
            courier_experience = st.number_input("Courier Experience (yrs)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
            
        with col2:
            weather_options = encoders['Weather'].classes_
            weather = st.selectbox("Weather", weather_options)
            
            traffic_options = encoders['Traffic_Level'].classes_
            traffic_level = st.selectbox("Traffic Level", traffic_options)
            
            time_options = encoders['Time_of_Day'].classes_
            time_of_day = st.selectbox("Time of Day", time_options)
            
            vehicle_options = encoders['Vehicle_Type'].classes_
            vehicle_type = st.selectbox("Vehicle Type", vehicle_options)
            
        submit_btn = st.form_submit_button("Predict Delivery Time")
        
    if submit_btn:
        # Prepare input
        try:
            # Encode inputs
            weather_enc = encoders['Weather'].transform([weather])[0]
            traffic_enc = encoders['Traffic_Level'].transform([traffic_level])[0]
            time_enc = encoders['Time_of_Day'].transform([time_of_day])[0]
            vehicle_enc = encoders['Vehicle_Type'].transform([vehicle_type])[0]
            
            # Feature order: ['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs']
            features_dict = {
                'Distance_km': [distance],
                'Weather': [weather_enc],
                'Traffic_Level': [traffic_enc],
                'Time_of_Day': [time_enc],
                'Vehicle_Type': [vehicle_enc],
                'Preparation_Time_min': [preparation_time],
                'Courier_Experience_yrs': [courier_experience]
            }
            features_df = pd.DataFrame(features_dict)
            
            # Predict
            prediction = model.predict(features_df)[0]
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Estimated Delivery Time</h3>
                <h1 style="font-size: 3rem;">{prediction:.0f} min</h1>
                <p>Enjoy your meal! 🍽️</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

