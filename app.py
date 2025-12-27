import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    with open("best_random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style="text-align:center;">🍔 Food Delivery Time Predictor</h1>
    <p style="text-align:center; color:gray;">
        Predict how long your food delivery will take 🚀
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ------------------ INPUT FORM ------------------
with st.form("prediction_form"):
    st.subheader("📦 Order Details")

    col1, col2 = st.columns(2)

    with col1:
        distance_km = st.number_input(
            "📍 Distance (km)", min_value=0.1, max_value=100.0, value=10.0, step=0.1
        )

        weather = st.selectbox(
            "🌤️ Weather Condition",
            ['Clear', 'Cloudy', 'Foggy', 'Rainy', 'Snowy', 'Stormy', 'Windy']
        )

        traffic_level = st.selectbox(
            "🚦 Traffic Level",
            ['Low', 'Medium', 'High', 'Jam']
        )

    with col2:
        time_of_day = st.selectbox(
            "⏰ Time of Day",
            ['Morning', 'Afternoon', 'Evening', 'Night']
        )

        vehicle_type = st.selectbox(
            "🛵 Vehicle Type",
            ['Bike', 'Car', 'Scooter']
        )

        preparation_time_min = st.number_input(
            "🍳 Preparation Time (min)", min_value=1, max_value=60, value=15
        )

    courier_experience_yrs = st.slider(
        "👨‍✈️ Courier Experience (years)", 0.0, 30.0, 2.0, 0.5
    )

    submit = st.form_submit_button("🔮 Predict Delivery Time")

# ------------------ ENCODING ------------------
def encode(value, options):
    return sorted(options).index(value)

if submit:
    input_df = pd.DataFrame([{
        "Distance_km": distance_km,
        "Weather": encode(weather, ['Clear','Cloudy','Foggy','Rainy','Snowy','Stormy','Windy']),
        "Traffic_Level": encode(traffic_level, ['Low','Medium','High','Jam']),
        "Time_of_Day": encode(time_of_day, ['Morning','Afternoon','Evening','Night']),
        "Vehicle_Type": encode(vehicle_type, ['Bike','Car','Scooter']),
        "Preparation_Time_min": preparation_time_min,
        "Courier_Experience_yrs": courier_experience_yrs
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"⏱️ **Estimated Delivery Time:** `{prediction:.2f} minutes`")
    st.balloons()

# ------------------ FOOTER ------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray;">
        Built with ❤️ using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
