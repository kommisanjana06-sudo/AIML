import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Load trained XGBoost model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "❌ Model file not found!\n\n"
        "Please place 'xgboost_model.pkl' in the same folder as 'app.py'."
    )
    st.stop()


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Electricity Consumption Prediction",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Electricity Consumption Prediction")

st.write(
    "Enter the details below to predict electricity consumption."
)

st.divider()


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

temperature_c = st.number_input(
    "🌡️ Temperature (°C)",
    value=25.0
)

humidity_percent = st.number_input(
    "💧 Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

occupancy_percent = st.number_input(
    "👥 Occupancy (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

hour = st.number_input(
    "🕐 Hour",
    min_value=0,
    max_value=23,
    value=12
)

day = st.number_input(
    "📅 Day",
    min_value=1,
    max_value=31,
    value=1
)

month = st.number_input(
    "📆 Month",
    min_value=1,
    max_value=12,
    value=1
)

day_of_week = st.number_input(
    "📅 Day of Week",
    min_value=0,
    max_value=6,
    value=0
)

is_weekend = st.number_input(
    "🏖️ Is Weekend",
    min_value=0,
    max_value=1,
    value=0
)

is_peak_hour = st.number_input(
    "⚡ Is Peak Hour",
    min_value=0,
    max_value=1,
    value=0
)

season = st.selectbox(
    "🌦️ Season",
    ["Winter"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Electricity Consumption"):

    input_data = pd.DataFrame({
        "temperature_c": [temperature_c],
        "humidity_percent": [humidity_percent],
        "occupancy_percent": [occupancy_percent],
        "hour": [hour],
        "day": [day],
        "month": [month],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "is_peak_hour": [is_peak_hour],
        "season_Winter": [1 if season == "Winter" else 0]
    })

    try:
        prediction = model.predict(input_data)

        st.success(
            f"⚡ Predicted Electricity Consumption: "
            f"{prediction[0]:.2f} kWh"
        )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")