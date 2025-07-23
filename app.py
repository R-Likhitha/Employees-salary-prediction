# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model & feature order
model = joblib.load("salary_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# UI
st.set_page_config(page_title="Smart Salary Predictor", layout="centered")
st.title("💼 Smart Salary Predictor")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)

occupation = st.selectbox("Occupation", [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
    'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
    'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
])

capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)

hours = st.slider("Hours per Week", 1, 100, 40)

native_country = st.selectbox("Native Country", [
    'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England', 'Cuba',
    'Jamaica', 'China', 'Puerto-Rico', 'South'
])

workclass = st.selectbox("Workclass", [
    'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
    'Local-gov', 'State-gov', 'Without-pay'
])

# On click
if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        'age': age,
        'occupation': occupation,
        'capital-gain': capital_gain,
        'hours-per-week': hours,
        'native-country': native_country,
        'workclass': workclass
    }])

    # Align columns
    input_data.columns = input_data.columns.str.strip().str.lower().str.replace(" ", "-")
    input_data = input_data[feature_columns]

    try:
        result = model.predict(input_data)[0]
        st.success(f"💰 Predicted Salary: {'>50K' if result == '>50K' else '<=50K'}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
