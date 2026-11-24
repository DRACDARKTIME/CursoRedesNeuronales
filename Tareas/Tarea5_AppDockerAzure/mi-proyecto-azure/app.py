import os
import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Predictor de Compra de Vivienda", page_icon="🏡", layout="centered")

st.title("Predictor de Compra de Vivienda 🏡")
st.write("Ingrese las características de la propiedad y del cliente para estimar la probabilidad de compra.")

MODEL_PATH = "model/modelo_casas_simple.keras"
SCALER_PATH = "model/scaler_simple.joblib"

@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

model, scaler = load_artifacts()

with st.form("house_form"):
    st.subheader("Datos del Cliente y Financiamiento")
    col1, col2 = st.columns(2)
    with col1:
        customer_salary = st.number_input("Salario mensual ($)", min_value=0.0, value=50000.0, step=1000.0)
        down_payment = st.number_input("Enganche / Pago inicial ($)", min_value=0.0, value=50000.0, step=1000.0)
        loan_amount = st.number_input("Monto del préstamo ($)", min_value=0.0, value=200000.0, step=5000.0)
    with col2:
        loan_tenure_years = st.number_input("Plazo del préstamo (años)", min_value=1, max_value=40, value=15)
        emi_to_income_ratio = st.number_input("Relación cuota/ingreso (EMI/Income)", min_value=0.0, max_value=5.0, value=0.30, step=0.01)
        satisfaction_score = st.slider("Nivel de satisfacción del cliente (1-10)", min_value=1, max_value=10, value=7)

    st.subheader("Datos de la Propiedad")
    col3, col4 = st.columns(2)
    with col3:
        price = st.number_input("Precio de la casa ($)", min_value=0.0, value=250000.0, step=5000.0)
        property_size_sqft = st.number_input("Tamaño (pies cuadrados)", min_value=100.0, value=1500.0, step=50.0)
    with col4:
        crime_cases_reported = st.number_input("Casos de criminalidad en la zona", min_value=0, value=0, step=1)
        legal_cases_on_property = st.number_input("Casos legales sobre la propiedad", min_value=0, value=0, step=1)

    submitted = st.form_submit_button("Calcular Probabilidad")

if submitted:
    if model is None or scaler is None:
        st.error("No se encontraron `modelo_casas_simple.keras` o `scaler_simple.joblib` en `model/`.")
    else:
        input_df = pd.DataFrame([{
            "satisfaction_score": satisfaction_score,
            "customer_salary": customer_salary,
            "loan_tenure_years": loan_tenure_years,
            "legal_cases_on_property": legal_cases_on_property,
            "crime_cases_reported": crime_cases_reported,
            "emi_to_income_ratio": emi_to_income_ratio,
            "property_size_sqft": property_size_sqft,
            "loan_amount": loan_amount,
            "price": price,
            "down_payment": down_payment
        }])

        input_scaled = scaler.transform(input_df)
        prob = float(model.predict(input_scaled, verbose=0)[0][0])

        st.divider()
        st.metric(label="Probabilidad de Compra", value=f"{prob * 100:.2f}%")
        st.progress(min(max(prob, 0.0), 1.0))

        if prob >= 0.5:
            st.success("Alta probabilidad de cierre.")
        else:
            st.warning("Baja probabilidad de cierre.")