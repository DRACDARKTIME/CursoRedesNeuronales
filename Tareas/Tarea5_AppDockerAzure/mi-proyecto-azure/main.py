import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

model = tf.keras.models.load_model("model/modelo_casas_simple.keras")
scaler = joblib.load("model/scaler_simple.joblib")

class HouseInput(BaseModel):
    satisfaction_score: float
    customer_salary: float
    loan_tenure_years: float
    legal_cases_on_property: float
    crime_cases_reported: float
    emi_to_income_ratio: float
    property_size_sqft: float
    loan_amount: float
    price: float
    down_payment: float

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(data: HouseInput):
    df = pd.DataFrame([data.model_dump()])
    scaled = scaler.transform(df)
    prob = float(model.predict(scaled, verbose=0)[0][0])
    return {"probability": prob}