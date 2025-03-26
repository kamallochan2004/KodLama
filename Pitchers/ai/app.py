from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel

# Load model and scaler
model = joblib.load("sepsis_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

class PatientVitals(BaseModel):
    heart_rate: float
    blood_pressure: float
    temperature: float
    respiratory_rate: float
    spo2: float

@app.post("/predict")
def predict_sepsis(vitals: PatientVitals):
    # Prepare input data
    input_data = np.array([[vitals.heart_rate, vitals.blood_pressure, vitals.temperature, vitals.respiratory_rate, vitals.spo2]])
    input_scaled = scaler.transform(input_data)
    
    # Get risk percentage prediction
    risk_percentage = model.predict(input_scaled)[0]
    risk_percentage = round(float(risk_percentage), 2)  # Round to 2 decimal places

    return {"Sepsis Risk (%)": risk_percentage}
