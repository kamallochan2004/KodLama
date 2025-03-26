import numpy as np
import joblib
import pandas as pd
import requests
import time

# Load the model and scaler for local testing/comparison
model = joblib.load("sepsis_model.pkl")
scaler = joblib.load("scaler.pkl")

# API endpoint
API_URL = "http://localhost:8000/predict"  # Adjust if your server runs on a different port

# Function to generate a single set of random vital signs
def generate_random_vitals():
    return {
        "heart_rate": np.random.uniform(50, 150),
        "blood_pressure": np.random.uniform(70, 140),
        "temperature": np.random.uniform(36, 41),
        "respiratory_rate": np.random.uniform(10, 40),
        "spo2": np.random.uniform(70, 100)
    }

# Function to send vitals to the API
def send_to_api(vitals):
    try:
        response = requests.post(API_URL, json=vitals)
        if response.status_code == 200:
            return response.json()["Sepsis Risk (%)"]
        else:
            print(f"Error: API returned status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Generate and send data in real-time
def generate_and_send_continuous(interval=5, count=10):
    """Generate and send vital signs every interval seconds"""
    print(f"Sending {count} test cases to API every {interval} seconds...")
    
    for i in range(count):
        vitals = generate_random_vitals()
        api_risk = send_to_api(vitals)
        
        print(f"\nTest Case {i+1}:")
        print(f"  Heart Rate: {vitals['heart_rate']:.1f} bpm")
        print(f"  Blood Pressure: {vitals['blood_pressure']:.1f} mmHg")
        print(f"  Temperature: {vitals['temperature']:.1f} °C")
        print(f"  Respiratory Rate: {vitals['respiratory_rate']:.1f} breaths/min")
        print(f"  SpO2: {vitals['spo2']:.1f}%")
        
        if api_risk is not None:
            print(f"  API SEPSIS RISK: {api_risk}%")
        
        time.sleep(interval)

# Run the continuous data sender
if __name__ == "__main__":
    print("===== REALTIME SEPSIS RISK PREDICTION =====")
    generate_and_send_continuous(interval=5, count=20)  # Send 20 test cases, 5 seconds apart
    
    # Also send the high-risk test case
    print("\n===== HIGH RISK TEST CASE =====")
    high_risk = {
        "heart_rate": 135,
        "blood_pressure": 75,
        "temperature": 40.2,
        "respiratory_rate": 35,
        "spo2": 80
    }
    
    api_risk = send_to_api(high_risk)
    print(f"  Heart Rate: {high_risk['heart_rate']} bpm")
    print(f"  Blood Pressure: {high_risk['blood_pressure']} mmHg")
    print(f"  Temperature: {high_risk['temperature']} °C")
    print(f"  Respiratory Rate: {high_risk['respiratory_rate']} breaths/min")
    print(f"  SpO2: {high_risk['spo2']}%")
    
    if api_risk is not None:
        print(f"  API SEPSIS RISK: {api_risk}%")