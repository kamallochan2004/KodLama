import pandas as pd
import numpy as np

# Number of samples
num_samples = 500

# Function to calculate a risk percentage
def calculate_risk(heart_rate, blood_pressure, temperature, respiratory_rate, spo2):
    # Define weightage for each factor (modify these values as needed)
    weight_hr = 0.25   # High impact
    weight_bp = 0.20   # Medium impact
    weight_temp = 0.20 # Medium impact
    weight_rr = 0.20   # Medium impact
    weight_spo2 = -0.15  # Negative impact (higher SpO2 means lower risk)

    # Normalize each factor between 0 and 1
    hr_score = (heart_rate - 50) / (150 - 50)  # Assuming HR range 50-150
    bp_score = (blood_pressure - 70) / (140 - 70)  # BP range 70-140
    temp_score = (temperature - 36) / (41 - 36)  # Temp range 36-41°C
    rr_score = (respiratory_rate - 10) / (40 - 10)  # RR range 10-40
    spo2_score = (100 - spo2) / (100 - 70)  # SpO2 range 70-100 (inverted)

    # Calculate weighted risk score
    risk_score = (
        weight_hr * hr_score +
        weight_bp * bp_score +
        weight_temp * temp_score +
        weight_rr * rr_score +
        weight_spo2 * spo2_score
    )

    # Scale to 0-100% risk
    risk_percentage = np.clip(risk_score * 100, 0, 100)
    return risk_percentage

# Generate dataset
data = []
for _ in range(num_samples):
    heart_rate = np.random.uniform(50, 150)
    blood_pressure = np.random.uniform(70, 140)
    temperature = np.random.uniform(36, 41)
    respiratory_rate = np.random.uniform(10, 40)
    spo2 = np.random.uniform(70, 100)

    risk = calculate_risk(heart_rate, blood_pressure, temperature, respiratory_rate, spo2)
    data.append([heart_rate, blood_pressure, temperature, respiratory_rate, spo2, risk])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Heart Rate", "Blood Pressure", "Temperature", "Respiratory Rate", "SpO2", "Risk Percentage"])

# Save dataset
df.to_csv("sepsis_data.csv", index=False)
print("✅ Dataset generated and saved as 'sepsis_data.csv'")
