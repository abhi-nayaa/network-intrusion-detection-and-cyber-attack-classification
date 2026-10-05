# Import required libraries
import requests
import pandas as pd
import joblib

# API URL
API_URL = "http://127.0.0.1:5000/api/predict"

# Load dataset
df = pd.read_csv("nids28.csv")

# Create the two engineered features used by the model
df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
df["total_error_rate"] = df["serror_rate"] + df["rerror_rate"]

# Load selected features
selected_features = joblib.load("models/selected_features.pkl")

# Select one record for testing
sample = df.iloc[25000]

# Prepare input data
input_data = {
    feature: float(sample[feature])
    for feature in selected_features
}

# Send prediction request to Flask API
response = requests.post(API_URL, json=input_data)

# Display API response
print("API Status Code:", response.status_code)
print("\nAPI Response:")
print(response.json())

# Display actual class from dataset for reference
print("\nActual class in dataset:", sample["attack_type"])