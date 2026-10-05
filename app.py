# Import required libraries
from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

# Create Flask application
app = Flask(__name__)

# Get project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model_path = os.path.join(BASE_DIR, "models", "final_model.pkl")
model = joblib.load(model_path)

# Load selected features
features_path = os.path.join(BASE_DIR, "models", "selected_features.pkl")
selected_features = joblib.load(features_path)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:
            # Read the raw input values
            input_data = {}

            for feature in selected_features:

                # Engineered features are calculated below
                if feature not in ["total_bytes", "total_error_rate"]:
                    input_data[feature] = float(request.form[feature])

            # Feature engineering
            input_data["total_bytes"] = (
                input_data["src_bytes"] +
                input_data["dst_bytes"]
            )

            input_data["total_error_rate"] = (
                input_data["serror_rate"] +
                input_data["rerror_rate"]
            )

            # Arrange features in the exact order used during training
            input_data = {
                feature: input_data[feature]
                for feature in selected_features
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Make prediction
            prediction = model.predict(input_df)[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(input_df)[0]

            # Calculate confidence
            confidence = max(probabilities) * 100

            # Create probability dictionary
            class_probabilities = {
                str(class_name): round(float(probability) * 100, 2)
                for class_name, probability
                in zip(model.classes_, probabilities)
            }

            return render_template(
                "result.html",
                prediction=prediction,
                confidence=round(confidence, 2),
                probabilities=class_probabilities
            )

        except Exception as e:

            return f"Prediction Error: {str(e)}"

    return render_template(
        "predict.html",
        features=selected_features
    )

# Dashboard page
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# Model information page
@app.route("/model-info")
def model_info():
    return render_template("model_info.html")


# Run application
if __name__ == "__main__":
    app.run(debug=True)