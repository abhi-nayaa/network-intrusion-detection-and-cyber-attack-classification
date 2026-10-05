# Import required libraries
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Create Flask application
app = Flask(__name__)

# Get the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(BASE_DIR, "models", "final_model.pkl")
model = joblib.load(model_path)

# Load the selected feature names
features_path = os.path.join(BASE_DIR, "models", "selected_features.pkl")
selected_features = joblib.load(features_path)


# Home route
@app.route("/")
def home():
    return jsonify({
        "message": "Network Intrusion Detection API is running",
        "status": "success"
    })


# Health check route
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# Prediction route
@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check whether data was provided
        if not data:
            return jsonify({
                "error": "No input data provided"
            }), 400

        # Check whether all required features are present
        missing_features = [
            feature for feature in selected_features
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Create DataFrame using the exact feature order
        input_data = pd.DataFrame(
            [[data[feature] for feature in selected_features]],
            columns=selected_features
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Find confidence
        confidence = max(probabilities) * 100

        # Create probability dictionary
        class_probabilities = {
            str(class_name): round(float(probability) * 100, 2)
            for class_name, probability
            in zip(model.classes_, probabilities)
        }

        # Return prediction result
        return jsonify({
            "prediction": str(prediction),
            "confidence": round(confidence, 2),
            "class_probabilities": class_probabilities
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Run the application
if __name__ == "__main__":
    app.run(debug=True)