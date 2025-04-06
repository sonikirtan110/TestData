from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import zipfile
import os

app = Flask(__name__)

# Define category options for the dropdown in the UI
categories = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel"
]

# Function to extract and load the ML model from a ZIP file
def load_model(zip_path, model_name):
    # Ensure a folder exists to extract the model into
    extract_folder = "model"
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    extracted_path = os.path.join(extract_folder, model_name)
    if not os.path.exists(extracted_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(model_name, path=extract_folder)
    return joblib.load(extracted_path)

# Load the ML model (model file inside the ZIP should be named exactly as below)
pipeline = load_model("best_fraud_detection_pipeline1.1.zip", "best_fraud_detection_pipeline1.1.pkl")

# Recommendation logic based on predicted probability
def get_recommendation(probability):
    if probability > 0.8:
        return "High risk. Immediately verify the transaction or block the card."
    elif probability > 0.5:
        return "Medium risk. Consider additional authentication or user verification."
    else:
        return "Low risk. Transaction seems normal, but continue monitoring."

@app.route('/')
def index():
    # Render the UI with the category options
    return render_template("index.html", categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form submission
        data = request.form if request.form else request.get_json()
        amt = float(data['amt'])
        city_pop = float(data['city_pop'])
        lat = float(data['lat'])
        long_val = float(data['long'])
        merch_lat = float(data['merch_lat'])
        merch_long = float(data['merch_long'])
        unix_time = float(data['unix_time'])
        category = data['category']

        # Create a DataFrame that matches your model's expected input format
        input_data = pd.DataFrame([[amt, city_pop, lat, long_val, merch_lat, merch_long, unix_time, category]],
                                  columns=['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'unix_time', 'category'])
        
        # Get the prediction and probability
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]
        
        # Determine the label and recommendation
        pred_label = "Fraudulent" if probability >= 0.8 else "Non-Fraudulent"
        recommendation = get_recommendation(probability)

        # Return the result as JSON
        return jsonify({
            "prediction": pred_label,
            "probability": round(probability, 2),
            "recommendation": recommendation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
