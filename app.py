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

def compress_model(pkl_path, zip_path):
    """
    Compress a pickle (.pkl) file into a ZIP archive.
    """
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pkl_path, arcname=os.path.basename(pkl_path))
    print(f"Compressed {pkl_path} into {zip_path}")

def load_model(zip_path, model_name):
    """
    Extract and load the model from a ZIP file.
    If the ZIP file does not exist, compress the .pkl file first.
    """
    # If the ZIP doesn't exist, compress the pkl file
    if not os.path.exists(zip_path):
        print(f"{zip_path} not found. Compressing the model file...")
        compress_model(model_name, zip_path)
    
    # Create an extraction folder if needed
    extract_folder = "model"
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    
    extracted_path = os.path.join(extract_folder, model_name)
    
    # Extract the model from the zip archive
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(model_name, path=extract_folder)
    
    print(f"Loading model from {extracted_path}")
    return joblib.load(extracted_path)

# Specify file names
pkl_filename = "best_fraud_detection_pipeline1.1.pkl"
zip_filename = "best_fraud_detection_pipeline1.1.zip"

# Load the ML model from the zip file (compress first if needed)
pipeline = load_model(zip_filename, pkl_filename)

def get_recommendation(probability):
    """
    Provide a recommendation based on the predicted probability.
    """
    if probability > 0.8:
        return "High risk. Immediately verify the transaction or block the card."
    elif probability > 0.5:
        return "Medium risk. Consider additional authentication or user verification."
    else:
        return "Low risk. Transaction seems normal, but continue monitoring."

@app.route('/')
def index():
    # Render the HTML template, passing the category list for the dropdown
    return render_template("index.html", categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form submission (HTML) or JSON payload (API)
        data = request.form if request.form else request.get_json()
        amt = float(data['amt'])
        city_pop = float(data['city_pop'])
        lat = float(data['lat'])
        long_val = float(data['long'])
        merch_lat = float(data['merch_lat'])
        merch_long = float(data['merch_long'])
        unix_time = float(data['unix_time'])
        category = data['category']

        # Create a DataFrame matching the model's input format
        input_data = pd.DataFrame([[amt, city_pop, lat, long_val, merch_lat, merch_long, unix_time, category]],
                                  columns=['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'unix_time', 'category'])

        # Make prediction using the loaded pipeline
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        # Determine label and recommendation
        pred_label = "Fraudulent" if probability >= 0.8 else "Non-Fraudulent"
        recommendation = get_recommendation(probability)

        # Return response as JSON
        return jsonify({
            "prediction": pred_label,
            "probability": round(probability, 2),
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
