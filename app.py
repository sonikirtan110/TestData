from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load("best_fraud_detection_model1.pkl")

# Map category dropdown to numerical values (if necessary)
categories = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel"
]

@app.route('/')
def index():
    return render_template("index.html", categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form or JSON data
        data = request.form if request.form else request.get_json()

        # Extract features
        amt = float(data['amt'])
        city_pop = float(data['city_pop'])
        lat = float(data['lat'])
        long = float(data['long'])
        merch_lat = float(data['merch_lat'])
        merch_long = float(data['merch_long'])
        unix_time = float(data['unix_time'])
        category = data['category']

        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([[amt, city_pop, lat, long, merch_lat, merch_long, unix_time, category]],
                                  columns=['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'unix_time', 'category'])

        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        # Return response
        return jsonify({
            "prediction": "Fraudulent" if prediction >= 0.50 else "Non-Fraudulent",
            "probability": round(probability, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
