from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Database Configuration ---
# Use the external Render PostgreSQL URL
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://root:2oTg4rHiQOwISIoI9Pes8BOK14XiZ4My@dpg-cvn8qj24d50c73fv33d0-a.oregon-postgres.render.com/card_2etv"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define a Transaction model to store incoming predictions
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amt = db.Column(db.Float, nullable=False)
    city_pop = db.Column(db.Float, nullable=False)
    lat = db.Column(db.Float, nullable=False)
    long = db.Column(db.Float, nullable=False)
    merch_lat = db.Column(db.Float, nullable=False)
    merch_long = db.Column(db.Float, nullable=False)
    unix_time = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)

# --- Load the trained model pipeline ---
pipeline = joblib.load("best_fraud_detection_pipeline1.1.pkl")

# --- Category Options for the HTML Dropdown ---
categories = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel"
]

def get_recommendation(probability):
    """
    A simple AI-based recommendation function based on the fraud probability.
    """
    if probability > 0.8:
        return "High risk. Immediately verify the transaction or block the card."
    elif probability > 0.5:
        return "Medium risk. Consider additional authentication or user verification."
    else:
        return "Low risk. Transaction seems normal, but continue monitoring."

@app.route('/')
def index():
    # Render the HTML UI, passing the category list for the dropdown
    return render_template("index.html", categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form (HTML) or JSON (API)
        data = request.form if request.form else request.get_json()

        # Extract features from the data
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

        # For this example, we treat the prediction as "Fraudulent" if probability >= 0.8
        pred_label = "Fraudulent" if probability >= 0.8 else "Non-Fraudulent"
        recommendation = get_recommendation(probability)

        # Save the transaction record in the database
        new_transaction = Transaction(
            amt=amt,
            city_pop=city_pop,
            lat=lat,
            long=long_val,
            merch_lat=merch_lat,
            merch_long=merch_long,
            unix_time=unix_time,
            category=category,
            prediction=pred_label,
            probability=round(probability, 2)
        )
        db.session.add(new_transaction)
        db.session.commit()

        # Return response (JSON)
        return jsonify({
            "prediction": pred_label,
            "probability": round(probability, 2),
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    app.run(debug=True)
