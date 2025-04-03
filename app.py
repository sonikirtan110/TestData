from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import pandas as pd
import bz2
import os

app = Flask(__name__)

# --- Configure MySQL Database from Railway Environment Variables ---
MYSQL_USER = os.getenv("MYSQLUSER", "root")
MYSQL_PASSWORD = os.getenv("MYSQLPASSWORD", "1978")
MYSQL_HOST = os.getenv("MYSQLHOST", "mysql.railway.internal")
MYSQL_PORT = os.getenv("MYSQLPORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQLDATABASE", "credit")

DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:YsYbZPfbTqXRtNMmuXtwhGtbOLYfArWA@shuttle.proxy.rlwy.net:10338/railway'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define the Transaction model
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

# Load the trained ML model
with bz2.open("best_fraud_detection_pipeline1.1.pkl.bz2", "rb") as f:
    pipeline = joblib.load(f)

categories = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
              "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
              "personal_care", "shopping_net", "shopping_pos", "travel"]

def get_recommendation(probability):
    if probability > 0.8:
        return "High risk. Verify transaction."
    elif probability > 0.5:
        return "Medium risk. Consider verification."
    else:
        return "Low risk."

@app.route('/')
def index():
    return render_template("index.html", categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form if request.form else request.get_json()
        amt = float(data['amt'])
        city_pop = float(data['city_pop'])
        lat = float(data['lat'])
        long_val = float(data['long'])
        merch_lat = float(data['merch_lat'])
        merch_long = float(data['merch_long'])
        unix_time = float(data['unix_time'])
        category = data['category']

        input_data = pd.DataFrame([[amt, city_pop, lat, long_val, merch_lat, merch_long, unix_time, category]],
                                  columns=['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'unix_time', 'category'])

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        pred_label = "Fraudulent" if probability >= 0.8 else "Non-Fraudulent"
        recommendation = get_recommendation(probability)

        new_transaction = Transaction(
            amt=amt, city_pop=city_pop, lat=lat, long=long_val, merch_lat=merch_lat,
            merch_long=merch_long, unix_time=unix_time, category=category,
            prediction=pred_label, probability=round(probability, 2)
        )
        db.session.add(new_transaction)
        db.session.commit()

        return jsonify({
            "prediction": pred_label,
            "probability": round(probability, 2),
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
