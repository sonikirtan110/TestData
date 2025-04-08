from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model with compatibility fix
try:
    pipeline = joblib.load("best_fraud_detection_pipeline.pkl")
except:
    pipeline = joblib.load("best_fraud_detection_pipeline.pkl", mmap_mode='c')


categories = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel"
]

def get_recommendation(probability):
    if probability > 0.8:
        return "High risk. Immediately verify the transaction or block the card."
    elif probability > 0.5:
        return "Medium risk. Consider additional authentication or user verification."
    else:
        return "Low risk. Transaction seems normal, but continue monitoring."

@app.route('/')
def index():
    return render_template("index.html", categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form if request.form else request.get_json()
        
        input_data = pd.DataFrame([[
            float(data['amt']),
            float(data['city_pop']),
            float(data['lat']),
            float(data['long']),
            float(data['merch_lat']),
            float(data['merch_long']),
            float(data['unix_time']),
            data['category']
        ]], columns=['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'unix_time', 'category'])

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        return jsonify({
            "prediction": "Fraudulent" if prediction == 1 else "Non-Fraudulent",
            "probability": round(probability, 2),
            "recommendation": get_recommendation(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
