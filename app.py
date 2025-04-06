from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import joblib
import zipfile

app = Flask(__name__)

# Load categories
categories = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
              "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
              "personal_care", "shopping_net", "shopping_pos", "travel"]

# Extract and Load ML model from ZIP
def load_model(zip_path, model_name):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(model_name)
    return joblib.load(model_name)

pipeline = load_model("best_fraud_detection_pipeline1.1.zip", "best_fraud_detection_pipeline1.1.pkl")

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
        data = request.form
        amt = float(data['amt'])
        city_pop = int(data['city_pop'])
        lat = float(data['lat'])
        long_val = float(data['long'])
        merch_lat = float(data['merch_lat'])
        merch_long = float(data['merch_long'])
        unix_time = int(data['unix_time'])
        category = data['category']

        input_data = pd.DataFrame([[amt, city_pop, lat, long_val, merch_lat, merch_long, unix_time, category]],
                                  columns=['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'unix_time', 'category'])

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]
        pred_label = "Fraudulent" if probability >= 0.8 else "Non-Fraudulent"
        recommendation = get_recommendation(probability)

        # Save to Excel (append mode)
        excel_file = "transactions.xlsx"
        record = {
            "amt": amt,
            "city_pop": city_pop,
            "lat": lat,
            "long": long_val,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "unix_time": unix_time,
            "category": category,
            "prediction": pred_label,
            "probability": round(probability, 2)
        }

        if os.path.exists(excel_file):
            df_existing = pd.read_excel(excel_file)
            df_new = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
        else:
            df_new = pd.DataFrame([record])

        df_new.to_excel(excel_file, index=False)

        return jsonify({
            "prediction": pred_label,
            "probability": round(probability, 2),
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
