from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
app = Flask(__name__)

# Load model and preprocessing components
model = joblib.load("model/logistic_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
categorical_columns = joblib.load("model/categorical_columns.pkl")

# Derive numeric columns
numeric_columns = [col for col in feature_columns if col not in categorical_columns]

# Dropdown choices for form
categorical_choices = {
    'sex': ['Male', 'Female'],
    'urban_or_rural': ['Urban', 'Rural'],
    'family_history': ['Yes', 'No'],
    'physical_activity_level': ['Low', 'Medium', 'High'],
    'diet_type': ['Vegetarian', 'Non-Vegetarian', 'Vegan'],
    'processed_food_consumption': ['Low', 'Medium', 'High'],
    'stress_level': ['Low', 'Medium', 'High'],
    'work_life_balance': ['Poor', 'Average', 'Good'],
    'Smoking': ['Yes', 'No'],
    'Diabetes': ['Yes', 'No'],
    'Low_HDL_Cholesterol': ['Yes', 'No'],
    'High_LDL_Cholesterol': ['Yes', 'No'],
    'HadAngina': ['Yes', 'No'],
    'alcohol_intake': ['Low', 'Moderate', 'High']
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           numeric=numeric_columns,
                           categorical=categorical_choices,
                           result=None)  # result is now handled via JavaScript

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Prepare input
        input_data = {}
        for col in numeric_columns:
            input_data[col] = float(data.get(col, 0))
        for col in categorical_columns:
            input_data[col] = data.get(col)

        df = pd.DataFrame([input_data])

        # Transform
        df[categorical_columns] = encoder.transform(df[categorical_columns])
        df_scaled = scaler.transform(df)
        result = model.predict(df_scaled)[0]

        # Response
        prediction = "Heart Disease Detected 💔" if result == 1 else " No Heart Disease 💚"
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"prediction": f"⚠️ Error: {str(e)}"}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

