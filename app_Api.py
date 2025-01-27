from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS

# Load model and other object
model = joblib.load("/Users/sot/final_model.pk1")
col_names = joblib.load('/Users/sot/col_names.pk1')
scaler = joblib.load('/Users/sot/final_scaler.pk1')

@app.route('/loan_prediction', methods=['POST'])
def predict():
    try:
        feat_data = request.json
        if not isinstance(feat_data, dict):
            raise ValueError("Input data must be a JSON object with feature values.")

        # Convert to DataFrame
        df = pd.DataFrame([feat_data])
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=col_names, fill_value=0)  

        # Scale features and make prediction
        df_scaled = scaler.transform(df)
        predictions = model.predict(df_scaled)

        # Prepare response
        prediction_messages = [
            '(1) -> This client will not be able to pay back when given a loan' if pred == 1 
            else '(0) -> This client would pay back when given a loan' for pred in predictions
        ]

        return jsonify({'prediction': prediction_messages})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Make sure it binds to all IPs and the correct port
