from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

@app.route('/loan_prediction', methods=['POST'])
def predict():
    try:
        feat_data = request.json
        df = pd.DataFrame(feat_data)
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=col_names, fill_value=0)  
        
        df = scaler.transform(df)
        predictions = model.predict(df)
        prediction_messages = ['(1) -> This client will not be able to pay back when given a loan' if pred == 1 
        else '(0) -> This client would pay back when given a loan' for pred in predictions]

        return jsonify({'prediction': prediction_messages})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    model_path = os.getenv('MODEL_PATH')
    col_names_path = os.getenv('COL_NAMES_PATH')
    scaler_path = os.getenv('SCALER_PATH')

    model = joblib.load(model_path)  
    col_names = joblib.load(col_names_path) 
    scaler = joblib.load(scaler_path)                 

    app.run(debug=True)
