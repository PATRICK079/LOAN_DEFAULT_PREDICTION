from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("/Users/sot/final_model.pk1")  
col_names = joblib.load('/Users/sot/col_names.pk1') 
scaler = joblib.load('/Users/sot/final_scaler.pk1') 

@app.route("/")
def index():
    return '<h1>This is a flask api</h1>'

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
    model = joblib.load("/Users/sot/final_model.pk1")  
    col_names = joblib.load('/Users/sot/col_names.pk1') 
    scaler = joblib.load('/Users/sot/final_scaler.pk1')                 

    app.run(debug=True)

