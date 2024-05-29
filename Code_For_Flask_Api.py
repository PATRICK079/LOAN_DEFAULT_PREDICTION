from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/loan_prediction', methods=['POST'])
def predict():
    try:
        feat_data = request.json
        df = pd.DataFrame(feat_data)
        df = pd.get_dummies(df, drop_first= True)
        df = df.reindex(columns=col_names, fill_value=0)  
         
        df = scaler.transform(df)
        prediction = list(model.predict(df))

        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}),400

if __name__ == '__main__':
    model = joblib.load("/Users/sot/final_model.pk1")  
    col_names = joblib.load('/Users/sot/col_names.pk1') 
    scaler =   joblib.load('final_scaler.pk1')                 


    app.run(debug=True)