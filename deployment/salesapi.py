from flask import Flask ,request, jsonify
import numpy as np 
from tensorflow.keras.models import load_model
import joblib
import pandas as pd 

    

app = Flask(__name__)


@app.route("/")
def index():
	return '<h1> FLASH APP IS RUNNING333</h1>'



if __name__ == '__main__':
	app.run()