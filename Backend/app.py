import os
from flask import Flask, flash, json, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sintomas = request.json['sintomas']
        result = realizePrediction(sintomas[0]['isSelected'], sintomas[1]['isSelected'], sintomas[2]['isSelected'],
                                   sintomas[3]['isSelected'], sintomas[4]['isSelected'], sintomas[5]['isSelected'], sintomas[6]['isSelected'],
                                   sintomas[7]['isSelected'], sintomas[8]['isSelected'], sintomas[9]['isSelected'], sintomas[10]['isSelected'],
                                   sintomas[11]['isSelected'], sintomas[12]['isSelected'], sintomas[13]['isSelected'])
        return jsonify(
            response="Successfully",
            prediction=json.dumps(str(result[0]))
        )


def realizePrediction(tos, dolorcabeza, congnasal, difresp, dolorgarg, fiebre, diarrea, nauseas, nooftgust, dolorab, dolorart, dolormus, dolorpech, otros):
    print(tos, dolorcabeza, congnasal, difresp, dolorgarg, fiebre, diarrea,
          nauseas, nooftgust, dolorab, dolorart, dolormus, dolorpech, otros)
    model = load_model('CovidPredictorCNN.h5')
    predictions = model.predict([[tos, dolorcabeza, congnasal, difresp, dolorgarg, fiebre, diarrea,
                                  nauseas, nooftgust, dolorab, dolorart, dolormus, dolorpech, otros]])
    return predictions[0]
