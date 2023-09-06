from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from flask_cors import CORS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
CORS(app)

loaded_modeld1 = load_model('Depression_1.h5')
loaded_modeld2 = load_model('Depression_2.h5')
loaded_models1 = load_model('Stress_1.h5')
loaded_models2 = load_model('Stress_2.h5')
loaded_modela1 = load_model('Anxiety_1.h5')
loaded_modela2 = load_model('Anxiety_2.h5')
loaded_modelts = load_model('Total_Score_1.h5')


def depression1(age, rmt, dep, stress, anx):
    x_in = np.array([[rmt, age, dep, stress, anx]])
    pred = loaded_modeld1.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred


def depression2(age, rmt, dep, stress, anx, dep1):
    x_in = np.array([[rmt, age, dep, stress, anx, dep1]])
    pred = loaded_modeld2.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred


def stress1(age, rmt, dep, stress, anx):
    x_in = np.array([[rmt, age, dep, stress, anx]])
    pred = loaded_models1.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred


def stress2(age, rmt, dep, stress, anx, stress1):
    x_in = np.array([[rmt, age, dep, stress, anx, stress1]])
    pred = loaded_models2.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred


def anx1(age, rmt, dep, stress, anx):
    x_in = np.array([[rmt, age, dep, stress, anx]])
    pred = loaded_modela1.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred


def anx2(age, rmt, dep, stress, anx, anx1):
    x_in = np.array([[rmt, age, dep, stress, anx, anx1]])
    pred = loaded_modela2.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred


def totalscore(age, rmt, dep, stress, anx, dep1, stress1, anx1):
    x_in = np.array([[age, rmt, dep, stress, anx, dep1, stress1, anx1]])
    pred = loaded_modelts.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred


def pred(age, rmt, dep, stress, anx):
    predicted_depression_1 = depression1(age, rmt, dep, stress, anx)
    predicted_stress_1 = stress1(age, rmt, dep, stress, anx)
    predicted_anxiety_1 = anx1(age, rmt, dep, stress, anx)
    finald = [dep, predicted_depression_1]
    finals = [stress, predicted_stress_1]
    finala = [anx, predicted_anxiety_1]
    no_of_sessions = 10
    ans = {'RXD': finald, 'RXS': finals, 'RXA': finala,
           'finald': predicted_depression_1, 'finals': predicted_stress_1, 'finala': predicted_anxiety_1, 'no_of_session': no_of_sessions}

    if predicted_depression_1 > 15:
        predicted_depression_2 = depression2(
            age, rmt, dep, stress, anx, predicted_depression_1)
        predicted_stress_2 = stress2(
            age, rmt, dep, stress, anx, predicted_stress_1)
        predicted_anxiety_2 = anx2(
            age, rmt, dep, stress, anx, predicted_anxiety_1)
        finald.append(predicted_depression_2)
        finals.append(predicted_stress_2)
        finala.append(predicted_anxiety_2)
        no_of_sessions = 20

        ans['finald'] = predicted_depression_2
        ans['finals'] = predicted_stress_2
        ans['finala'] = predicted_anxiety_2
        ans['no_of_session'] = no_of_sessions
        ans['MADRS'] = abs(totalscore(age, rmt, dep, stress, anx,
                           predicted_depression_2, predicted_stress_2, predicted_anxiety_2))
    else:
        ans['MADRS'] = abs(totalscore(age, rmt, dep, stress, anx,
                           predicted_depression_1, predicted_stress_1, predicted_anxiety_1))
    return ans


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input values from the JSON data
        age = data['age']
        rmt = data['rmt']
        dep = data['dep']
        stress = data['stress']
        anx = data['anx']

        # Call your prediction function
        result = pred(age, rmt, dep, stress, anx)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/getdata', methods=['GET'])
def predictgod():
    age = int(request.args.get('age'))
    rmt = float(request.args.get('rmt'))
    dep = int(request.args.get('dep'))
    stress = int(request.args.get('stress'))
    anx = int(request.args.get('anx'))

    result = pred(age, rmt, dep, stress, anx)

    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)
