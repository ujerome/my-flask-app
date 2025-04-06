# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 22:48:47 2025

@author: uwany
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import africastalking

app = Flask(__name__)

# Africa's Talking API credentials
username = "uwanyagasanijerome@gmail.com"
api_key = "atsk_3edd69d7db7bb3646317de8f57f6508efd537e847c802969e7e62379cbe6511906544182"
africastalking.initialize(username, api_key)
sms = africastalking.SMS

# Load the trained model and scaler
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature order (must match model training)
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

# Encoding dictionaries (matching the training process)
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
work_map = {'Private': 1, 'Self-employed': 2, 'Govt_job': 3}
residence_map = {'Urban': 1, 'Rural': 0}
smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}

@app.route('/ussd', methods=['POST'])
def ussd():
    # Receive the USSD request
    phone_number = request.values.get('phoneNumber')
    session_id = request.values.get('sessionId')
    service_code = request.values.get('serviceCode')
    text = request.values.get('text')  # User input text

    # Split the text input into separate values for prediction
    # Example: "1|25|1|0|Yes|Private|Urban|95.8|25.0|never smoked"
    data = text.split('|')

    if len(data) == 10:  # Ensure the correct number of inputs
        gender = "Male" if data[0] == "1" else "Female"
        age = data[1]
        hypertension = data[2]
        heart_disease = data[3]
        ever_married = "Yes" if data[4] == "1" else "No"
        work_type = {1: "Private", 2: "Self-employed", 3: "Govt_job"}[int(data[5])]
        residence_type = "Urban" if data[6] == "1" else "Rural"
        avg_glucose_level = data[7]
        bmi = data[8]
        smoking_status = {0: "never smoked", 1: "formerly smoked", 2: "smokes"}[int(data[9])]

        # Map values to model input format
        model_input = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        # Convert to DataFrame
        input_data = pd.DataFrame([model_input], columns=features)

        # Scale numerical features
        input_data[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(
            input_data[['age', 'avg_glucose_level', 'bmi']]
        )

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Stroke Risk Detected" if prediction == 1 else "No Stroke Risk"

        # Respond to the user
        response = f"Result: {result}"
    else:
        response = "Invalid input. Please provide the required information in the correct format."

    # USSD response format
    return jsonify({
        "response": response
    })

if __name__ == '__main__':
    app.run(debug=True)
