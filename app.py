
import pickle
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the KMeans model
try:
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    print("KMeans model loaded successfully!")
except FileNotFoundError:
    print("Error: kmeans_model.pkl not found. Please ensure it's in the same directory.")
    kmeans_model = None

@app.route('/')
def home():
    return send_file('index.html', mimetype='text/html')

@app.route('/predict', methods=['POST'])
def predict():
    if kmeans_model is None:
        return jsonify({'error': 'Model not loaded. Cannot make predictions.'}), 500

    data = request.get_json(force=True)
    if not data or 'annual_income' not in data or 'spending_score' not in data:
        return jsonify({'error': 'Invalid input. Please provide annual_income and spending_score.'}), 400

    try:
        annual_income = float(data['annual_income'])
        spending_score = float(data['spending_score'])
    except ValueError:
        return jsonify({'error': 'Invalid data types for annual_income or spending_score. Must be numbers.'}), 400

    features = np.array([[annual_income, spending_score]])
    prediction = kmeans_model.predict(features)

    return jsonify({
        'cluster': int(prediction[0]),
        'annual_income': annual_income,
        'spending_score': spending_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
