from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- Configuration for Model and Encoder Files ---
MODEL_FILE = 'fertilizer_model.pkl'
ENCODER_FILE = 'label_encoder.pkl'
CATEGORIES_FILE = 'categories.pkl'

# --- Global Variables for Model/Encoder/Categories ---
model = None
label_encoder = None
soil_types = []
crop_types = []

def load_assets():
    """Loads the trained ML model, label encoder, and category lists."""
    global model, label_encoder, soil_types, crop_types
    try:
        model = joblib.load(MODEL_FILE)
        label_encoder = joblib.load(ENCODER_FILE)
        categories_data = joblib.load(CATEGORIES_FILE)
        soil_types = categories_data['soil_types']
        crop_types = categories_data['crop_types']
        print(f"Assets loaded successfully. Model ready. Soil Types: {soil_types}, Crop Types: {crop_types}")
    except FileNotFoundError:
        print(f"Error: Required files ({MODEL_FILE}, {ENCODER_FILE}, {CATEGORIES_FILE}) not found.")
        print("Please run 'fertilizer_predictor.py' first to train and save the model.")
        exit()
    except Exception as e:
        print(f"An error occurred loading assets: {e}")
        exit()

# Load assets before the first request
with app.app_context():
    load_assets()

@app.route('/')
def index():
    """Renders the main prediction form page."""
    # Pass the unique categories to the HTML template for dropdowns
    return render_template('index.html', soil_types=soil_types, crop_types=crop_types)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the user interface."""
    if not model or not label_encoder:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        # 1. Extract data from the form
        data = request.form

        # 2. Create a Pandas DataFrame from the input data
        # Ensure the column order exactly matches the training data features:
        # ['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']
        input_data = pd.DataFrame({
            'Temparature': [float(data['Temparature'])],
            'Humidity': [float(data['Humidity'])],
            'Moisture': [float(data['Moisture'])],
            'Nitrogen': [int(data['Nitrogen'])],
            'Potassium': [int(data['Potassium'])],
            'Phosphorous': [int(data['Phosphorous'])],
            'Soil_Type': [data['Soil_Type']],
            'Crop_Type': [data['Crop_Type']],
        })

        # Reorder columns to ensure consistency with the training pipeline
        # Note: 'Nitrogen', 'Potassium', 'Phosphorous' are numerical features but were placed
        # after categorical in the form for better UI grouping.
        feature_order = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil_Type', 'Crop_Type']
        input_data = input_data[feature_order]


        # 3. Make prediction using the loaded pipeline
        prediction_label = model.predict(input_data)[0]

        # 4. Inverse transform the label to get the fertilizer name
        predicted_fertilizer = label_encoder.inverse_transform([prediction_label])[0]

        # 5. Return the result as JSON
        return jsonify({
            'prediction': predicted_fertilizer,
            'status': 'success'
        })

    except ValueError as e:
        # Handle cases where non-numeric input is passed for numeric fields
        return jsonify({'error': f'Invalid input format for a numeric field: {e}'}), 400
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    # To run: 'flask run' or 'python app.py'
    # Ensure you are in the directory containing app.py and the .pkl files
    app.run()
