

import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Make sure templates directory exists
if not os.path.exists('templates'):
    os.makedirs('templates')

# Path to the models
model_path = '/Users/sachidhoka/Desktop/PFE/models/extra_trees_model.pkl'
preprocessor_path = '/Users/sachidhoka/Desktop/PFE/models/preprocessor.pkl'
scaler_path = '/Users/sachidhoka/Desktop/PFE/models/scaler.pkl'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is saved correctly.")

# Load the models
try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the model or preprocessors: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        form_data = request.form.to_dict()
        
        # Add required columns with default values if missing
        required_columns = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 
                          'ever_married', 'work_type', 'Residence_type', 
                          'avg_glucose_level', 'bmi', 'smoking_status']
        
        # Initialize DataFrame with correct dtypes
        input_data = pd.DataFrame({col: pd.Series(dtype='object') for col in required_columns})
        
        # Add a dummy row with the form data
        input_data.loc[0] = [None] * len(required_columns)  # Initialize with None
        input_data['id'] = 0  # Add dummy ID
        
        # Update values from form data
        for key in form_data:
            if key in required_columns:
                input_data.loc[0, key] = form_data[key]
        
        # Convert numeric columns
        numeric_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        for col in numeric_columns:
            if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Preprocessing with loaded preprocessor
        input_data_processed = preprocessor.transform(input_data)
        input_data_scaled = scaler.transform(input_data_processed)
        
        # Make prediction
        prediction = model.predict_proba(input_data_scaled)[0][1]
        stroke_risk = round(prediction * 100, 2)
        
        # Determine precautions based on risk percentage
        if stroke_risk <= 20:
            precautions = "Low risk. Maintain a healthy lifestyle."
        elif stroke_risk <= 40:
            precautions = "Moderate risk. Regular check-ups and healthy diet recommended."
        elif stroke_risk <= 60:
            precautions = "High risk. Consult a doctor and monitor health regularly."
        elif stroke_risk <= 80:
            precautions = "Very high risk. Immediate medical consultation advised."
        else:
            precautions = "Critical risk. Urgent medical attention required."
        
        return render_template('index.html', stroke_risk=stroke_risk, precautions=precautions)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5001)