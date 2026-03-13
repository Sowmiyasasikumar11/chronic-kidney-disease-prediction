import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model when the application starts
model = None

def load_model():
    """Load the pre-trained model"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'ckd_nb_model.pkl')
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Load the model when the app starts
load_model()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Prepare features for prediction
        features = [
            float(data['age']),
            float(data['bp']),
            float(data['sg']),
            float(data['al']),
            float(data['su']),
            int(data['rbc']),
            int(data['pc']),
            int(data['pcc']),
            int(data['ba'])
        ]
        
        # Make prediction
        prediction = model.predict([features])
        
        # Return prediction
        return jsonify({
            'status': 'success',
            'prediction': int(prediction[0])
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Move HTML file to templates directory
    import shutil
    if os.path.exists('index.html'):
        shutil.move('index.html', 'templates/index.html')
    
    # Run the app
    app.run(debug=True, port=5000)
