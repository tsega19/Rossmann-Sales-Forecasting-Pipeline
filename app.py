from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.get_json()

        # Example: Assuming input_data is a list of feature values for a single sample
        features = np.array(input_data['features']).reshape(1, -1)

        # Preprocess input if necessary (e.g., scaling)
        # features = scaler.transform(features)  # Uncomment if a scaler is used

        # Make predictions
        prediction = model.predict(features)

        # Format predictions as a list
        response = {'prediction': prediction.tolist()}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
