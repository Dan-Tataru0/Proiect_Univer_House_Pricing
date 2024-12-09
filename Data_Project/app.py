from flask import Flask, render_template, request, jsonify
import joblib
import sklearn
import numpy as np

# Load the pre-trained model
try:
    model = joblib.load('model/gb_preds.pkl')  # Înlocuiește cu calea corectă
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

# Initialize Flask app
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('input_form.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        data = request.json
        print("Input data:", data)

        # Validarea datelor primite
        features = data.get('features')
        if not features or not isinstance(features, list):
            return jsonify({'error': 'Invalid input data. Features must be a list.'}), 400

        # Asigură-te că este un array numpy
        features = np.array(features).reshape(1, -1)

        # Realizează predicția
        prediction = model.predict(features)
        print("Prediction:", prediction)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
