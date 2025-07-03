from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ ADD THIS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ ENABLE CORS FOR ALL ROUTES

# Load model
model = pickle.load(open("../model/model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            float(data['price']),
            int(data['promotion']),
            int(data['day']),
            int(data['month']),
            int(data['day_of_week'])
        ]])
        prediction = model.predict(features)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
