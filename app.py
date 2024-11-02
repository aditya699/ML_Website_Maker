from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features
        features = np.array([[
            float(data['square_feet']),
            float(data['bedrooms']),
            float(data['bathrooms']),
            float(data['age'])
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return jsonify({
            'predicted_price': round(prediction, 2)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)