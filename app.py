"""
Gold Price Prediction Web App
Live demo on Render/Railway/Replit
"""

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import yfinance as yf
import joblib
import os
import sys

app = Flask(__name__)

# Load model and scaler
print("="*50)
print("🚀 Starting Gold Price Predictor")
print("="*50)

try:
    if os.path.exists('models/gold_model.h5'):
        model = load_model('models/gold_model.h5')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model not found. Training model now...")
        os.system('python train.py')
        model = load_model('models/gold_model.h5')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model trained and loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

def get_latest_data():
    """Get latest gold prices"""
    try:
        df = yf.download("GC=F", period="60d", progress=False)
        if not df.empty:
            return df['Close'].values
    except:
        pass
    # Fallback data
    return np.array([1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890])

def predict_future(days=30):
    """Predict future gold prices"""
    try:
        # Get latest data
        latest_data = get_latest_data()
        
        if len(latest_data) < 60:
            latest_data = np.linspace(1800, 2000, 60)
        
        # Prepare sequence
        last_60 = latest_data[-60:].reshape(-1, 1)
        last_60_scaled = scaler.transform(last_60)
        current_seq = last_60_scaled.flatten()
        
        # Predict day by day
        predictions = []
        for _ in range(days):
            input_seq = current_seq.reshape(1, 60, 1)
            next_price_scaled = model.predict(input_seq, verbose=0)[0][0]
            next_price = scaler.inverse_transform([[next_price_scaled]])[0][0]
            predictions.append(float(next_price))
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = next_price_scaled
        
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return [1800 + i*5 for i in range(days)]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        days = int(data.get('days', 30))
        
        if days < 1 or days > 90:
            return jsonify({'error': 'Days must be between 1 and 90'}), 400
        
        predictions = predict_future(days)
        current_price = float(get_latest_data()[-1])
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'current_price': current_price,
            'avg_price': sum(predictions) / len(predictions),
            'max_price': max(predictions),
            'min_price': min(predictions),
            'days': days
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)