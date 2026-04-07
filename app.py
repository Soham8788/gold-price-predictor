cat > app.py << 'EOF'
"""
Gold Price Prediction Web App (using scikit-learn)
"""
from flask import Flask, render_template, request, jsonify
import numpy as np
import yfinance as yf
import joblib
import os

app = Flask(__name__)

# Load model and scalers
print("="*50)
print("🚀 Starting Gold Price Predictor")
print("="*50)

try:
    model = joblib.load('models/gold_model.pkl')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Please run 'python train_simple.py' first")
    model = None
    scaler_X = None
    scaler_y = None

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
    """Predict future gold prices using iterative predictions"""
    if model is None or scaler_X is None or scaler_y is None:
        # Fallback: simple linear projection
        current = get_latest_data()[-1]
        return [float(current + i*2) for i in range(days)]
    
    try:
        # Get latest data
        latest_data = get_latest_data()
        if len(latest_data) < 30:
            latest_data = np.linspace(1800, 2000, 30)
        
        predictions = []
        current_window = latest_data[-30:].copy()
        
        for _ in range(days):
            # Scale and predict
            window_scaled = scaler_X.transform(current_window.reshape(1, -1))
            pred_scaled = model.predict(window_scaled)[0]
            pred_price = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            predictions.append(float(pred_price))
            
            # Update window (shift and add prediction)
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred_price
        
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback
        current = get_latest_data()[-1]
        return [float(current + i*2) for i in range(days)]

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
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
EOF