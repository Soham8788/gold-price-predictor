"""
Simple Gold Price Predictor (No TensorFlow)
Uses scikit-learn instead - much more reliable on Windows
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 GOLD PRICE PREDICTION - SIMPLE TRAINER")
print("="*60)

# Create models directory
os.makedirs('models', exist_ok=True)

# Step 1: Download data
print("\n📥 Downloading gold price data...")
try:
    df = yf.download("GC=F", period="2y", progress=False)
    if df.empty or len(df) < 100:
        raise Exception("No data downloaded")
    prices = df['Close'].values
    print(f"✅ Downloaded {len(prices)} days of data")
    print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
except Exception as e:
    print(f"⚠️ Using simulated data: {e}")
    # Create simulated gold price data
    np.random.seed(42)
    days = 500
    base_price = 1800
    trend = np.linspace(0, 200, days)
    noise = np.random.normal(0, 15, days)
    prices = base_price + trend + noise
    print(f"✅ Created {len(prices)} days of simulated data")

# Step 2: Prepare features
print("\n🔄 Creating features...")
def create_features(data, window=30):
    """Create features for prediction"""
    X, y = [], []
    for i in range(window, len(data)):
        # Features: last 'window' days
        features = data[i-window:i]
        X.append(features)
        y.append(data[i])
    return np.array(X), np.array(y)

window_size = 30
X, y = create_features(prices, window_size)
print(f"✅ Created {len(X)} samples")

# Step 3: Scale data
print("\n📊 Scaling data...")
# Create separate scalers for X and y
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Step 4: Train model
print("\n🏗️ Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=50,  # Fewer trees for smaller file size
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled, y_scaled)

# Step 5: Save model and scalers
print("\n💾 Saving model...")
joblib.dump(model, 'models/gold_model.pkl')
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
print("✅ Model saved to 'models/gold_model.pkl'")
print("✅ Scalers saved")

# Step 6: Evaluate
print("\n📊 Model evaluation...")
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

# Calculate accuracy
errors = np.abs(y_actual - y_pred)
avg_error = np.mean(errors)
accuracy_pct = 100 - (avg_error / np.mean(y_actual) * 100)

print(f"Average prediction error: ${avg_error:.2f}")
print(f"Accuracy: {accuracy_pct:.2f}%")

# Test prediction for next day
last_30 = prices[-30:].reshape(1, -1)
last_30_scaled = scaler_X.transform(last_30)
next_pred_scaled = model.predict(last_30_scaled)[0]
next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0][0]
current_price = prices[-1]

print(f"\n🎯 Next day prediction:")
print(f"Current price: ${current_price:.2f}")
print(f"Predicted price: ${next_pred:.2f}")
print(f"Expected change: ${next_pred - current_price:.2f}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE! Model saved using scikit-learn")
print("="*60)
