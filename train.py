"""
Train Gold Price Prediction Model
Run this file first to train and save the model
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 GOLD PRICE PREDICTION - TRAINING")
print("="*60)

# Step 1: Download data
print("\n📥 Downloading gold price data...")
try:
    df = yf.download("GC=F", start="2015-01-01", end="2024-12-31", progress=False)
    if df.empty:
        raise Exception("No data downloaded")
    df = df[['Close']].dropna()
    print(f"✅ Downloaded {len(df)} days of data")
    print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
except:
    print("⚠️ Using sample data for demonstration")
    dates = pd.date_range(start='2015-01-01', periods=2000, freq='D')
    prices = 1200 + np.linspace(0, 500, 2000) + np.random.randn(2000) * 20
    df = pd.DataFrame({'Close': prices}, index=dates)

# Step 2: Prepare data
print("\n🔄 Preparing data for LSTM...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create sequences
seq_length = 60
X, y = [], []

for i in range(seq_length, len(scaled_data)):
    X.append(scaled_data[i-seq_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples: {len(X_test)}")

# Step 3: Build model
print("\n🏗️ Building LSTM model...")
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Step 4: Train model
print("\n🏋️ Training model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Step 5: Save model
print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)
model.save('models/gold_model.h5')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Model saved to 'models/gold_model.h5'")
print("✅ Scaler saved to 'models/scaler.pkl'")

# Step 6: Evaluate
print("\n📊 Model Evaluation:")
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
print(f"Training Loss: {train_loss:.6f}")
print(f"Validation Loss: {val_loss:.6f}")

# Test prediction
last_60 = scaled_data[-60:].reshape(1, 60, 1)
pred_scaled = model.predict(last_60, verbose=0)
pred_price = scaler.inverse_transform(pred_scaled)[0][0]
actual_price = df['Close'].iloc[-1]

print(f"\n🎯 Test Prediction:")
print(f"Actual Price: ${actual_price:.2f}")
print(f"Predicted Price: ${pred_price:.2f}")
print(f"Accuracy: {(1 - abs(actual_price - pred_price)/actual_price)*100:.2f}%")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE! Run 'python app.py' to start the web app")
print("="*60)