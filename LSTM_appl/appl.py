import yfinance as yf
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Download stock data
df = yf.download('AAPL', start='2017-01-01', end='2025-01-13')

# Feature engineering
df['daily_return'] = df['Close'].pct_change()
df['volatility'] = (df['High'] - df['Low']) / df['Open']
df['ma_5'] = df['Close'].rolling(window=5).mean()
df['ma_20'] = df['Close'].rolling(window=20).mean()
df.dropna(inplace=True)

# Scale features
features = ['High', 'Low', 'Open', 'Volume', 'daily_return', 'volatility', 'ma_5', 'ma_20']
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

# Prepare sequences
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i, 0])  # Predicting 'High' (first column of scaled_features)
    return np.array(X), np.array(y)

time_steps = 30
X, y = create_sequences(scaled_features, time_steps)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Save datasets
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(1)  # Predict single value
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate and save the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Save model in the newer `.keras` format
model.save('aapl_lstm_model.keras')
