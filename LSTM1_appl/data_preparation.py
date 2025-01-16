import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return sma, upper_band, lower_band

# Function to download and preprocess stock data
def prepare_data(stock_symbol, start_date, end_date, time_steps=30):
    # Download stock data
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data found for {stock_symbol}.")
    
    # Feature engineering
    df['daily_return'] = df['Close'].pct_change()
    df['volatility'] = (df['High'] - df['Low']) / df['Open']
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Scale features
    features = [
        'High', 'Low', 'Open', 'Volume', 'daily_return',
        'volatility', 'ma_5', 'ma_20', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower'
    ]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Prepare sequences
    def create_sequences(data):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i - time_steps:i])
            y.append(data[i, 0])  # Predicting 'High' (first column of scaled_features)
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_features)
    
    # Split data into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Verify file existence
def verify_file_existence(symbols):
    for symbol in symbols:
        for dataset in ['X_train', 'X_test', 'y_train', 'y_test']:
            if not os.path.exists(f'{symbol}_{dataset}.npy'):
                print(f"File {symbol}_{dataset}.npy is missing!")

# Save datasets for specific stock(s)
def save_data(stock_symbols, start_date, end_date):
    for stock_symbol in stock_symbols:
        try:
            X_train, X_test, y_train, y_test, _ = prepare_data(stock_symbol, start_date, end_date)
            np.save(f'{stock_symbol}_X_train.npy', X_train)
            np.save(f'{stock_symbol}_X_test.npy', X_test)
            np.save(f'{stock_symbol}_y_train.npy', y_train)
            np.save(f'{stock_symbol}_y_test.npy', y_test)

            # Verify files saved correctly
            verify_file_existence([stock_symbol])
        except Exception as e:
            print(f"Error processing {stock_symbol}: {e}")

# Example usage
if __name__ == "__main__":
    save_data(['AAPL', 'MSFT', 'GOOG'], '2017-01-01', '2025-01-13')
