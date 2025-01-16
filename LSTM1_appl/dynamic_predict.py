import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def fetch_data(stock_symbol, start_date, end_date, interval='1d'):
    return yf.download(stock_symbol, start=start_date, end=end_date, interval=interval)

def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[feature]])
    return scaler, scaled_data

def predict_future_prices(model, scaler, last_sequence, future_steps):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_steps):
        next_price_scaled = model.predict(current_sequence)
        predictions.append(next_price_scaled[0, 0])
        current_sequence = np.append(current_sequence[:, 1:, :], next_price_scaled.reshape(1, 1, 1), axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def dynamic_predict(stock_symbol, future_steps=10, time_unit='days', interval='1d', model_path=None):
    print(f"\nPredicting for {stock_symbol} ({future_steps} future {time_unit})...\n")
    
    try:
        model = load_model(model_path)
        print(f"Loaded model from {model_path}.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    try:
        df = fetch_data(stock_symbol, '2017-01-01', '2025-01-13', interval=interval)
        scaler, scaled_data = preprocess_data(df)
        last_sequence = scaled_data[-model.input_shape[1]:].reshape(1, model.input_shape[1], 1)
    except Exception as e:
        print(f"Error fetching or preprocessing data: {e}")
        return

    try:
        future_prices = predict_future_prices(model, scaler, last_sequence, future_steps)
        print(f"Predicted Future Prices for {stock_symbol}: {future_prices}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, future_steps + 1), future_prices, label='Future Predictions', color='green')
        plt.title(f"Future Predicted Prices ({future_steps} {time_unit}) for {stock_symbol}")
        plt.xlabel(f"{time_unit.capitalize()}")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Prediction failed: {e}")
