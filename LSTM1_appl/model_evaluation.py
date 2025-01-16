import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def load_data(stock_symbol):
    X_test = np.load(f'{stock_symbol}_X_test.npy')
    y_test = np.load(f'{stock_symbol}_y_test.npy')
    return X_test, y_test

def get_scaler(stock_symbol, start_date='2017-01-01', end_date='2025-01-13'):
    # Get the original data to fit the scaler
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    scaler = MinMaxScaler()
    scaler.fit(df[['High']])  # Since we're predicting High prices
    return scaler

def calculate_nrmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mean_y_true = np.mean(y_true)
    nrmse = rmse / mean_y_true
    return nrmse * 100  # Convert to percentage
 
def evaluate_model(stock_symbol):
    try:
        X_test, y_test = load_data(stock_symbol)
        model = load_model(f'{stock_symbol}_lstm_model.keras')
        scaler = get_scaler(stock_symbol)
        print(f"Loaded model for {stock_symbol}.")
    except Exception as e:
        print(f"Error loading model or data for {stock_symbol}: {e}")
        return
    
    print(f"Evaluating {stock_symbol}...")
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE): {test_loss}")
    
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    # Calculate MAPE on unscaled values
    mape = np.mean(np.abs((y_test_unscaled - y_pred_unscaled) / y_test_unscaled)) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Calculate NRMSE on unscaled values (for accuracy)
    nrmse = calculate_nrmse(y_test_unscaled, y_pred_unscaled)
    accuracy_percentage = 100 - nrmse
    print(f"Normalized RMSE (Accuracy): {accuracy_percentage:.2f}%")
    
    # Visualization with actual prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(y_pred_unscaled, label='Predicted Price', color='red', alpha=0.7)
    plt.title(f'{stock_symbol}: Actual vs. Predicted Stock Prices\nMAPE: {mape:.2f}%, Accuracy: {accuracy_percentage:.2f}%')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save unscaled predictions
    np.save(f'{stock_symbol}_y_pred_unscaled.npy', y_pred_unscaled)

def evaluate_multiple_stocks(stocks):
    for stock in stocks:
        print(f"\nEvaluating model for {stock}...")
        evaluate_model(stock)

if __name__ == "__main__":
    stocks = ['AAPL', 'GOOG', 'MSFT']
    evaluate_multiple_stocks(stocks)
