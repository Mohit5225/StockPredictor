import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

class StockPredictor:
    def __init__(self, stock_symbol, lookback_days=30):
        self.stock_symbol = stock_symbol
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = None
        
    def load_trained_model(self):
        try:
            self.model = load_model(f'{self.stock_symbol}_lstm_model.keras')
            return True
        except Exception as e:
            print(f"Error loading model for {self.stock_symbol}: {e}")
            return False
            
    def prepare_prediction_data(self, end_date=None):
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=self.lookback_days + 30)  # Extra days for feature calculation
        
        # Download recent data
        df = yf.download(self.stock_symbol, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data found for {self.stock_symbol}")
            
        # Calculate features (same as in your data_preparation.py)
        df['daily_return'] = df['Close'].pct_change()
        df['volatility'] = (df['High'] - df['Low']) / df['Open']
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
        
        df.dropna(inplace=True)
        
        features = [
            'High', 'Low', 'Open', 'Volume', 'daily_return',
            'volatility', 'ma_5', 'ma_20', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower'
        ]
        
        # Initialize and fit scaler with recent data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df[features])
        
        # Create sequence for prediction
        X = scaled_data[-self.lookback_days:].reshape(1, self.lookback_days, len(features))
        return X, df['High'].iloc[-1]
    
    def predict_future(self, days_ahead=5):
        if not self.load_trained_model():
            return None, None
            
        try:
            X, last_price = self.prepare_prediction_data()
            
            # Make predictions
            predictions = []
            prediction_dates = []
            current_X = X
            
            for i in range(days_ahead):
                # Predict next day
                pred = self.model.predict(current_X, verbose=0)
                unscaled_pred = self.scaler.inverse_transform(
                    np.array([[pred[0][0]] + [0]*(self.scaler.n_features_in_-1)])
                )[0][0]
                
                predictions.append(unscaled_pred)
                next_date = datetime.now() + timedelta(days=i+1)
                prediction_dates.append(next_date)
                
                # Update sequence for next prediction
                new_row = current_X[0][-1:].copy()
                new_row[0][0] = pred[0][0]  # Update only the 'High' value
                current_X = np.append(current_X[0][1:], new_row, axis=0)
                current_X = current_X.reshape(1, self.lookback_days, X.shape[2])
            
            return prediction_dates, predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None, None
    
    def evaluate_model(self):
        if not self.load_trained_model():
            return
            
        try:
            X_test = np.load(f'{self.stock_symbol}_X_test.npy')
            y_test = np.load(f'{self.stock_symbol}_y_test.npy')
            
            # Load scaler for test data
            _, last_price = self.prepare_prediction_data()
            
            print(f"\nEvaluating {self.stock_symbol}...")
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Loss (MSE): {test_loss}")
            
            y_pred = self.model.predict(X_test, verbose=0)
            
            # Inverse transform predictions and actual values
            y_test_unscaled = self.scaler.inverse_transform(
                np.array([[y] + [0]*(self.scaler.n_features_in_-1) for y in y_test])
            )[:, 0]
            y_pred_unscaled = self.scaler.inverse_transform(
                np.array([[y] + [0]*(self.scaler.n_features_in_-1) for y in y_pred])
            )[:, 0]
            
            # Calculate metrics
            mape = np.mean(np.abs((y_test_unscaled - y_pred_unscaled) / y_test_unscaled)) * 100
            nrmse = self._calculate_nrmse(y_test_unscaled, y_pred_unscaled)
            accuracy = 100 - nrmse
            
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Model Accuracy: {accuracy:.2f}%")
            
            return mape, accuracy, y_test_unscaled, y_pred_unscaled
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None, None, None, None
    
    def plot_predictions(self, days_ahead=5):
        dates, predictions = self.predict_future(days_ahead)
        if dates is None:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        df = yf.download(self.stock_symbol, start=start_date, end=end_date)
        plt.plot(df.index, df['High'], label='Historical Prices', color='blue')
        
        # Plot predictions
        plt.plot(dates, predictions, label='Predicted Prices', color='red', linestyle='--')
        
        plt.title(f'{self.stock_symbol} Stock Price Prediction\nNext {days_ahead} Days')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_bollinger_bands(data, window=20, num_std=2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        return sma, upper_band, lower_band
    
    @staticmethod
    def _calculate_nrmse(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return (rmse / np.mean(y_true)) * 100

# Example usage
if __name__ == "__main__":
    # Let user choose stock and prediction days
    available_stocks = ['AAPL', 'GOOG', 'MSFT']
    print("Available stocks:", available_stocks)
    
    stock_symbol = input("Enter stock symbol: ").upper()
    if stock_symbol not in available_stocks:
        print(f"Sorry, {stock_symbol} is not available. Please choose from {available_stocks}")
        exit()
    
    days_ahead = int(input("Enter number of days to predict ahead (1-30): "))
    if not 1 <= days_ahead <= 30:
        print("Please enter a number between 1 and 30")
        exit()
    
    predictor = StockPredictor(stock_symbol)
    
    # Evaluate model performance
    mape, accuracy, _, _ = predictor.evaluate_model()
    if mape is not None:
        print("\nMaking future predictions...")
        predictor.plot_predictions(days_ahead)