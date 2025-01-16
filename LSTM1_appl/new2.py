import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class StockPredictor:
    def __init__(self, stock_symbol, lookback_days=30):
        self.stock_symbol = stock_symbol
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = None
        self.feature_count = 12
        
    def prepare_data(self, start_date='2017-01-01', end_date=None):
        """Prepare data for training or prediction"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\nFetching data for {self.stock_symbol} from {start_date} to {end_date}")
        df = yf.download(self.stock_symbol, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.stock_symbol}")
        
        features_df = self._calculate_features(df)
        scaled_data = self._scale_features(features_df)
        
        return self._create_sequences(scaled_data)
    
    def _calculate_features(self, df):
        """Calculate technical indicators"""
        df['daily_return'] = df['Close'].pct_change()
        df['volatility'] = (df['High'] - df['Low']) / df['Open']
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
        
        df.dropna(inplace=True)
        
        return df[[
            'High', 'Low', 'Open', 'Volume', 'daily_return',
            'volatility', 'ma_5', 'ma_20', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower'
        ]]
    
    def _scale_features(self, features_df):
        """Scale features using MinMaxScaler"""
        self.scaler = MinMaxScaler()
        return self.scaler.fit_transform(features_df)
    
    def _create_sequences(self, data):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(self.lookback_days, len(data)):
            X.append(data[i - self.lookback_days:i])
            y.append(data[i, 0])  # 0 index for 'High' price
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test (80-20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"\nData shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, epochs=100, batch_size=32):
        """Train the LSTM model"""
        print("\nPreparing training data...")
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        print("\nBuilding model...")
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_days, self.feature_count)),
            LSTM(50, return_sequences=False),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.model.save(f'{self.stock_symbol}_lstm_model.keras')
        np.save(f'{self.stock_symbol}_X_test.npy', X_test)
        np.save(f'{self.stock_symbol}_y_test.npy', y_test)
        
        return history
    
    def predict_future(self, days_ahead=5):
        """Make future predictions"""
        if self.model is None:
            try:
                self.model = load_model(f'{self.stock_symbol}_lstm_model.keras')
            except Exception as e:
                print(f"Error loading model: {e}")
                return None, None
        
        try:
            # Get recent data for prediction
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 30)
            
            df = yf.download(self.stock_symbol, start=start_date, end=end_date)
            features_df = self._calculate_features(df)
            scaled_data = self._scale_features(features_df)
            
            # Create sequence for prediction
            X = scaled_data[-self.lookback_days:].reshape(1, self.lookback_days, self.feature_count)
            
            # Make predictions
            predictions = []
            prediction_dates = []
            current_X = X
            
            for i in range(days_ahead):
                pred = self.model.predict(current_X, verbose=0)
                unscaled_pred = self.scaler.inverse_transform(
                    np.array([[pred[0][0]] + [0]*(self.feature_count-1)])
                )[0][0]
                
                predictions.append(unscaled_pred)
                next_date = end_date + timedelta(days=i+1)
                prediction_dates.append(next_date)
                
                # Update sequence for next prediction
                new_row = current_X[0][-1:].copy()
                new_row[0][0] = pred[0][0]
                current_X = np.append(current_X[0][1:], new_row, axis=0)
                current_X = current_X.reshape(1, self.lookback_days, self.feature_count)
            
            return prediction_dates, predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None, None
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.model is None:
            try:
                self.model = load_model(f'{self.stock_symbol}_lstm_model.keras')
            except Exception as e:
                print(f"Error loading model: {e}")
                return None, None, None, None
        
        try:
            X_test = np.load(f'{self.stock_symbol}_X_test.npy')
            y_test = np.load(f'{self.stock_symbol}_y_test.npy')
            
            print(f"\nEvaluating {self.stock_symbol}...")
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Loss (MSE): {test_loss}")
            
            y_pred = self.model.predict(X_test, verbose=0)
            
            # Scale back predictions
            y_test_unscaled = self.scaler.inverse_transform(
                np.array([[y] + [0]*(self.feature_count-1) for y in y_test])
            )[:, 0]
            y_pred_unscaled = self.scaler.inverse_transform(
                np.array([[y] + [0]*(self.feature_count-1) for y in y_pred])
            )[:, 0]
            
            mape = np.mean(np.abs((y_test_unscaled - y_pred_unscaled) / y_test_unscaled)) * 100
            accuracy = 100 - mape
            
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Model Accuracy: {accuracy:.2f}%")
            
            return mape, accuracy, y_test_unscaled, y_pred_unscaled
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None, None, None, None
    
    def plot_predictions(self, days_ahead=5):
        """Plot future predictions"""
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

def main():
    """Main function to run the stock prediction system"""
    available_stocks = ['AAPL', 'GOOG', 'MSFT']
    print("\nStock Prediction System")
    print("=====================")
    print("\nAvailable stocks:", available_stocks)
    
    stock_symbol = input("\nEnter stock symbol: ").upper()
    if stock_symbol not in available_stocks:
        print(f"Sorry {stock_symbol} is not available. Please choose from {available_stocks}")
        return
    
    predictor = StockPredictor(stock_symbol)
    
    while True:
        print("\nOptions:")
        print("1. Train new model")
        print("2. Evaluate existing model")
        print("3. Make future predictions")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            epochs = int(input("Enter number of epochs (default 100): ") or "100")
            predictor.train_model(epochs=epochs)
        
        elif choice == '2':
            predictor.evaluate_model()
        
        elif choice == '3':
            days = int(input("Enter number of days to predict ahead (1-30): "))
            if 1 <= days <= 30:
                predictor.plot_predictions(days)
            else:
                print("Please enter a number between 1 and 30")
        
        elif choice == '4':
            print("\nExiting...")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()