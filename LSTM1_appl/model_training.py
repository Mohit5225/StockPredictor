from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

def load_data(stock_symbol):
    X_train = np.load(f'{stock_symbol}_X_train.npy')
    y_train = np.load(f'{stock_symbol}_y_train.npy')
    X_test = np.load(f'{stock_symbol}_X_test.npy')
    y_test = np.load(f'{stock_symbol}_y_test.npy')
    return X_train, y_train, X_test, y_test

def train_model(stock_symbol):
    file_suffixes = ['_X_train.npy', '_y_train.npy', '_X_test.npy', '_y_test.npy']
    current_dir = os.getcwd()  # Get current working directory
    if all(os.path.isfile(os.path.join(current_dir, f'{stock_symbol}{suffix}')) for suffix in file_suffixes):
        X_train, y_train, X_test, y_test = load_data(stock_symbol)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(50, return_sequences=False),
            Dense(1)
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save model
        model.save(f'{stock_symbol}_lstm_model.keras')
        return model, history
    else:
        raise FileNotFoundError(f"Missing files for {stock_symbol}. Please check if the files are saved correctly.")

def train_multiple_stocks(stocks):
    for stock in stocks:
        print(f"Training model for {stock}...")
        try:
            model, history = train_model(stock)
            print(f"Model for {stock} trained successfully")
        except Exception as e:
            print(f"Error training model for {stock}: {e}")

if __name__ == "__main__":
    stocks = ['AAPL', 'GOOG', 'MSFT']
    train_multiple_stocks(stocks)
