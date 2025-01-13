import yfinance as yf
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from scipy.stats import zscore
plt.style.use('fivethirtyeight')

 # Download stock data for AAPL
df = yf.download('AAPL', start='2017-01-01', end='2025-01-13')

# Manually add the 'Ticker' column
df['Ticker'] = 'AAPL'  # In the future, we will add other tickers dynamically

# Mapping sector and ticker IDs
sector_mapping = {'AAPL': 1, 'MSFT': 1, 'TSLA': 2}
ticker_mapping = {'AAPL': 1, 'MSFT': 2, 'TSLA': 3}

df['sector_id'] = df['Ticker'].map(sector_mapping)
df['ticker_id'] = df['Ticker'].map(ticker_mapping)

# Feature engineering
df['daily_return'] = df['Close'].pct_change()
df['volatility'] = (df['High'] - df['Low']) / df['Open']
df['ma_5'] = df['Close'].rolling(window=5).mean()
df['ma_20'] = df['Close'].rolling(window=20).mean()

# Drop NaN values created by rolling window calculations
df.dropna(inplace=True)

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale everything except 'Close'
scaled_data = scaler.fit_transform(df[['High', 'Low', 'Open', 'Volume', 
                                       'sector_id', 'ticker_id', 
                                     'daily_return', 'volatility', 
                                       'ma_5', 'ma_20']])

# Add back the 'Close' column (do not scale it)
scaled_df = pd.DataFrame(scaled_data, columns=['High', 'Low', 'Open', 'Volume', 
                                               'sector_id', 'ticker_id', 
                                               'daily_return', 'volatility', 
                                               'ma_5', 'ma_20'])

scaled_df['Close'] = df['Close']  # Add original 'Close' back

print(scaled_df.head())

print(df.head())