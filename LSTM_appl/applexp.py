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

# Fetch data using yfinance
# Download data
df = yf.download('AAPL', start='2017-01-01', end='2025-01-13')

# Check the first few rows to verify the data
# print(df.head())

# # Check for missing values (NaNs)
# print(df.isnull().sum())

# # Verify the range of dates
# print(df.index.min(), df.index.max())

# # Check the length of the data to see if it's consistent
# print(len(df))
# Print the first few rows of the DataFrame to verify the data
# print(df.head())
# print(df.tail())
# df['z_score'] = zscore(df['Close'])

# # Identify outliers (e.g., z-scores greater than 3 or less than -3)
# # outliers = df[df['z_score'].abs() > 3]
# # print(outliers)
# print(df['z_score'].head())  # Print first few z-scores to check
# outliers = df[df['z_score'].abs() > 2]  # Use z-score > 2 for a broader range of outliers
# print(outliers)

# df['z_score'].plot(figsize=(10, 6))
# plt.title("Z-scores of AAPL Stock Prices")
# plt.show()

# Visualize the closing price history

# plt.figure(figsize=(14,6))
# plt.title('Close Price History')
# plt.plot(df['Close'])  # This will plot the 'Close' column from the DataFrame
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

# # print(df.isnull().sum())
# df_short = yf.download('AAPL', start='2017-01-01', end='2025-01-13')

# print(df_short.head())

# print(df_short.index.min(), df_short.index.max())
# print((df_short == 0).sum())  # Check if there are any 0 values, which might indicate missing data
# print(len(df_short))  # Make sure this number makes sense (should be around 252 rows for one year of daily trading data)


sector_mapping = {'AAPL': 1, 'MSFT': 1, 'TSLA': 2}
ticker_mapping = {'AAPL': 1, 'MSFT': 2, 'TSLA': 3}

df['sector_id'] = df['Ticker'].map(sector_mapping)
df['ticker_id'] = df['Ticker'].map(ticker_mapping)


df['daily_return'] = df['Close'].pct_change()
df['volatility'] = (df['High'] - df['Low']) / df['Open']
df['ma_5'] = df['Close'].rolling(window=5).mean()
df['ma_20'] = df['Close'].rolling(window=20).mean()
