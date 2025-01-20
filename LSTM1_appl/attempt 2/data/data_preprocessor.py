import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler  
from config import DataConfig

class TechnicalIndicators:
    """The Ancient Arts of Technical Analysis"""

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        ma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return upper, lower

    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        true_range = high_low.combine(high_close, max).combine(low_close, max)
        return true_range.rolling(window=window).mean()

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        df['RSI'] = TechnicalIndicators.calculate_rsi(df)
        df['MACD'], df['MACD_Signal'] = TechnicalIndicators.calculate_macd(df)
        df['BB_Upper'], df['BB_Lower'] = TechnicalIndicators.calculate_bollinger_bands(df)
        df['ATR'] = TechnicalIndicators.calculate_atr(df)
        return df

class DataPreprocessor:
    """The Data Transformation Dojo"""
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = MinMaxScaler()

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        df = TechnicalIndicators.calculate_all(df)

        required_features = set(self.config.get_active_features())
        if not required_features.issubset(df.columns):
            missing_features = required_features - set(df.columns)
            raise ValueError(f"Preprocessed data is missing required features: {missing_features}")


        # Drop NaN values
        df.dropna(inplace=True)

        # Scale selected features
        selected_features = self.config.features
        scaled_data = self.scaler.fit_transform(df[selected_features])

        # Create sequences
        X, y = self._create_sequences(scaled_data)
        return X, y, self.scaler

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.config.time_steps, len(data)):
            X.append(data[i - self.config.time_steps:i])
            y.append(data[i, :4])  # Predicting Open, High, Low, Close
        return np.array(X), np.array(y)
