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
        df['ATR'] = TechnicalIndicators.calculate_atr(df)
        return df


class DataPreprocessor:
    """The Data Transformation Dojo"""
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = MinMaxScaler()

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        # Step 1: Calculate Technical Indicators
        df = TechnicalIndicators.calculate_all(df)

        # Step 2: Verify Required Features
        required_features = set(self.config.get_active_features())
        if not required_features.issubset(df.columns):
            missing_features = required_features - set(df.columns)
            raise ValueError(f"Preprocessed data is missing required features: {missing_features}")

        # Step 3: Drop NaN values
        df.dropna(inplace=True)

        # Step 4: Scale Selected Features
        scaled_features = [f for f in self.config.base_features if 'return' not in f]  # Don't scale returns
        raw_features = [f for f in self.config.base_features if 'return' in f]

        scaled_data = self.scaler.fit_transform(df[scaled_features])
        raw_data = df[raw_features].values

        # Combine Scaled and Raw Features
        combined_data = np.hstack([raw_data, scaled_data])

        # Step 5: Create Sequences
        X, y = self._create_sequences(combined_data)
        return X, y, self.scaler

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.config.time_steps, len(data)):
            X.append(data[i - self.config.time_steps:i])
            y.append(data[i, :3])  # Predicting High, Low, Close
        return np.array(X), np.array(y)
