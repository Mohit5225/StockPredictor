import pandas as pd
import numpy as np
from typing import Tuple, Dict
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
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        true_range = np.maximum.reduce([high_low, high_close, low_close])
        return pd.Series(true_range).rolling(window=window).mean()
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        df['RSI'] = TechnicalIndicators.calculate_rsi(df)
        df['MACD'], df['MACD_Signal'] = TechnicalIndicators.calculate_macd(df)
        df['ATR'] = TechnicalIndicators.calculate_atr(df)
        return df


class DataPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
    
    def _process_single_stock(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        return stock_data
    
    def preprocess_multiple(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, np.ndarray], np.ndarray, MinMaxScaler, MinMaxScaler]:
        """Process multiple stocks together"""
        combined_features = []
        combined_targets = []
        stock_ids = []
        
        feature_names = self.config.get_active_features
        
        for symbol, df in stock_data.items():
            # Process each stock
            df = self._process_single_stock(df)
            df = df.dropna() 
            
            # Get stock identifier
            stock_id = self.config.stock_identifier_mapping[symbol]
            stock_ids.extend([stock_id] * len(df))
            
            # Store features and targets
            features = df[feature_names].values
            targets = df[['High', 'Low', 'Close']].values
            
            combined_features.append(features)
            combined_targets.append(targets)
        
        # Combine all data
        X_np = np.concatenate(combined_features, axis=0)
        y = np.concatenate(combined_targets, axis=0)
        stock_ids = np.array(stock_ids)

        X = pd.DataFrame(X_np, columns=feature_names)

        if np.any(np.isnan(X.values)) or np.any(np.isinf(X.values)):
            raise ValueError("NaN or Inf detected in features before scaling.")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("NaN or Inf detected in targets before scaling.")

        # Fit scalers
        self.feature_scaler.fit(X)
        self.target_scaler.fit(y)
        # Scale features and targets
        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y)
        X_scaled = np.clip(X_scaled, 0, 1)
        y_scaled = np.clip(y_scaled, 0, 1)
        print("X_scaled min:", X_scaled.min(), "max:", X_scaled.max())
        print("y_scaled min:", y_scaled.min(), "max:", y_scaled.max())
 
        # Create sequences with stock IDs
        return self._create_sequences_with_ids(X_scaled, y_scaled, stock_ids), self.feature_scaler, self.target_scaler

    def _create_sequences_with_ids(self, features: np.ndarray, 
                                 targets: np.ndarray, 
                                 stock_ids: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        X, y, ids = [], [], []
        
        for i in range(self.config.time_steps, len(features)):
            X.append(features[i - self.config.time_steps:i])
            y.append(targets[i])
            ids.append(stock_ids[i])
            
        return {
            'price_input': np.array(X),
            'stock_input': np.array(ids)
        }, np.array(y)