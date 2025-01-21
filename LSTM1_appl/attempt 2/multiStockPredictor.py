import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
from data.data_preprocessor import TechnicalIndicators
from config import DataConfig

class MultiStockPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = MinMaxScaler()
        
    def preprocess_multiple(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Process multiple stocks simultaneously"""
        combined_features = []
        combined_targets = []
        
        for symbol, df in stock_data.items():
            # Add stock identifier
            df = df.copy()
            df['stock_id'] = self.config.stock_metadata[symbol].sector.value
            
            # Calculate technical indicators
            df = TechnicalIndicators.calculate_all(df)
            
            # Get features
            features = self._extract_features(df, symbol)
            targets = self._extract_targets(df)
            
            combined_features.append(features)
            combined_targets.append(targets)
            
        # Combine all stock data
        X = np.concatenate(combined_features, axis=0)
        y = np.concatenate(combined_targets, axis=0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        return self._create_sequences(X_scaled, y)
        
    def _extract_features(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Extract features for a single stock"""
        feature_list = self.config.get_symbol_active_features(symbol)
        return df[feature_list].values
        
    def _extract_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target variables"""
        return df[['High', 'Low', 'Close']].values
        
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(self.config.time_steps, len(features)):
            X.append(features[i-self.config.time_steps:i])
            y.append(targets[i])
        return np.array(X), np.array(y)