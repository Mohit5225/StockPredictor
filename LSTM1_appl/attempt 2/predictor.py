# predictor.py
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from data.data_preprocessor import TechnicalIndicators
from config import DataConfig
import yfinance as yf
import pickle


@dataclass
class PredictionConfig:
    model_path: Path
    feature_scaler_path: Path
    target_scaler_path: Path
    time_steps: int = 30
    batch_size: int = 32

class StockPredictor:
    """The Oracle System - For when you need predictions faster than Gojo's Domain Expansion"""
    
    def __init__(self, config: PredictionConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load your weapons like Tanjiro unsheathes his sword"""
        self.model = tf.keras.models.load_model(self.config.model_path)
        with open(self.config.feature_scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        try :
            self.feature_names = list(self.feature_scaler.feature_names_in_)
        except AttributeError:
            self.feature_names = self.data_config.get_active_features
            
        with open(self.config.target_scaler_path, 'rb') as f:
            self.target_scaler = pickle.load(f)
            # --- DEBUG: print how many features each scaler expects ---
        print("🔍 feature_scaler.n_features_in_ =", 
              getattr(self.feature_scaler, 'n_features_in_', None))
        print("🔍 target_scaler.n_features_in_ =", 
              getattr(self.target_scaler, 'n_features_in_', None))
    
    def prepare_prediction_data(self, df: pd.DataFrame, symbol: str) -> Dict[str, np.ndarray]:
        """Prepare data faster than Killua's Godspeed"""
        # Add your technical indicators here
        # Make a copy to avoid SettingWithCopyWarning
        features_df = df.copy()
        features_df = TechnicalIndicators.calculate_all(features_df)
        features_df = features_df.ffill().bfill()
        # Use config's active features instead of scaler.feature_names_in_
        feature_names = self.feature_names  
        # sanity check
        missing = set(feature_names) - set(features_df.columns)
        if missing:
            raise ValueError(f"Missing feature columns after indicator calc: {missing}")

        features = features_df[feature_names]

         
        
        if len(features) < self.config.time_steps:
            raise ValueError(
                f"Not enough data after fill: {len(features)} < {self.config.time_steps}"
            )
        scaled = self.feature_scaler.transform(features.values)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
        scaled = np.clip(scaled, 0.0, 1.0)

       # Create sequence for price data
        
        seq = scaled[-self.config.time_steps:]
        price_input = np.expand_dims(seq, axis=0)


        # Get stock ID
        stock_id = self.data_config.stock_identifier_mapping.get(symbol)
        if stock_id is None:
            raise ValueError(f"Symbol {symbol} not found in stock_identifier_mapping.")
        stock_input = np.array([[stock_id]])

        return {
            'price_input': price_input,
            'stock_input': stock_input
        }
    
    def predict_from_processed_data(self, processed_input: Dict[str, np.ndarray]) -> Dict[str, Union[float, Dict]]:
        """Make predictions from already processed data"""
        # Make prediction
        scaled_pred = self.model.predict(processed_input, batch_size=self.config.batch_size)
        
        # Inverse transform using target scaler
        dummy_for_inverse = np.zeros((scaled_pred.shape[0], self.target_scaler.scale_.shape[0]))
        dummy_for_inverse[:, :3] = scaled_pred
        predictions = self.target_scaler.inverse_transform(dummy_for_inverse)[:, :3]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'High': float(predictions[0, 0]),
                'Low': float(predictions[0, 1]),
                'Close': float(predictions[0, 2])
            }
        }
    
    def predict(self, live_data: pd.DataFrame, symbol: str) -> Dict[str, Union[float, Dict]]:
        """Make predictions from preprocessed data"""
        # Prepare data
        X = self.prepare_prediction_data(live_data, symbol)
        
        # Use the core prediction function
        result = self.predict_from_processed_data(X)
        result['confidence_score'] = self._calculate_confidence(live_data, result['predictions'])
        
        return result
    
    def _calculate_confidence(self, recent_data: pd.DataFrame, 
                            prediction: Dict) -> float:
        """Calculate confidence like All Might measuring his remaining power"""
        # Add your confidence calculation logic here
        # This is a simplified example
        recent_volatility = recent_data['High'].std() / recent_data['High'].mean()
        confidence = max(0, min(1, 1 - recent_volatility))
        return float(confidence)