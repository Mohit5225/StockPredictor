# predictor.py
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from data.data_preprocessor import TechnicalIndicators
import yfinance as yf


import pickle


@dataclass
class PredictionConfig:
    model_path: Path
    scaler_path: Path
    time_steps: int = 30
    batch_size: int = 32

class StockPredictor:
    """The Oracle System - For when you need predictions faster than Gojo's Domain Expansion"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load your weapons like Tanjiro unsheathes his sword"""
        self.model = tf.keras.models.load_model(self.config.model_path)
        with open(self.config.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data faster than Killua's Godspeed"""
        # Add your technical indicators here
        features = TechnicalIndicators.calculate_all(df)
        
        # Scale features
        scaled_data = self.scaler.transform(features)
        
        # Create sequence
        sequence = scaled_data[-self.config.time_steps:]
        return np.expand_dims(sequence, axis=0)
    
    def predict(self, symbol: str, 
                live_data: Optional[pd.DataFrame] = None) -> Dict[str, Union[float, Dict]]:
        """Make predictions like Yagami Light plotting his next move"""
        
        if live_data is None:
            # Fetch latest data if not provided
            live_data = yf.download(symbol, 
                                  start=(datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'),
                                  end=datetime.now().strftime('%Y-%m-%d'))
        
        # Prepare data
        X = self.prepare_prediction_data(live_data)
        
        # Make prediction
        scaled_pred = self.model.predict(X, batch_size=self.config.batch_size)
        
        # Inverse transform
        dummy = np.zeros((scaled_pred.shape[0], self.scaler.scale_.shape[0]))
        dummy[:, :3] = scaled_pred
        predictions = self.scaler.inverse_transform(dummy)[:, :3]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'High': float(predictions[0, 0]),
                'Low': float(predictions[0, 1]),
                'Close': float(predictions[0, 2])
            },
            'confidence_score': self._calculate_confidence(live_data, predictions)
        }
    
    def _calculate_confidence(self, recent_data: pd.DataFrame, 
                            prediction: np.ndarray) -> float:
        """Calculate confidence like All Might measuring his remaining power"""
        # Add your confidence calculation logic here
        # This is a simplified example
        recent_volatility = recent_data['High'].std() / recent_data['High'].mean()
        confidence = max(0, min(1, 1 - recent_volatility))
        return float(confidence)