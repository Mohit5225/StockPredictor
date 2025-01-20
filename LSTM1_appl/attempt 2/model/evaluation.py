# models/evaluation.py
from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class ModelEvaluator:
    """The Judgment Council"""
    @staticmethod
    def evaluate(model: tf.keras.Model, X_test: np.ndarray, 
                y_test: np.ndarray, scaler: MinMaxScaler) -> Dict[str, Any]:
        predictions = model.predict(X_test)
        
        # Inverse transform if needed
        if scaler:
            dummy = np.zeros((predictions.shape[0], scaler.scale_.shape[0]))
            dummy[:, :3] = predictions
            predictions = scaler.inverse_transform(dummy)[:, :3]
            
            dummy = np.zeros((y_test.shape[0], scaler.scale_.shape[0]))
            dummy[:, :3] = y_test
            y_test = scaler.inverse_transform(dummy)[:, :3]
        
        metrics = {
            'MAPE': {
                'High': mean_absolute_percentage_error(y_test[:, 0], predictions[:, 0]),
                'Low': mean_absolute_percentage_error(y_test[:, 1], predictions[:, 1]),
                'Close': mean_absolute_percentage_error(y_test[:, 2], predictions[:, 2])
            },
            'R2': {
                'High': r2_score(y_test[:, 0], predictions[:, 0]),
                'Low': r2_score(y_test[:, 1], predictions[:, 1]),
                'Close': r2_score(y_test[:, 2], predictions[:, 2])
            }
        }
        
        return metrics, predictions