from typing import Dict, Any
import tensorflow as tf
import numpy as np
from config import ModelConfig

class ModelTrainer:
    """The Training Grounds"""
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def train(self, model: tf.keras.Model, 
              X_train: Dict[str, np.ndarray], y_train: np.ndarray,
              X_val: Dict[str, np.ndarray], y_val: np.ndarray) -> Dict[str, Any]:
        
        # Ensuring data is properly preprocessed
        self._check_preprocessed_data(X_train, X_val)
        
        # Callbacks for early stopping and learning rate reduction
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.patience
            )
        ]
        
        # Model training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val), 
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log training and validation loss
        train_loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])
        
        # Return or log metrics
        return history.history, {"train_loss": train_loss, "val_loss": val_loss}

    def _check_preprocessed_data(self, X_train: Dict[str, np.ndarray], X_val: Dict[str, np.ndarray]) -> None:
        """Ensure data is properly scaled or normalized"""
        for key in X_train:
            
            print(f"{key} train min/max:", X_train[key].min(), X_train[key].max())
            print(f"{key} val min/max:", X_val[key].min(), X_val[key].max())
            if np.any(np.isnan(X_train[key])) or np.any(np.isnan(X_val[key])):
                raise ValueError("Training and validation data contain NaN values. Please ensure data is preprocessed correctly.")
            if key != 'stock_input':
                if not (np.all((0 <= X_train[key]) & (X_train[key] <= 1)) and np.all((0 <= X_val[key]) & (X_val[key] <= 1))):
                    raise ValueError("Expected scaled data within range [0, 1]. Please ensure data is preprocessed correctly.")