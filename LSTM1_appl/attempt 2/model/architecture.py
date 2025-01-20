# models/architecture.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2

from config import ModelConfig

class StockPredictor:
    """The Prophecy Engine"""
    def __init__(self, config: ModelConfig, input_shape: tuple):
        self.config = config
        self.model = self._build_model(input_shape)
    
    def _build_model(self, input_shape: tuple) -> tf.keras.Model:
        model = Sequential()
        
        for i, units in enumerate(self.config.lstm_units):
            model.add(LSTM(
                units,
                return_sequences=i < len(self.config.lstm_units) - 1,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                input_shape=input_shape if i == 0 else None
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config.dropout_rates[i]))
        
        model.add(Dense(3, activation='linear'))  # High, Low, Close
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss='huber')
        
        return model
