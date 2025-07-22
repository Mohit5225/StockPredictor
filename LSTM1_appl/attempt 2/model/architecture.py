import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Embedding, Concatenate
from tensorflow.keras.regularizers import l1_l2
import logging
from config import ModelConfig
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, config: ModelConfig, input_shape: tuple , stock_identifier_mapping: dict):
        self.config = config
        self.stock_identifier_mapping = stock_identifier_mapping
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape: tuple) -> tf.keras.Model:
        try:
            # Main input for price/indicator data
            price_input = Input(shape=input_shape, name='price_input')
            
            # Stock identifier input
            stock_input = Input(shape=(1,), name='stock_input')
            
            # Embedding layer for stock identifiers
            stock_embedding = Embedding(
                input_dim=len(self.stock_identifier_mapping), 
                output_dim=8,
                name='stock_embedding'
            )(stock_input)
            stock_embedding = tf.keras.layers.Flatten()(stock_embedding)
            
            # LSTM layers for price processing
            x = price_input
            for i, units in enumerate(self.config.lstm_units):
                x = LSTM(
                    units,
                    return_sequences=i < len(self.config.lstm_units) - 1,
                    kernel_regularizer=l1_l2(
                        l1=self.config.l1_regularizer,
                        l2=self.config.l2_regularizer
                    )
                )(x)
                x = BatchNormalization()(x)
                x = Dropout(self.config.dropout_rates[i])(x)
            
            # Combine LSTM output with stock embedding
            combined = Concatenate()([x, stock_embedding])
            
            # Output layer
            outputs = Dense(self.config.output_dim, activation='linear')(combined)
            
            # Create model with multiple inputs
            model = tf.keras.Model(
                inputs=[price_input, stock_input],
                outputs=outputs
            )
            
            # Compile
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            model.compile(optimizer=optimizer, loss='huber')
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise