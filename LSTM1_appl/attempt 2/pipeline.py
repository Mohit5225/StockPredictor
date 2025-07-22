import logging
from pathlib import Path
import pickle
from typing import Dict, Any, List
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from config import Config, DataConfig, ModelConfig
from data.dataFetcher import DataFetcher
from data.data_preprocessor import DataPreprocessor
from model.architecture import StockPredictor
from model.model_training import ModelTrainer
from model.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictionPipeline:
    """Master Pipeline for Stock Prediction System"""
    
    def __init__(self, config_path: str = None):
        """Initialize pipeline with configuration"""
        self.config = Config()
        self.setup_directories()
        self.results_cache = {}
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config.base_path,
            self.config.model_path,
            Path("logs"),
            Path("results"),
            Path("plots")
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def train_multiple_stocks(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Train one model for multiple stocks"""
        logger.info(f"Starting training pipeline for stocks: {symbols}")
        
        try:
            # Initialize components
            fetcher = DataFetcher(self.config.data)
            preprocessor = DataPreprocessor(self.config.data)
            
            # Fetch data for all stocks
            stock_data = {}
            for symbol in symbols:
                df = fetcher.fetch_data(symbol, start_date, end_date)
                stock_data[symbol] = df
                
            (X_dict, y_array), feature_scaler, target_scaler = preprocessor.preprocess_multiple(stock_data)
            
            # Split data
            train_size = int(len(X_dict['price_input']) * self.config.data.train_split)
            val_size = int(len(X_dict['price_input']) * self.config.data.val_split)
            
            splits = {
                'train': ({
                    'price_input': X_dict['price_input'][:train_size],
                    'stock_input': X_dict['stock_input'][:train_size]
                }, y_array[:train_size]),
                'val': ({
                    'price_input': X_dict['price_input'][train_size:train_size + val_size],
                    'stock_input': X_dict['stock_input'][train_size:train_size + val_size]
                }, y_array[train_size:train_size + val_size]),
                'test': ({
                    'price_input': X_dict['price_input'][train_size + val_size:],
                    'stock_input': X_dict['stock_input'][train_size + val_size:]
                }, y_array[train_size + val_size:])
            }
            # Initialize and train model
            input_shape = (X_dict['price_input'].shape[1], X_dict['price_input'].shape[2])
            predictor = StockPredictor(self.config.model, input_shape ,self.config.data.stock_identifier_mapping)
            trainer = ModelTrainer(self.config.model)
            
            # Train
            history, metrics = trainer.train(
                predictor.model,
                splits['train'][0], splits['train'][1],
                splits['val'][0], splits['val'][1]
            )
            
            # Save model
            model_path = self.config.model_path / "multi_stock_model.keras"
            predictor.model.save(model_path)
            print("🛠 feature_scaler.n_features_in_ =", feature_scaler.n_features_in_)
            print("🛠 target_scaler.n_features_in_  =", target_scaler.n_features_in_)
            # Save scalers
            feature_scaler_path = self.config.model_path / "multi_stock_feature_scaler.pkl"
            target_scaler_path = self.config.model_path / "multi_stock_target_scaler.pkl"
            with open(feature_scaler_path, "wb") as f:
                pickle.dump(feature_scaler, f)
            with open(target_scaler_path, "wb") as f:
                pickle.dump(target_scaler, f)

            return {
                'symbols': symbols,
                'history': history,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in multi-stock training: {e}")
            raise

def main():
    """Main execution function"""
    pipeline = StockPredictionPipeline()
    
    # Train multiple stocks together
    symbols = ["AAPL", "MSFT", "GOOGL" , "NVDA" , "TSLA" , "AMD" , "META"]
    results = pipeline.train_multiple_stocks(
        symbols=symbols,
        start_date="2017-01-01",
        end_date="2025-07-15"
    )
    print("Training History:", results['history'])
    print("Evaluation Metrics:", results['metrics'])

if __name__ == "__main__":
    main()