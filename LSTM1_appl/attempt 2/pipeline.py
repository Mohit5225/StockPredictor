import logging
from pathlib import Path
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

    def train_model(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Train model for a specific stock"""
        logger.info(f"Starting training pipeline for {symbol}")
        
        try:
            # Initialize components
            fetcher = DataFetcher(self.config.data)
            preprocessor = DataPreprocessor(self.config.data)
            
            # Fetch and preprocess data
            df = fetcher.fetch_data(symbol, start_date, end_date)
            X, y, scaler = preprocessor.preprocess(df)
            
            # Split data
            train_size = int(len(X) * self.config.data.train_split)
            val_size = int(len(X) * self.config.data.val_split)
            
            splits = {
                'train': (X[:train_size], y[:train_size]),
                'val': (X[train_size:train_size + val_size], y[train_size:train_size + val_size]),
                'test': (X[train_size + val_size:], y[train_size + val_size:])
            }
            
            # Initialize and train model
            predictor = StockPredictor(self.config.model, input_shape=(X.shape[1], X.shape[2]))
            trainer = ModelTrainer(self.config.model)
            
            # Train
            history, metrics = trainer.train(
                predictor.model,
                splits['train'][0], splits['train'][1],
                splits['val'][0], splits['val'][1]
            )
            
            # Evaluate
            evaluator = ModelEvaluator()
            eval_metrics, predictions = evaluator.evaluate(
                predictor.model, 
                splits['test'][0], 
                splits['test'][1],
                scaler
            )
            
            # Save model and artifacts
            model_path = self.config.model_path / f"{symbol}_model.keras"
            predictor.model.save(model_path)
            
            # Save results
            results = {
                'symbol': symbol,
                'training_history': history,
                'training_metrics': metrics,
                'evaluation_metrics': eval_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self._save_results(symbol, results)
            self._plot_training_history(symbol, history)
            
            logger.info(f"Training completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline for {symbol}: {e}")
            raise

    def predict(self, symbol: str, days_ahead: int = 5) -> Dict[str, Any]:
        """Generate predictions for a stock"""
        try:
            model_path = self.config.model_path / f"{symbol}_model.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"No trained model found for {symbol}")
            
            predictor = StockPredictor(self.config.model)
            return predictor.predict(symbol, days_ahead)
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise

    def _save_results(self, symbol: str, results: Dict[str, Any]):
        """Save results to JSON"""
        results_path = Path("results") / f"{symbol}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

    def _plot_training_history(self, symbol: str, history: Dict[str, List]):
        """Plot and save training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{symbol} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(Path("plots") / f"{symbol}_training_history.png")
        plt.close()

def main():
    """Main execution function"""
    pipeline = StockPredictionPipeline()
    
    # Available stocks from config
    available_stocks = list(pipeline.config.data.stock_metadata.keys())
    
    print("\nStock Prediction System")
    print("=====================")
    print("\nAvailable stocks:", available_stocks)
    
    while True:
        print("\nOptions:")
        print("1. Train new model")
        print("2. Make predictions")
        print("3. View results")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            symbol = input(f"\nEnter stock symbol {available_stocks}: ").upper()
            if symbol not in available_stocks:
                print(f"Invalid symbol. Please choose from {available_stocks}")
                continue
                
            start_date = input("Enter start date (YYYY-MM-DD) [default: 2017-01-01]: ") or "2017-01-01"
            end_date = input("Enter end date (YYYY-MM-DD) [default: 2025-01-13]: ") or "2025-01-13"
            
            results = pipeline.train_model(symbol, start_date, end_date)
            print(f"\nTraining completed for {symbol}")
            print(f"Final validation loss: {results['training_metrics']['val_loss'][-1]:.4f}")
            
        elif choice == '2':
            symbol = input(f"\nEnter stock symbol {available_stocks}: ").upper()
            if symbol not in available_stocks:
                print(f"Invalid symbol. Please choose from {available_stocks}")
                continue
                
            days = int(input("Enter number of days to predict (1-30): "))
            if not 1 <= days <= 30:
                print("Please enter a number between 1 and 30")
                continue
                
            predictions = pipeline.predict(symbol, days)
            print(f"\nPredictions for {symbol} (next {days} days):")
            for day, pred in enumerate(predictions['predictions'], 1):
                print(f"Day {day}: ${pred:.2f}")
                
        elif choice == '3':
            symbol = input(f"\nEnter stock symbol {available_stocks}: ").upper()
            results_path = Path("results") / f"{symbol}_results.json"
            
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
                print(f"\nResults for {symbol}:")
                print(f"Training completed: {results['timestamp']}")
                print(f"Final metrics: {results['evaluation_metrics']}")
            else:
                print(f"No results found for {symbol}")
                
        elif choice == '4':
            print("\nExiting...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()