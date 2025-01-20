from config import Config
from data.dataFetcher import DataFetcher
from data.data_preprocessor import DataPreprocessor
from model.architecture import StockPredictor
from model.model_training import ModelTrainer
from model.evaluation import ModelEvaluator  
from typing import Dict, Any



class StockMarketDojo:
    """The Ultimate Fusion"""
    def __init__(self, config_path: str = "config.json"):
        self.config = Config()  # Load from JSON in production
        self.setup_directories()
    
    def setup_directories(self):
        self.config.base_path.mkdir(parents=True, exist_ok=True)
        self.config.model_path.mkdir(parents=True, exist_ok=True)
    
    def train_model(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        # Initialize components
        fetcher = DataFetcher(self.config.data)
        preprocessor = DataPreprocessor(self.config.data)
        
        # Fetch and preprocess data
        df = fetcher.fetch_data(symbol, start_date, end_date)
        X, y, scaler = preprocessor.preprocess(df)
        
        # Split data
        train_size = int(len(X) * self.config.data.train_split)
        val_size = int(len(X) * self.config.data.val_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Initialize and train model
        predictor = StockPredictor(self.config.model, input_shape=(X.shape[1], X.shape[2]))
        trainer = ModelTrainer(self.config.model)
        history = trainer.train(predictor.model, X_train, y_train, X_val, y_val)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics, predictions = evaluator.evaluate(predictor.model, X_test, y_test, scaler)
        
        # Save artifacts
        predictor.model.save(self.config.model_path / f"{symbol}_model.keras")
        
        return {
            'history': history,
            'metrics': metrics,
            'predictions': predictions
        }

# Example usage
if __name__ == "__main__":
    dojo = StockMarketDojo()
    results = dojo.train_model('AAPL', '2017-01-01', '2025-01-13')
    print("Training History:", results['history'])
    print("Evaluation Metrics:", results['metrics'])