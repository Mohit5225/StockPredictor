from model.architecture import StockPredictor
from typing import Any , Dict , List
from datetime import timedelta 
import datetime
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from config import PredictionConfig

class ModelEvaluator:
    """The Judge, Jury, and Technical Executioner"""
    
    def __init__(self, predictor: StockPredictor):
        self.predictor = predictor
    
    def evaluate_prediction(self, symbol: str, 
                          evaluation_period: str = 'ytd') -> Dict[str, Any]:
        """Evaluate predictions like Aizawa judging his students"""
        
        # Get actual data for comparison
        end_date = datetime.now()
        if evaluation_period == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        else:
            start_date = end_date - timedelta(days=int(evaluation_period))
        
        actual_data = yf.download(symbol, start=start_date, end=end_date)
        predictions = []
        actuals = []
        
        # Generate predictions for each day
        for i in range(len(actual_data) - self.predictor.config.time_steps):
            window = actual_data.iloc[i:i+self.predictor.config.time_steps]
            pred = self.predictor.predict(symbol, window)
            predictions.append(pred['predictions'])
            actuals.append({
                'High': actual_data.iloc[i+self.predictor.config.time_steps]['High'],
                'Low': actual_data.iloc[i+self.predictor.config.time_steps]['Low'],
                'Close': actual_data.iloc[i+self.predictor.config.time_steps]['Close']
            })
        
        return self._calculate_metrics(predictions, actuals)
    
    def _calculate_metrics(self, predictions: List[Dict], 
                         actuals: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics like Senku calculating chemical formulas"""
        metrics = {}
        for key in ['High', 'Low', 'Close']:
            pred_values = [p[key] for p in predictions]
            actual_values = [a[key] for a in actuals]
            
            metrics[key] = {
                'MAPE': mean_absolute_percentage_error(actual_values, pred_values),
                'R2': r2_score(actual_values, pred_values),
                'MAE': mean_absolute_error(actual_values, pred_values)
            }
        
        return metrics

# Example Usage
if __name__ == "__main__":
    # For training your model
    dojo = StockMarketDojo()
    dojo.train_model('AAPL', '2017-01-01', '2025-01-13')
    
    # For making predictions
    config = PredictionConfig(
        model_path=Path('models/saved/AAPL_model.keras'),
        scaler_path=Path('models/saved/AAPL_scaler.pkl')
    )
    
    predictor = StockPredictor(config)
    
    # Get latest prediction
    prediction = predictor.predict('AAPL')
    print(f"Latest prediction: {prediction}")
    
    # Evaluate model performance
    evaluator = ModelEvaluator(predictor)
    metrics = evaluator.evaluate_prediction('AAPL', evaluation_period='30d')
    print(f"30-day performance metrics: {metrics}")