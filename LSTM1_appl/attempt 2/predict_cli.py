from predictor import PredictionConfig, StockPredictor
from data.dataFetcher import DataFetcher
from config import DataConfig
from pathlib import Path
import sys
from datetime import datetime, timedelta

def main():
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "saved" / "multi_stock_model.keras"
    feature_scaler_path = base_dir / "models" / "saved" / "multi_stock_feature_scaler.pkl"
    target_scaler_path = base_dir / "models" / "saved" / "multi_stock_target_scaler.pkl"

    # Check if model and scaler files exist before proceeding
    for p, desc in [
        (model_path, "Model file"),
        (feature_scaler_path, "Feature scaler file"),
        (target_scaler_path, "Target scaler file")
    ]:
        if not p.exists():
            print(f"Error: {desc} not found at {p.resolve()}. Please train the model first.")
            sys.exit(1)

    config = PredictionConfig(
        model_path=model_path,
        feature_scaler_path=feature_scaler_path,
        target_scaler_path=target_scaler_path,
        time_steps=30
    )
    data_config = DataConfig()  # Initialize DataConfig for DataFetcher
    predictor = StockPredictor(config, data_config)
    fetcher = DataFetcher(data_config)  # Pass DataConfig to DataFetcher

    try:
        # Ask for stock symbol
        symbol = input("Enter stock symbol (e.g. AAPL): ").strip().upper()
        
        # Ask for the date to predict
        prediction_date_str = input("Enter the date to predict for (YYYY-MM-DD): ").strip()
        prediction_date = datetime.strptime(prediction_date_str, "%Y-%m-%d").date()

        # Calculate date range to fetch historical data
        # Fetch more than time_steps to account for non-trading days
        end_date = prediction_date
        start_date = end_date - timedelta(days=config.time_steps * 2)

        # Fetch the data
        print(f"Fetching historical data for {symbol} up to {end_date} to make a prediction...")
        data = fetcher.fetch_data(symbol, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
        
        if data.shape[0] < config.time_steps:
            print(f"Not enough historical data found ({data.shape[0]} days) to make a prediction for {symbol}. Need at least {config.time_steps} days.")
            sys.exit(1)

        # Process and predict
        print("Processing data and making prediction...")
        result = predictor.predict(data, symbol)
        
        # Display the prediction
        print(f"\nPrediction for {symbol} on {prediction_date_str}:")
        print(f"  High: ${result['predictions']['High']:.2f}")
        print(f"  Low: ${result['predictions']['Low']:.2f}")
        print(f"  Close: ${result['predictions']['Close']:.2f}")
        print(f"  Confidence: {result['confidence_score']:.2f}")
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()