import json
import dynamic_predict

if __name__ == "__main__":
    # Load default settings
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Get user input
    stock_symbol = input(f"Enter stock symbol (default: {config['default_stocks'][0]}): ") or config['default_stocks'][0]
    future_steps = int(input(f"Enter number of future steps (default: {config['default_future_steps']}): ") or config['default_future_steps'])
    time_unit = input(f"Predict for days or hours (default: {config['default_time_unit']}): ") or config['default_time_unit']
    interval = '1h' if time_unit == 'hours' else '1d'

    # Define model path
    model_path = f"{stock_symbol}_lstm_model.keras"

    # Predict future prices
    dynamic_predict.dynamic_predict(stock_symbol, future_steps, time_unit, interval, model_path)
