import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from typing import Optional
import logging
from config import DataConfig
from data_preprocessor import TechnicalIndicators


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataFetcher:
    """The Sacred Scroll of Market Data"""
    def __init__(self, config: DataConfig):
        """
        Initialize the DataFetcher with the given configuration.
        """
        self.config = config

        # Initialize Alpha Vantage TimeSeries client if needed
        if config.api_source == "alphavantage" and config.api_key:
            self.ts = TimeSeries(key=config.api_key)
        elif config.api_source == "alphavantage" and not config.api_key:
            raise ValueError("API Key is required for Alpha Vantage!")

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data for a given symbol and date range.

        Args:
            symbol (str): The stock symbol to fetch data for.
            start_date (str): The start date in YYYY-MM-DD format.
            end_date (str): The end date in YYYY-MM-DD format.

        Returns:
            pd.DataFrame: A DataFrame containing the fetched and processed data.
        """
        # Fetch data based on API source
        if self.config.api_source == "yahoo":
            df = yf.download(symbol, start=start_date, end=end_date)
        elif self.config.api_source == "alphavantage":
            data, _ = self.ts.get_daily(symbol=symbol, outputsize="full")
            df = pd.DataFrame(data).loc[start_date:end_date]
        else:
            raise ValueError(f"Invalid API source: {self.config.api_source}. Fix your config!")

        # Reset index for cleaner data
        df.reset_index(inplace=True)

        # Validate that base features are present
        missing_features = set(self.config.base_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Fetched data is missing required base features: {missing_features}")

        # Add derived features
        if "daily_return" in self.config.derived_features:
            df['daily_return'] = df['Close'].pct_change()
        if "volatility" in self.config.derived_features:
            df['volatility'] = df['daily_return'].rolling(window=5).std()

        # Add technical indicators dynamically
        if self.config.technical_indicators.get("RSI", False):
            df['RSI'] = TechnicalIndicators.calculate_rsi(df)
        if self.config.technical_indicators.get("MACD", False):
            df['MACD'], df['MACD_Signal'] = TechnicalIndicators.calculate_macd(df)
        if self.config.technical_indicators.get("BBANDS", False):
            df['BB_Upper'], df['BB_Lower'] = TechnicalIndicators.calculate_bollinger_bands(df)
        if self.config.technical_indicators.get("ATR", False):
            df['ATR'] = TechnicalIndicators.calculate_atr(df)

        # Filter active features
        active_features = self.config.get_active_features()
        df = df[active_features]

        logger.info(f"Fetched data for {symbol}: {len(df)} rows, {len(df.columns)} columns.")
        return df
