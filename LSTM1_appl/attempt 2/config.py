from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)

class MarketCap(Enum):
    LARGE = "large"
    MID = "mid"
    SMALL = "small"

class Volatility(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Sector(Enum):
    TECH = "Technology"
    AUTO = "Automotive"
    FINANCE = "Finance"

@dataclass
class StockMetadata:
    symbol: str
    sector: Sector
    market_cap: MarketCap
    volatility: Volatility
    description: str

@dataclass
class DataConfig:
    # Base Configuration - As immutable as Saitama's training routine
    api_source: str = "yahoo"
    api_key: Optional[str] = None
    time_steps: int = 30
    train_split: float = field(default=0.8)
    val_split: float = field(default=0.1)
    batch_size: int = 32
     # Add multi-stock training config
    enable_multi_stock: bool = True
    # Embedding Configuration
    stock_identifier_mapping: Dict[str, int] = field(default_factory=lambda: {
        "AAPL": 0,
        "TSLA": 1,
        "MSFT": 2,
        "NVDA": 3,
        "GOOGL": 4,
        "AMD": 5,
        "META": 6
    })
     
    # Feature Management - More organized than Todoroki's dual quirk
    base_features: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume"
    ])
    
    derived_features: List[str] = field(default_factory=lambda: [
        "volatility" , "daily_return" 
    ])
  
    @property
    def get_active_features(self) -> List[str]:
        active_indicators = [
            key for key, is_active in self.technical_indicators.items() if is_active
        ]
        return self.base_features + self.derived_features + active_indicators
    

    technical_indicators: Dict[str, bool] = field(default_factory=lambda: {
        "RSI": True,
        "MACD": True,
        "BBANDS": False,
        "ATR": True,
        "OBV": False
    })

    # Stock Universe - More detailed than a Hunter x Hunter power system
    stock_metadata: Dict[str, StockMetadata] = field(default_factory=lambda: {
        "AAPL": StockMetadata(
            "AAPL", Sector.TECH, MarketCap.LARGE, Volatility.MEDIUM,
            "The iPhone company that makes Android users question their life choices"
        ),
        "TSLA": StockMetadata(
            "TSLA", Sector.AUTO, MarketCap.LARGE, Volatility.HIGH,
            "Memes + Cars + Rockets = Stonks"
        ),
        "MSFT": StockMetadata(
            "MSFT", Sector.TECH, MarketCap.LARGE, Volatility.MEDIUM,
            "The software giant that’s as embedded in PCs as Windows"
        ),
        "NVDA": StockMetadata(
            "NVDA", Sector.TECH, MarketCap.LARGE, Volatility.HIGH,
            "Graphics cards that made gaming an art form and AI a reality"
        ),
        "GOOGL": StockMetadata(
            "GOOGL", Sector.TECH, MarketCap.LARGE, Volatility.MEDIUM,
            "The king of search engines and the overlord of data"
        ),
        "AMD": StockMetadata(
            "AMD", Sector.TECH, MarketCap.LARGE, Volatility.HIGH,
            "The underdog fighting Intel for CPU dominance and winning over gamers"
        ),
        "META": StockMetadata(
            "META", Sector.TECH, MarketCap.LARGE, Volatility.MEDIUM,
            "The parent of Facebook, trying to reinvent social with the Metaverse"
        ),
        # Add more stocks like you're collecting Dragon Balls
    })

    def __post_init__(self):
        # Validation stronger than All Might's punch
        if self.train_split + self.val_split >= 1.0:
            raise ValueError("Your splits add up to more than 1! What are you, a math dropout?")
        
        if self.batch_size <= 0:
            raise ValueError("Negative batch size? Are you trying to train backwards in time?")

    def get_symbol_active_features(self, symbol: str) -> List[str]:
        """Get features like you're assembling the Infinity Gauntlet"""
        metadata = self.stock_metadata[symbol]
        features = self.base_features + self.derived_features
        
        # Add technical indicators based on stock characteristics
        if metadata.volatility == Volatility.HIGH:
            features.extend(["ATR", "BB_Width"])
        
        # Add enabled technical indicators
        features.extend([ind for ind, enabled in self.technical_indicators.items() if enabled])
        
        return list(dict.fromkeys(features))  # Preserves order, removes duplicates


    def prepare_stock_data(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Process data like you're crafting the perfect jutsu"""
        metadata = self.stock_metadata[symbol]
        df = data.copy()  # Never modify the original, young padawan    
        
        # Apply filters based on market cap
        if metadata.market_cap == MarketCap.SMALL:
            df = df[df['volatility'] < df['volatility'].quantile(0.95)]
        
        # Add sector-specific features
        if metadata.sector == Sector.TECH:
            df['tech_momentum'] = self._calculate_tech_momentum(df)
        
        return df

    def _calculate_tech_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate tech momentum like you're charging a Spirit Bomb"""
        return df['Close'].pct_change(20).rolling(window=5).mean()

@dataclass
class ModelConfig:
    epochs: int = 100
    patience: int = 5
    reduce_lr_factor: float = 0.2
    learning_rate: float = 0.001
    lstm_units: list = field(default_factory=lambda: [64, 32])  # Example default units
    dropout_rates: list = field(default_factory=lambda: [0.2, 0.3])  # Example dropout rates
    output_dim: int = 3  # Default to High, Low, Close
    l1_regularizer: float = 1e-5
    l2_regularizer: float = 1e-4
    batch_size: int = 32

    def __post_init__(self):
        if self.patience >= self.epochs:
            raise ValueError("Patience should be less than total epochs.")
        if not (0 < self.reduce_lr_factor < 1):
            raise ValueError("reduce_lr_factor must be between 0 and 1.")
        if len(self.lstm_units) != len(self.dropout_rates):
            raise ValueError("lstm_units and dropout_rates lengths must match.")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    base_path: Path = field(default_factory=lambda: Path("data/processed"))
    model_path: Path = field(default_factory=lambda: Path("models/saved"))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
