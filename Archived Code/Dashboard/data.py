import os
import pickle
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import polars as pl
import numpy as np
import yfinance as yf
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# --- Configure Logging ---
logger = logging.getLogger(__name__)

class ModelCache:
    """Simple in-memory cache for loaded models"""
    def __init__(self, max_size: int = 50):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key: str):
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()

class DataManager:
    """Centralized data and model management"""
    
    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.model_cache = ModelCache()
        self.loaded_models = {}
        self.data_cache = {}
        
        # Create directory structure
        self.models_dir = self.base_path / "models"
        self.arima_dir = self.models_dir / "arima"
        self.lstm_dir = self.models_dir / "lstm" 
        self.sentiment_dir = self.models_dir / "sentiment"
        self.data_dir = self.base_path / "data"
        
        # Create directories if they don't exist
        for directory in [self.arima_dir, self.lstm_dir, self.sentiment_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize the data manager"""
        logger.info("Initializing DataManager...")
        
        # Create sample models if they don't exist
        await self._create_sample_models()
        
        # Load available stocks
        self.available_stocks = await self._scan_available_stocks()
        
        logger.info(f"DataManager initialized with {len(self.available_stocks)} stocks")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.model_cache.clear()
        self.data_cache.clear()
        logger.info("DataManager cleaned up")
    
    async def _scan_available_stocks(self) -> List[str]:
        """Scan for available stock models"""
        stocks = set()
        
        for model_dir in [self.arima_dir, self.lstm_dir, self.sentiment_dir]:
            if model_dir.exists():
                for file_path in model_dir.glob("*.pkl"):
                    stocks.add(file_path.stem.upper())
        
        # Default stocks if no models found
        if not stocks:
            stocks = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"}
        
        return sorted(list(stocks))
    
    async def _create_sample_models(self):
        """Create sample model files for demonstration"""
        sample_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        for symbol in sample_stocks:
            # Create dummy ARIMA model
            arima_path = self.arima_dir / f"{symbol}.pkl"
            if not arima_path.exists():
                dummy_arima = {
                    'model_type': 'arima',
                    'symbol': symbol,
                    'coefficients': np.random.normal(0, 0.1, 5),
                    'created_at': datetime.now().isoformat(),
                    'metrics': {
                        'mae': np.random.uniform(0.5, 2.0),
                        'mape': np.random.uniform(2.0, 8.0),
                        'rmse': np.random.uniform(1.0, 3.0),
                        'r2_score': np.random.uniform(0.7, 0.95)
                    }
                }
                with open(arima_path, 'wb') as f:
                    pickle.dump(dummy_arima, f)
            
            # Create dummy LSTM model
            lstm_path = self.lstm_dir / f"{symbol}.pkl"
            if not lstm_path.exists():
                dummy_lstm = {
                    'model_type': 'lstm',
                    'symbol': symbol,
                    'weights': np.random.normal(0, 0.1, (50, 1)),
                    'created_at': datetime.now().isoformat(),
                    'metrics': {
                        'mae': np.random.uniform(0.3, 1.5),
                        'mape': np.random.uniform(1.5, 6.0),
                        'rmse': np.random.uniform(0.8, 2.5),
                        'r2_score': np.random.uniform(0.75, 0.98)
                    }
                }
                with open(lstm_path, 'wb') as f:
                    pickle.dump(dummy_lstm, f)
            
            # Create dummy Sentiment model
            sentiment_path = self.sentiment_dir / f"{symbol}.pkl"
            if not sentiment_path.exists():
                dummy_sentiment = {
                    'model_type': 'sentiment',
                    'symbol': symbol,
                    'neural_weights': np.random.normal(0, 0.1, (100, 1)),
                    'sentiment_weights': np.random.normal(0, 0.05, 10),
                    'created_at': datetime.now().isoformat(),
                    'metrics': {
                        'mae': np.random.uniform(0.4, 1.8),
                        'mape': np.random.uniform(2.2, 7.5),
                        'rmse': np.random.uniform(1.2, 2.8),
                        'r2_score': np.random.uniform(0.72, 0.96)
                    }
                }
                with open(sentiment_path, 'wb') as f:
                    pickle.dump(dummy_sentiment, f)
    
    async def get_available_stocks(self) -> List[str]:
        """Get list of available stock symbols"""
        return self.available_stocks
    
    async def load_model(self, symbol: str, model_type: str):
        """Load a model from disk with caching"""
        cache_key = f"{symbol}_{model_type}"
        
        # Check cache first
        cached_model = self.model_cache.get(cache_key)
        if cached_model:
            return cached_model
        
        # Load from disk
        model_dirs = {
            'arima': self.arima_dir,
            'lstm': self.lstm_dir,
            'sentiment': self.sentiment_dir
        }
        
        model_path = model_dirs[model_type] / f"{symbol}.pkl"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Cache the model
            self.model_cache.set(cache_key, model)
            
            logger.info(f"Loaded {model_type} model for {symbol}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
            return None
    
    async def generate_predictions(self, symbol: str, model_type: str, days: int) -> List[Dict]:
        """Generate predictions using specified model"""
        model = await self.load_model(symbol, model_type)
        
        if not model:
            return []
        
        # Get current price for baseline
        current_price = await self.get_current_price(symbol)
        
        predictions = []
        base_date = datetime.now()
        
        # Generate predictions based on model type
        if model_type == 'arima':
            predictions = await self._generate_arima_predictions(model, current_price, days, base_date)
        elif model_type == 'lstm':
            predictions = await self._generate_lstm_predictions(model, current_price, days, base_date)
        elif model_type == 'sentiment':
            predictions = await self._generate_sentiment_predictions(model, symbol, current_price, days, base_date)
        
        return predictions
    
    async def _generate_arima_predictions(self, model: Dict, current_price: float, days: int, base_date: datetime) -> List[Dict]:
        """Generate ARIMA-based predictions"""
        predictions = []
        price = current_price
        
        # Simple ARIMA simulation using coefficients
        coefficients = model.get('coefficients', np.random.normal(0, 0.01, 5))
        
        for i in range(1, days + 1):
            # Simple autoregressive prediction with noise
            change = np.sum(coefficients * np.random.normal(0, 0.02, len(coefficients)))
            price = price * (1 + change)
            
            pred_date = base_date + timedelta(days=i)
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'predicted_price': round(price, 2),
                'confidence_interval_low': round(price * 0.95, 2),
                'confidence_interval_high': round(price * 1.05, 2)
            })
        
        return predictions
    
    async def _generate_lstm_predictions(self, model: Dict, current_price: float, days: int, base_date: datetime) -> List[Dict]:
        """Generate LSTM-based predictions"""
        predictions = []
        price = current_price
        
        # LSTM simulation with neural network weights
        weights = model.get('weights', np.random.normal(0, 0.01, (50, 1)))
        
        for i in range(1, days + 1):
            # Simulate LSTM prediction with more complex patterns
            sequence_input = np.random.normal(0, 0.1, 50)
            prediction_change = np.dot(sequence_input, weights.flatten()) * 0.01
            
            price = price * (1 + prediction_change)
            
            pred_date = base_date + timedelta(days=i)
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'predicted_price': round(price, 2),
                'confidence_interval_low': round(price * 0.92, 2),
                'confidence_interval_high': round(price * 1.08, 2)
            })
        
        return predictions
    
    async def _generate_sentiment_predictions(self, model: Dict, symbol: str, current_price: float, days: int, base_date: datetime) -> List[Dict]:
        """Generate sentiment-enhanced predictions"""
        predictions = []
        price = current_price
        
        # Get current sentiment
        sentiment_data = await self.get_sentiment_analysis(symbol)
        sentiment_factor = sentiment_data.get('polarity', 0) if sentiment_data else 0
        
        # Neural network and sentiment weights
        neural_weights = model.get('neural_weights', np.random.normal(0, 0.01, (100, 1)))
        sentiment_weights = model.get('sentiment_weights', np.random.normal(0, 0.005, 10))
        
        for i in range(1, days + 1):
            # Neural network prediction
            neural_input = np.random.normal(0, 0.1, 100)
            neural_prediction = np.dot(neural_input, neural_weights.flatten()) * 0.01
            
            # Sentiment influence (decays over time)
            sentiment_decay = np.exp(-i * 0.1)  # Sentiment effect decreases over time
            sentiment_influence = sentiment_factor * sentiment_decay * 0.02
            
            # Combined prediction
            total_change = neural_prediction + sentiment_influence
            price = price * (1 + total_change)
            
            pred_date = base_date + timedelta(days=i)
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'predicted_price': round(price, 2),
                'confidence_interval_low': round(price * 0.90, 2),
                'confidence_interval_high': round(price * 1.10, 2)
            })
        
        return predictions
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current stock price from Yahoo Finance"""
        cache_key = f"price_{symbol}"
        
        # Check cache (5-minute expiry)
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=5):
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                self.data_cache[cache_key] = (current_price, datetime.now())
                return current_price
            else:
                # Fallback to dummy price if no data
                return 150.0 + np.random.uniform(-10, 10)
        
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            # Return dummy price as fallback
            return 150.0 + np.random.uniform(-10, 10)
    
    async def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical stock data"""
        cache_key = f"hist_{symbol}_{days}"
        
        # Check cache (1-hour expiry)
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=1):
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            historical_data = []
            for date, row in hist.iterrows():
                historical_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'close': round(float(row['Close']), 2),
                    'volume': int(row['Volume']) if not np.isnan(row['Volume']) else None
                })
            
            self.data_cache[cache_key] = (historical_data, datetime.now())
            return historical_data
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            # Return dummy historical data
            return self._generate_dummy_historical_data(symbol, days)
    
    def _generate_dummy_historical_data(self, symbol: str, days: int) -> List[Dict]:
        """Generate dummy historical data for testing"""
        data = []
        base_price = 150.0
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            # Random walk for price
            base_price *= (1 + np.random.normal(0, 0.02))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'close': round(base_price, 2),
                'volume': np.random.randint(1000000, 10000000)
            })
        
        return data
    
    async def get_model_metrics(self, symbol: str) -> Dict[str, Dict]:
        """Get evaluation metrics for all models"""
        metrics = {}
        
        for model_type in ['arima', 'lstm', 'sentiment']:
            model = await self.load_model(symbol, model_type)
            if model and 'metrics' in model:
                metrics[model_type] = model['metrics']
        
        return metrics
    
    async def get_model_confidence(self, symbol: str, model_type: str) -> float:
        """Get model confidence score"""
        model = await self.load_model(symbol, model_type)
        
        if not model or 'metrics' not in model:
            return 0.85  # Default confidence
        
        # Calculate confidence based on R² score
        r2_score = model['metrics'].get('r2_score', 0.85)
        return min(0.99, max(0.5, r2_score))
    
    async def get_sentiment_analysis(self, symbol: str) -> Optional[Dict]:
        """Get sentiment analysis for a stock"""
        cache_key = f"sentiment_{symbol}"
        
        # Check cache (30-minute expiry)
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=30):
                return cached_data
        
        try:
            # Generate dummy sentiment data (in production, fetch from news APIs)
            sentiment_data = await self._generate_sentiment_data(symbol)
            
            self.data_cache[cache_key] = (sentiment_data, datetime.now())
            return sentiment_data
        
        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {str(e)}")
            return None
    
    async def _generate_sentiment_data(self, symbol: str) -> Dict:
        """Generate dummy sentiment data for demonstration"""
        # Simulate sentiment analysis
        polarity = np.random.uniform(-0.5, 0.5)
        subjectivity = np.random.uniform(0.3, 0.8)
        
        # Determine sentiment label
        if polarity > 0.1:
            sentiment_label = "Positive"
        elif polarity < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Generate dummy headlines
        headlines = [
            f"{symbol} shows strong quarterly performance",
            f"Market analysts upgrade {symbol} rating",
            f"{symbol} announces new product launch",
            f"Investors optimistic about {symbol} future",
            f"{symbol} reports better than expected earnings"
        ]
        
        # Randomly select 3-5 headlines
        selected_headlines = np.random.choice(headlines, size=np.random.randint(3, 6), replace=False).tolist()
        
        return {
            'symbol': symbol,
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'sentiment_label': sentiment_label,
            'recent_headlines': selected_headlines,
            'updated_at': datetime.now().isoformat()
        }
    
    async def retrain_model(self, symbol: str, model_type: str):
        """Retrain a model (background task)"""
        logger.info(f"Starting retraining of {model_type} model for {symbol}")
        
        try:
            # Simulate retraining process
            await asyncio.sleep(2)  # Simulate training time
            
            # Generate new model with updated metrics
            new_metrics = {
                'mae': np.random.uniform(0.2, 1.5),
                'mape': np.random.uniform(1.0, 6.0),
                'rmse': np.random.uniform(0.5, 2.0),
                'r2_score': np.random.uniform(0.8, 0.98)
            }
            
            # Load existing model and update metrics
            model = await self.load_model(symbol, model_type)
            if model:
                model['metrics'] = new_metrics
                model['retrained_at'] = datetime.now().isoformat()
                
                # Save updated model
                model_dirs = {
                    'arima': self.arima_dir,
                    'lstm': self.lstm_dir,
                    'sentiment': self.sentiment_dir
                }
                
                model_path = model_dirs[model_type] / f"{symbol}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Clear cache to force reload
                cache_key = f"{symbol}_{model_type}"
                if cache_key in self.model_cache.cache:
                    del self.model_cache.cache[cache_key]
                    del self.model_cache.access_times[cache_key]
                
                logger.info(f"Successfully retrained {model_type} model for {symbol}")
        
        except Exception as e:
            logger.error(f"Error retraining model for {symbol}: {str(e)}")
    
    async def clear_cache(self):
        """Clear all cached data"""
        self.model_cache.clear()
        self.data_cache.clear()
        logger.info("All caches cleared")

# --- Utility Functions ---
def preprocess_stock_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess stock data using Polars"""
    return (
        df
        .with_columns([
            pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d"),
            pl.col("Close").cast(pl.Float64),
            pl.col("Volume").cast(pl.Int64)
        ])
        .sort("Date")
        .with_columns([
            # Technical indicators
            pl.col("Close").rolling_mean(window_size=5).alias("MA_5"),
            pl.col("Close").rolling_mean(window_size=20).alias("MA_20"),
            pl.col("Close").pct_change().alias("Returns"),
            pl.col("Close").rolling_std(window_size=20).alias("Volatility")
        ])
        .drop_nulls()
    )

def calculate_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate additional technical indicators"""
    return (
        df
        .with_columns([
            # RSI calculation (simplified)
            pl.when(pl.col("Returns") > 0)
            .then(pl.col("Returns"))
            .otherwise(0)
            .rolling_mean(window_size=14)
            .alias("RSI_gain"),
            
            pl.when(pl.col("Returns") < 0)
            .then(pl.col("Returns").abs())
            .otherwise(0)
            .rolling_mean(window_size=14)
            .alias("RSI_loss")
        ])
        .with_columns([
            (100 - (100 / (1 + pl.col("RSI_gain") / pl.col("RSI_loss")))).alias("RSI")
        ])
        .drop(["RSI_gain", "RSI_loss"])
    )