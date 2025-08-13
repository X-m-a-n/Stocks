import os
import json
import logging
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import requests
from urllib.parse import urljoin
import tempfile

SIMULATION_DATE = "2024-10-01"

# GitHub repository configuration
GITHUB_BASE_URL = "https://raw.githubusercontent.com/X-m-a-n/Stocks/main/"
GITHUB_DATA_URL = GITHUB_BASE_URL + "Data/"
GITHUB_MODELS_URL = GITHUB_BASE_URL + "Models/"

def get_current_date():
    """Get current date or simulation date"""
    if SIMULATION_DATE:
        return datetime.strptime(SIMULATION_DATE, '%Y-%m-%d').date()
    else:
        return datetime.now().date()

# Configure logging
logger = logging.getLogger(__name__)

# Data structures
@dataclass
class PredictionResult:
    date: str
    predicted_price: float
    confidence_low: float
    confidence_high: float

@dataclass
class SentimentResult:
    company_sentiment: float
    market_sentiment: float
    sentiment_label: str
    headlines: List[Dict]
    wordcloud_b64: str = None

@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    mape: float
    accuracy: float

# Global data storage
_stocks_df = None
_sentiment_df = None
_model_cache = {}
_sentiment_cache = {}
_github_file_cache = {}

# ================================
# GITHUB DATA LOADING FUNCTIONS
# ================================

def download_from_github(url: str, cache_key: str = None) -> Optional[bytes]:
    """Download file from GitHub with caching"""
    if cache_key and cache_key in _github_file_cache:
        return _github_file_cache[cache_key]
    
    try:
        logger.info(f"Downloading from GitHub: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content = response.content
        if cache_key:
            _github_file_cache[cache_key] = content
        
        logger.info(f"Successfully downloaded {len(content)} bytes")
        return content
    
    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        return None

def load_stock_data_from_github() -> Optional[pl.DataFrame]:
    """Load stock data from GitHub parquet file"""
    try:
        url = GITHUB_DATA_URL + "clean_stock_data.parquet"
        content = download_from_github(url, "stock_data")
        
        if content is None:
            return None
        
        # Save to temporary file and load with polars
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            df = pl.read_parquet(tmp_path)
            logger.info(f"Loaded stock data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Ensure date column is properly formatted
            if df['date'].dtype == pl.Utf8:
                df = df.with_columns(pl.col('date').str.to_date())
            
            return df
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error loading stock data from GitHub: {e}")
        return None

def load_sentiment_data_from_github() -> Optional[pl.DataFrame]:
    """Load sentiment data from GitHub parquet file"""
    try:
        url = GITHUB_DATA_URL + "sentiment_data_FINAL.parquet"
        content = download_from_github(url, "sentiment_data")
        
        if content is None:
            return None
        
        # Save to temporary file and load with polars
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            df = pl.read_parquet(tmp_path)
            logger.info(f"Loaded sentiment data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Ensure date columns are properly formatted
            if 'publication_date' in df.columns:
                if df['publication_date'].dtype == pl.Utf8:
                    df = df.with_columns(
                        pl.col('publication_date').str.to_date().alias('date')
                    )
            
            return df
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error loading sentiment data from GitHub: {e}")
        return None

def get_available_stocks_from_github() -> List[str]:
    """Get list of stocks that have trained models on GitHub"""
    stocks = set()
    
    try:
        # Get ARIMA models list
        arima_models = fetch_github_directory_listing("Models/ARIMA/")
        for filename in arima_models:
            if filename.endswith("_arima_model.pkl"):
                symbol = filename.replace("_arima_model.pkl", "")
                stocks.add(symbol)
        
        # Get LSTM models list  
        lstm_models = fetch_github_directory_listing("Models/LSTM/")
        for filename in lstm_models:
            if filename.endswith("_model.pth"):
                symbol = filename.replace("_model.pth", "")
                stocks.add(symbol)
        
        logger.info(f"Found {len(stocks)} stocks with models on GitHub")
        return sorted(list(stocks))
    
    except Exception as e:
        logger.error(f"Error getting available stocks from GitHub: {e}")
        return []

def fetch_github_directory_listing(path: str) -> List[str]:
    """Fetch directory listing from GitHub API"""
    try:
        api_url = f"https://api.github.com/repos/X-m-a-n/Stocks/contents/{path}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        files = response.json()
        if isinstance(files, list):
            return [file['name'] for file in files if file['type'] == 'file']
        return []
    
    except Exception as e:
        logger.warning(f"Could not fetch directory listing for {path}: {e}")
        # Fallback: try common stock symbols
        common_symbols = ['AGL', 'BHP', 'CLS', 'DSY', 'FSR', 'GRT', 'INL', 'INP', 
                         'KIO', 'LHC', 'MTN', 'NED', 'NPN', 'SBK', 'SHP', 'SLM', 
                         'SOL', 'TKG', 'VOD', 'WHL']
        return common_symbols

def load_pytorch_model_from_github(symbol: str) -> Optional[torch.nn.Module]:
    """Load PyTorch LSTM model from GitHub"""
    cache_key = f"{symbol}_lstm_model"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        # Try LSTM directory first
        url = GITHUB_MODELS_URL + f"LSTM/{symbol}_model.pth"
        content = download_from_github(url, f"lstm_{symbol}")
        
        if content is None:
            # Try root models directory
            url = GITHUB_MODELS_URL + f"{symbol}_model.pth"
            content = download_from_github(url, f"lstm_root_{symbol}")
        
        if content is None:
            logger.warning(f"PyTorch model not found on GitHub for {symbol}")
            return None
        
        # Save to temporary file and load
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            input_size = 20
            model = MomentumLSTM(input_size)
            model.load_state_dict(torch.load(tmp_path, map_location='cpu'))
            model.eval()
            
            _model_cache[cache_key] = model
            logger.info(f"Loaded PyTorch LSTM model for {symbol} from GitHub")
            return model
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error loading PyTorch model for {symbol} from GitHub: {e}")
        return None

def load_arima_model_from_github(symbol: str) -> Optional[Dict]:
    """Load ARIMA model from GitHub"""
    cache_key = f"{symbol}_arima_model"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        url = GITHUB_MODELS_URL + f"ARIMA/{symbol}_arima_model.pkl"
        content = download_from_github(url, f"arima_{symbol}")
        
        if content is None:
            logger.warning(f"ARIMA model not found on GitHub for {symbol}")
            return None
        
        # Save to temporary file and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Ensure model has required structure
            if isinstance(model_data, dict):
                model_data['symbol'] = symbol
                model_data['model_type'] = 'arima'
            else:
                # If model is just the fitted model object, wrap it
                model_data = {
                    'fitted_model': model_data,
                    'symbol': symbol,
                    'model_type': 'arima',
                    'created_at': datetime.now().isoformat()
                }
            
            _model_cache[cache_key] = model_data
            logger.info(f"Loaded ARIMA model for {symbol} from GitHub")
            return model_data
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error loading ARIMA model for {symbol} from GitHub: {e}")
        return None

def load_model_metrics_from_github() -> Dict[str, ModelMetrics]:
    """Load model performance metrics from GitHub results files"""
    results_files = [
        "lstm_results.xlsx",
        "lstm_results.csv", 
        "parallel_lstm_results.csv"
    ]
    
    metrics = {}
    
    for filename in results_files:
        try:
            url = GITHUB_BASE_URL + filename
            content = download_from_github(url, f"metrics_{filename}")
            
            if content is None:
                continue
            
            # Save to temporary file and load
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(tmp_path, sheet_name='Successful Models')
                else:
                    df = pd.read_csv(tmp_path)
                
                for _, row in df.iterrows():
                    if row.get('status') == 'completed':
                        symbol = row['symbol']
                        metrics[symbol] = ModelMetrics(
                            mae=row.get('mae', 0.0),
                            rmse=row.get('rmse', 0.0),
                            mape=row.get('mape', 0.0),
                            accuracy=row.get('direction_accuracy', 0.5)
                        )
                
                logger.info(f"Loaded metrics for {len(metrics)} symbols from {filename}")
                break  # Use first successful file
            finally:
                os.unlink(tmp_path)
        
        except Exception as e:
            logger.warning(f"Could not load metrics from {filename}: {e}")
            continue
    
    return metrics

# ================================
# PYTORCH MODEL DEFINITION
# ================================

class MomentumLSTM(nn.Module):
    """PyTorch LSTM model - matches training script exactly"""
    def __init__(self, input_size):
        super(MomentumLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=192,
            num_heads=6,
            dropout=0.1,
            batch_first=True
        )
        
        self.enhancement = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.regression_head = nn.Linear(96, 1)
        self.classification_head = nn.Linear(96, 4)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        features = attn_out[:, -1, :]
        enhanced = self.enhancement(features)
        price_pred = self.regression_head(enhanced)
        direction_pred = self.classification_head(enhanced)
        return price_pred, direction_pred

# ================================
# INITIALIZATION FUNCTIONS
# ================================

def initialize_data_manager(stocks_df: pl.DataFrame = None, sentiment_df: pl.DataFrame = None, models_dir: str = None) -> List[str]:
    """Initialize the data manager with stock and sentiment data"""
    global _stocks_df, _sentiment_df
    
    # If no data provided, load from GitHub
    if stocks_df is None:
        _stocks_df = load_stock_data_from_github()
    else:
        _stocks_df = stocks_df
    
    if sentiment_df is None:
        _sentiment_df = load_sentiment_data_from_github()
    else:
        _sentiment_df = sentiment_df
    
    # Ensure date columns are properly formatted
    if _stocks_df is not None and _stocks_df['date'].dtype == pl.Utf8:
        _stocks_df = _stocks_df.with_columns(pl.col('date').str.to_date())
    
    if _sentiment_df is not None and 'publication_date' in _sentiment_df.columns:
        if _sentiment_df['publication_date'].dtype == pl.Utf8:
            _sentiment_df = _sentiment_df.with_columns(
                pl.col('publication_date').str.to_date().alias('date')
            )
    
    # Get available stocks (from GitHub if no local models_dir)
    if models_dir is None:
        available_stocks = get_available_stocks_from_github()
    else:
        available_stocks = get_available_stocks_local(models_dir)
    
    logger.info(f"Initialized with {len(available_stocks)} stocks")
    return available_stocks

def get_available_stocks(models_dir: str = None) -> List[str]:
    """Get list of stocks that have trained models"""
    if models_dir is None:
        return get_available_stocks_from_github()
    else:
        return get_available_stocks_local(models_dir)

def get_available_stocks_local(models_dir: str) -> List[str]:
    """Get list of stocks that have trained models locally"""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    stocks = set()
    
    # Check ARIMA models
    arima_dir = models_path / "ARIMA"
    if arima_dir.exists():
        for model_file in arima_dir.glob("*_arima_model.pkl"):
            symbol = model_file.stem.replace("_arima_model", "")
            stocks.add(symbol)
    
    # Check LSTM models
    lstm_dir = models_path / "LSTM"
    if lstm_dir.exists():
        for model_file in lstm_dir.glob("*_model.pth"):
            symbol = model_file.stem.replace("_model", "")
            stocks.add(symbol)
    
    # Also check root models directory for PyTorch files
    for model_file in models_path.glob("*_model.pth"):
        symbol = model_file.stem.replace("_model", "")
        stocks.add(symbol)
    
    return sorted(list(stocks))

def load_model_metrics(models_dir: str = None) -> Dict[str, ModelMetrics]:
    """Load model performance metrics"""
    if models_dir is None:
        return load_model_metrics_from_github()
    else:
        return load_model_metrics_local(models_dir)

def load_model_metrics_local(models_dir: str) -> Dict[str, ModelMetrics]:
    """Load model performance metrics from local results files"""
    results_files = [
        "lstm_results_*.xlsx",
        "lstm_results_*.csv",
        "parallel_lstm_results.csv"
    ]
    
    metrics = {}
    
    # Try to find and load results file
    for pattern in results_files:
        files = list(Path(".").glob(pattern))
        if files:
            try:
                # Use the most recent file
                results_file = max(files, key=os.path.getctime)
                
                if results_file.suffix == '.xlsx':
                    df = pd.read_excel(results_file, sheet_name='Successful Models')
                else:
                    df = pd.read_csv(results_file)
                
                for _, row in df.iterrows():
                    if row.get('status') == 'completed':
                        symbol = row['symbol']
                        metrics[symbol] = ModelMetrics(
                            mae=row.get('mae', 0.0),
                            rmse=row.get('rmse', 0.0),
                            mape=row.get('mape', 0.0),
                            accuracy=row.get('direction_accuracy', 0.5)
                        )
                break
            except Exception as e:
                logger.warning(f"Could not load {results_file}: {e}")
                continue
    
    return metrics

# ================================
# STOCK DATA FUNCTIONS
# ================================

def get_current_price(symbol: str, predictionDate: str = None) -> Optional[float]:
    """Get most recent closing price before prediction start date"""
    if _stocks_df is None:
        return None
    
    try:
        query = _stocks_df.filter(pl.col('symbol') == symbol)
        
        # If predictionDate provided, get price from before that date
        if predictionDate:
            cutoff_date = datetime.strptime(predictionDate, '%Y-%m-%d').date()
            query = query.filter(pl.col('date') < cutoff_date)
        
        latest_price = (
            query
            .sort('date', descending=True)
            .select('closing_price')
            .head(1)
            .item()
        )
        return float(latest_price)
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        return None

def get_historical_data(symbol: str, days: int = 30, prediction_start_date: str = None) -> List[Dict]:
    """Get historical price data ending before prediction start date"""
    if _stocks_df is None:
        return []
    
    try:
        query = _stocks_df.filter(pl.col('symbol') == symbol)
        
        # Add date filter if prediction_start_date provided
        if prediction_start_date:
            cutoff_date = datetime.strptime(prediction_start_date, '%Y-%m-%d').date()
            query = query.filter(pl.col('date') < cutoff_date)
        
        historical = (
            query
            .sort('date', descending=True)
            .head(days)
            .sort('date')
            .select(['date', 'closing_price', 'volume'])
        )
        
        result = []
        for row in historical.iter_rows(named=True):
            result.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'close': float(row['closing_price']),
                'volume': int(row.get('volume', 0))
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        return []

def get_features_for_sequence(symbol: str, end_date: str, sequence_length: int = 10) -> Optional[np.ndarray]:
    """Get feature sequence for model input - FIXED to match exact training features"""
    if _stocks_df is None:
        return None
    
    try:
        training_features = [
            "closing_price", "opening_price", "high_price", "low_price",     
            "intraday_return", "daily_range", "returns", "rsi_14",        
            "macd", "volume_ratio", "volatility_10d", "volatility_20d", 
            "sma_20", "sma_50", "volume_ma10", "volume_ma20", "volume",
            "signal_line", "avg_sentiment", "entity_specific_sentiment"
        ]
        
        logger.info(f"Using {training_features} for {symbol}")
        
        # Get data up to end_date
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        recent_data = (
            _stocks_df
            .filter(
                (pl.col('symbol') == symbol) &
                (pl.col('date') < end_dt)
            )
            .sort('date')
            .tail(sequence_length)
        )
        
        if len(recent_data) < sequence_length:
            logger.warning(f"Insufficient data for {symbol}: need {sequence_length}, got {len(recent_data)}")
            return None
        
        # Get available columns in the dataset
        available_columns = _stocks_df.columns
        
        # Build feature array with exact training features (ensuring 20 features)
        feature_arrays = []
        missing_count = 0
        
        for i, feature in enumerate(training_features):
            if feature in available_columns:
                # Use actual feature data
                try:
                    feature_values = recent_data.select(feature).to_numpy().flatten()
                    logger.debug(f"✅ Feature {i+1:2d}: {feature} (from data)")
                except Exception as e:
                    logger.warning(f"⚠ Error extracting {feature}: {e}, using zeros")
                    feature_values = np.zeros(len(recent_data))
                    missing_count += 1
            else:
                # Create placeholder for missing feature
                feature_values = np.zeros(len(recent_data))
                logger.warning(f"⚠ Feature {i+1:2d}: {feature} (MISSING - using zeros)")
                missing_count += 1
            
            # Handle NaN values
            feature_values = np.nan_to_num(feature_values, nan=0.0)
            feature_arrays.append(feature_values)
        
        # Create final array: shape (sequence_length, 20)
        data_array = np.array(feature_arrays).T
        
        logger.info(f"Feature extraction for {symbol}: shape={data_array.shape}, missing={missing_count}/20")
        
        # Verify shape is exactly what LSTM expects
        if data_array.shape != (sequence_length, 20):
            logger.error(f"Shape mismatch for {symbol}! Expected ({sequence_length}, 20), got {data_array.shape}")
            return None
        
        # Final verification that we have exactly 20 features
        if data_array.shape[1] != 20:
            logger.error(f"Feature count error for {symbol}: expected 20, got {data_array.shape[1]}")
            return None
        
        logger.info(f"✅ Successfully created feature sequence for {symbol}: {data_array.shape}")
        return data_array
    
    except Exception as e:
        logger.error(f"Error getting features for {symbol}: {e}")
        return None

# ================================
# MODEL FUNCTIONS
# ================================

def load_pytorch_model(symbol: str, models_dir: str = None) -> Optional[MomentumLSTM]:
    """Load PyTorch LSTM model"""
    if models_dir is None:
        return load_pytorch_model_from_github(symbol)
    else:
        return load_pytorch_model_local(symbol, models_dir)

def load_pytorch_model_local(symbol: str, models_dir: str) -> Optional[MomentumLSTM]:
    """Load PyTorch LSTM model from local directory"""
    cache_key = f"{symbol}_lstm_model"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        model_path = Path(models_dir) / "LSTM" / f"{symbol}_model.pth"
        
        if not model_path.exists():
            # Try alternative path
            model_path = Path(models_dir) / f"{symbol}_model.pth"
        
        if not model_path.exists():
            logger.warning(f"PyTorch model file not found: {model_path}")
            return None
        
        input_size = 20
        
        # Load model
        model = MomentumLSTM(input_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        _model_cache[cache_key] = model
        logger.info(f"Loaded PyTorch LSTM model for {symbol}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading PyTorch model for {symbol}: {e}")
        return None

def load_arima_model(symbol: str, models_dir: str = None) -> Optional[Dict]:
    """Load ARIMA model"""
    if models_dir is None:
        return load_arima_model_from_github(symbol)
    else:
        return load_arima_model_local(symbol, models_dir)

def load_arima_model_local(symbol: str, models_dir: str) -> Optional[Dict]:
    """Load ARIMA model from local pickle file"""
    cache_key = f"{symbol}_arima_model"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        model_path = Path(models_dir) / "ARIMA" / f"{symbol}_arima_model.pkl"
        
        if not model_path.exists():
            logger.warning(f"ARIMA model file not found: {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Ensure model has required structure
        if isinstance(model_data, dict):
            model_data['symbol'] = symbol
            model_data['model_type'] = 'arima'
        else:
            # If model is just the fitted model object, wrap it
            model_data = {
                'fitted_model': model_data,
                'symbol': symbol,
                'model_type': 'arima',
                'created_at': datetime.now().isoformat()
            }
        
        _model_cache[cache_key] = model_data
        logger.info(f"Loaded ARIMA model for {symbol}")
        return model_data
    
    except Exception as e:
        logger.error(f"Error loading ARIMA model for {symbol}: {e}")
        return None

def get_available_models(symbol: str, models_dir: str = None) -> List[str]:
    """Get list of available model types for a symbol"""
    if models_dir is None:
        # Check if models exist on GitHub (simplified check)
        if symbol in get_available_stocks_from_github():
            return ['arima', 'lstm']
        return []
    else:
        available_models = []
        
        # Check ARIMA
        arima_path = Path(models_dir) / "ARIMA" / f"{symbol}_arima_model.pkl"
        if arima_path.exists():
            available_models.append('arima')
        
        # Check LSTM (PyTorch)
        lstm_path = Path(models_dir) / "LSTM" / f"{symbol}_model.pth"
        if not lstm_path.exists():
            lstm_path = Path(models_dir) / f"{symbol}_model.pth"
        if lstm_path.exists():
            available_models.append('lstm')
        
        return available_models

def generate_lstm_prediction(symbol: str, prediction_date: str, models_dir: str = None) -> Optional[PredictionResult]:
    """Generate LSTM prediction for a specific date"""
    try:
        # Load model
        model = load_pytorch_model(symbol, models_dir)
        if model is None:
            return None
        
        # Get feature sequence
        features = get_features_for_sequence(symbol, prediction_date)
        if features is None:
            return None
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Generate prediction
        with torch.no_grad():
            price_pred, direction_pred = model(input_tensor)
            predicted_change = float(price_pred.item())
        
        # Get current price to calculate absolute prediction
        current_price = get_current_price(symbol, prediction_date)
        if current_price is None:
            return None
        
        # Convert percentage change to absolute price
        predicted_price = current_price * (1 + predicted_change / 100)
        
        # Calculate confidence intervals (±10% of prediction)
        confidence_width = abs(predicted_price * 0.1)
        
        return PredictionResult(
            date=prediction_date,
            predicted_price=round(predicted_price, 2),
            confidence_low=round(predicted_price - confidence_width, 2),
            confidence_high=round(predicted_price + confidence_width, 2)
        )
    
    except Exception as e:
        logger.error(f"Error generating LSTM prediction for {symbol}: {e}")
        return None

def generate_arima_prediction(symbol: str, prediction_date: str, models_dir: str = None) -> Optional[PredictionResult]:
    """Generate ARIMA prediction for a specific date"""
    try:
        # Load ARIMA model
        model_data = load_arima_model(symbol, models_dir)
        if model_data is None:
            return None
        
        fitted_model = model_data.get('fitted_model')
        if fitted_model is None:
            return None
        
        # Get current price
        current_price = get_current_price(symbol, prediction_date)
        if current_price is None:
            return None
        
        # Calculate days ahead
        pred_date = datetime.strptime(prediction_date, '%Y-%m-%d').date()
        today = get_current_date()
        days_ahead = (pred_date - today).days
        
        if days_ahead <= 0:
            return None
        
        # Generate ARIMA forecast
        try:
            forecast = fitted_model.forecast(steps=days_ahead)
            predicted_price = float(forecast[-1]) if hasattr(forecast, '__getitem__') else float(forecast)
        except Exception as e:
            logger.error(f"ARIMA forecast failed for {symbol}: {e}")
            predicted_price = current_price  # Fallback
        
        # Ensure reasonable bounds
        predicted_price = max(predicted_price, current_price * 0.5)
        predicted_price = min(predicted_price, current_price * 2.0)
        
        # Calculate confidence intervals
        confidence_width = abs(predicted_price * 0.15)  # ±15% for ARIMA
        
        return PredictionResult(
            date=prediction_date,
            predicted_price=round(predicted_price, 2),
            confidence_low=round(predicted_price - confidence_width, 2),
            confidence_high=round(predicted_price + confidence_width, 2)
        )
    
    except Exception as e:
        logger.error(f"Error generating ARIMA prediction for {symbol}: {e}")
        return None

def generate_predictions_for_dates(symbol: str, dates: List[str], model_type: str, models_dir: str = None) -> List[PredictionResult]:
    """Generate predictions for multiple dates using specified model type"""
    predictions = []
    
    for date in dates:
        if model_type == "lstm":
            prediction = generate_lstm_prediction(symbol, date, models_dir)
        elif model_type == "arima":
            prediction = generate_arima_prediction(symbol, date, models_dir)
        else:
            logger.error(f"Unknown model type: {model_type}")
            continue
        
        if prediction:
            predictions.append(prediction)
    
    return predictions

def get_next_business_days(start_date: str, num_days: int) -> List[str]:
    """Get next N business days from start date"""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        business_days = []
        current = start
        
        while len(business_days) < num_days:
            current += timedelta(days=1)
            # Skip weekends
            if current.weekday() < 5:
                business_days.append(current.strftime('%Y-%m-%d'))
        
        return business_days
    except Exception as e:
        logger.error(f"Error getting business days: {e}")
        return []

# ================================
# SENTIMENT FUNCTIONS
# ================================

def get_company_sentiment(symbol: str) -> SentimentResult:
    """Get sentiment analysis for a company"""
    if _sentiment_df is None:
        return SentimentResult(
            company_sentiment=0.0,
            market_sentiment=0.0,
            sentiment_label="Neutral",
            headlines=[]
        )
    
    try:
        # Get recent news for this company
        company_news = (
            _sentiment_df
            .filter(pl.col("jse_symbols").list.contains(symbol))
            .sort("publication_date", descending=True)
            .head(10)
        )
        
        if company_news.is_empty():
            company_sentiment = 0.0
            headlines = []
        else:
            # Calculate company sentiment from recent news
            company_sentiment = company_news["combined_sentiment_compound"].mean()
            
            # Get top 3 headlines
            headlines = []
            for row in company_news.head(3).iter_rows(named=True):
                headlines.append({
                    'title': row.get('title', 'No title'),
                    'date': row['date'].strftime('%Y-%m-%d') if row.get('date') else 'Unknown',
                    'sentiment': round(float(row.get('combined_sentiment_compound', 0.0)), 3)
                })
        
        # Calculate overall market sentiment
        market_sentiment = _sentiment_df["combined_sentiment_compound"].mean()
        
        # Determine sentiment label
        if company_sentiment > 0.1:
            sentiment_label = "Positive"
        elif company_sentiment < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Generate word cloud from latest article keywords
        wordcloud_b64 = generate_wordcloud_from_keywords(company_news.head(1), symbol)
        
        return SentimentResult(
            company_sentiment=round(float(company_sentiment), 3),
            market_sentiment=round(float(market_sentiment), 3),
            sentiment_label=sentiment_label,
            headlines=headlines,
            wordcloud_b64=wordcloud_b64
        )
    
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        return SentimentResult(
            company_sentiment=0.0,
            market_sentiment=0.0,
            sentiment_label="Neutral",
            headlines=[]
        )

def generate_wordcloud_from_keywords(news_df: pl.DataFrame, symbol: str) -> str:
    """Generate word cloud from article keywords"""
    try:
        if news_df.is_empty():
            return ""
        
        # Extract keywords from the latest article
        keywords_text = ""
        for row in news_df.iter_rows(named=True):
            keywords = row.get('keywords', '')
            if keywords:
                keywords_text += f" {keywords}"
        
        if not keywords_text.strip():
            return ""
        
        # Create word cloud
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color='white',
            max_words=30,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate(keywords_text)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.figure(figsize=(6, 3))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Latest News Keywords - {symbol}', fontsize=12)
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        img_b64 = base64.b64encode(img_buffer.read()).decode()
        
        return img_b64
    
    except Exception as e:
        logger.error(f"Error generating word cloud for {symbol}: {e}")
        return ""

# ================================
# MAIN API FUNCTIONS
# ================================

async def get_available_stocks_list() -> List[str]:
    """API function: Get list of available stocks"""
    return get_available_stocks()

async def get_stock_predictions(symbol: str, prediction_dates: List[str], model_type: str) -> Dict:
    """API function: Get predictions for specific dates"""
    try:
        predictions = generate_predictions_for_dates(symbol, prediction_dates, model_type)
        first_pred_date = min(datetime.strptime(date, '%Y-%m-%d').date() for date in prediction_dates).strftime('%Y-%m-%d')
        current_price = get_current_price(symbol, first_pred_date)
        historical_data = get_historical_data(symbol, 30, first_pred_date)
        
        # Get model metrics
        metrics = load_model_metrics()
        model_metrics = metrics.get(symbol, ModelMetrics(0, 0, 0, 0.5))
        
        return {
            'symbol': symbol,
            'model_type': model_type,
            'current_price': current_price,
            'predictions': [p.__dict__ for p in predictions],
            'historical': historical_data,
            'model_metrics': model_metrics.__dict__,
            'confidence': model_metrics.accuracy,
            'generated_at': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in get_stock_predictions: {e}")
        return {'error': str(e)}

async def get_sentiment_analysis(symbol: str) -> Dict:
    """API function: Get sentiment analysis"""
    try:
        sentiment = get_company_sentiment(symbol)
        
        return {
            'symbol': symbol,
            'company_sentiment': sentiment.company_sentiment,
            'market_sentiment': sentiment.market_sentiment,
            'sentiment_label': sentiment.sentiment_label,
            'recent_headlines': sentiment.headlines,
            'wordcloud_image': sentiment.wordcloud_b64,
            'updated_at': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in get_sentiment_analysis: {e}")
        return {'error': str(e)}

async def get_model_metrics(symbol: str) -> Dict:
    """API function: Get model performance metrics"""
    try:
        metrics = load_model_metrics()
        model_metrics = metrics.get(symbol, ModelMetrics(0, 0, 0, 0.5))
        
        return {
            'symbol': symbol,
            'lstm': model_metrics.__dict__
        }
    
    except Exception as e:
        logger.error(f"Error in get_model_metrics: {e}")
        return {'error': str(e)}

async def clear_cache():
    """Clear all caches"""
    global _model_cache, _sentiment_cache, _github_file_cache
    _model_cache.clear()
    _sentiment_cache.clear()
    _github_file_cache.clear()
    logger.info("All caches cleared")

# ================================
# CONVENIENCE FUNCTIONS
# ================================

def check_github_connection() -> bool:
    """Test connection to GitHub repository"""
    try:
        test_url = GITHUB_BASE_URL + "README.md"
        response = requests.get(test_url, timeout=10)
        return response.status_code == 200
    except:
        return False

def get_data_info() -> Dict:
    """Get information about loaded datasets"""
    info = {
        'stock_data_loaded': _stocks_df is not None,
        'sentiment_data_loaded': _sentiment_df is not None,
        'github_connection': check_github_connection(),
        'cached_files': list(_github_file_cache.keys()),
        'cached_models': list(_model_cache.keys())
    }
    
    if _stocks_df is not None:
        info['stock_data_shape'] = _stocks_df.shape
        info['stock_symbols'] = _stocks_df['symbol'].unique().to_list()[:10]  # First 10
        info['stock_date_range'] = [
            _stocks_df['date'].min().strftime('%Y-%m-%d'),
            _stocks_df['date'].max().strftime('%Y-%m-%d')
        ]
    
    if _sentiment_df is not None:
        info['sentiment_data_shape'] = _sentiment_df.shape
        if 'date' in _sentiment_df.columns:
            info['sentiment_date_range'] = [
                _sentiment_df['date'].min().strftime('%Y-%m-%d'),
                _sentiment_df['date'].max().strftime('%Y-%m-%d')
            ]
    
    return info

def refresh_data():
    """Force refresh of data from GitHub"""
    global _stocks_df, _sentiment_df, _github_file_cache
    
    logger.info("Refreshing data from GitHub...")
    
    # Clear cache
    _github_file_cache.clear()
    
    # Reload data
    _stocks_df = load_stock_data_from_github()
    _sentiment_df = load_sentiment_data_from_github()
    
    logger.info("Data refresh completed")
    return get_data_info()