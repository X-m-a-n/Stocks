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

SIMULATION_DATE = "2024-10-01"

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

def initialize_data_manager(stocks_df: pl.DataFrame, sentiment_df: pl.DataFrame = None, models_dir: str = "C:/Users/Joshh/Projects/Stocks/Models"):
    """Initialize the data manager with stock and sentiment data"""
    global _stocks_df, _sentiment_df
    
    _stocks_df = stocks_df
    _sentiment_df = sentiment_df
    
    # Ensure date columns are properly formatted
    if _stocks_df['date'].dtype == pl.Utf8:
        _stocks_df = _stocks_df.with_columns(pl.col('date').str.to_date())
    
    if _sentiment_df is not None and 'publication_date' in _sentiment_df.columns:
        if _sentiment_df['publication_date'].dtype == pl.Utf8:
            _sentiment_df = _sentiment_df.with_columns(
                pl.col('publication_date').str.to_date().alias('date')
            )
    
    available_stocks = get_available_stocks(models_dir)
    logger.info(f"Initialized with {len(available_stocks)} stocks")
    
    return available_stocks

def get_available_stocks(models_dir: str = "models") -> List[str]:
    """Get list of stocks that have trained models (ARIMA or LSTM)"""
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

def load_model_metrics(models_dir: str = "models") -> Dict[str, ModelMetrics]:
    """Load model performance metrics from results files"""
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
        
        # Rest of function stays the same...
        
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
                    logger.warning(f"❌ Error extracting {feature}: {e}, using zeros")
                    feature_values = np.zeros(len(recent_data))
                    missing_count += 1
            else:
                # Create placeholder for missing feature
                feature_values = np.zeros(len(recent_data))
                logger.warning(f"❌ Feature {i+1:2d}: {feature} (MISSING - using zeros)")
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

def load_pytorch_model(symbol: str, models_dir: str = "models") -> Optional[MomentumLSTM]:
    """Load PyTorch LSTM model"""
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

def load_arima_model(symbol: str, models_dir: str = "models") -> Optional[Dict]:
    """Load ARIMA model from pickle file"""
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

def get_available_models(symbol: str, models_dir: str = "models") -> List[str]:
    """Get list of available model types for a symbol"""
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

def generate_lstm_prediction(symbol: str, prediction_date: str, models_dir: str = "models") -> Optional[PredictionResult]:
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

def generate_arima_prediction(symbol: str, prediction_date: str, models_dir: str = "models") -> Optional[PredictionResult]:
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

def generate_predictions_for_dates(symbol: str, dates: List[str], model_type: str, models_dir: str = "models") -> List[PredictionResult]:
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
        start = start_date #datetime.strptime(start_date, '%Y-%m-%d')
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

async def get_stock_predictions(symbol: str, prediction_dates: List[str], model_type) -> Dict:
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
    global _model_cache, _sentiment_cache
    _model_cache.clear()
    _sentiment_cache.clear()
    logger.info("All caches cleared")
