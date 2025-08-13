"""
Simple Data Manager - Function-First Approach
Works with LSTM models trained by the parallel training script
"""

import os
import json
import logging
import pickle
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Configure logging
logger = logging.getLogger(__name__)

# Data structures using dataclasses
@dataclass
class PredictionPoint:
    date: str
    predicted_price: float
    confidence_low: float = None
    confidence_high: float = None

@dataclass
class SentimentData:
    company_sentiment: float
    market_sentiment: float
    sentiment_label: str
    headlines: List[dict]
    wordcloud_b64: str = None

@dataclass
class ModelInfo:
    symbol: str
    model_type: str
    accuracy: float
    mae: float
    rmse: float
    model_path: str

# Global data storage (simple approach)
_stocks_df = None
_models_cache = {}
_sentiment_cache = {}

# ================================
# INITIALIZATION FUNCTIONS
# ================================

def initialize_data(stocks_df: pl.DataFrame, models_dir: str = "models", sentiment_file: str = None):
    """Initialize the data manager with stock data and model directory"""
    global _stocks_df
    
    _stocks_df = stocks_df
    
    # Ensure date column is properly formatted
    if _stocks_df['date'].dtype == pl.Utf8:
        _stocks_df = _stocks_df.with_columns(pl.col('date').str.to_date())
    
    available_stocks = get_available_stocks(models_dir)
    logger.info(f"Initialized with {len(available_stocks)} stocks")
    
    return available_stocks

def get_available_stocks(models_dir: str = "models") -> List[str]:
    """Get list of stocks that have trained models"""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    stocks = set()
    for model_file in models_path.glob("*_model.pth"):
        symbol = model_file.stem.replace("_simple_model", "").replace("_advanced_model", "").replace("_model", "")
        stocks.add(symbol)
    
    return sorted(list(stocks))

def load_model_info(models_dir: str = "models") -> Dict[str, ModelInfo]:
    """Load model information from results CSV"""
    results_files = [
        "clean_lstm_results.csv",
        "optimized_lstm_results.csv", 
        "advanced_lstm_comparison.csv",
        "parallel_advanced_lstm_results.csv"
    ]
    
    model_info = {}
    
    for results_file in results_files:
        if os.path.exists(results_file):
            try:
                results_df = pd.read_csv(results_file)
                
                for _, row in results_df.iterrows():
                    if row.get('status') == 'completed':
                        symbol = row['symbol']
                        model_type = row.get('config', 'lstm')
                        
                        model_info[f"{symbol}_{model_type}"] = ModelInfo(
                            symbol=symbol,
                            model_type=model_type,
                            accuracy=row.get('direction_accuracy', 0.5),
                            mae=row.get('mae', 0.0),
                            rmse=row.get('rmse', 0.0),
                            model_path=row.get('model_path', f"models/{symbol}_model.pth")
                        )
                break
            except Exception as e:
                logger.warning(f"Could not load {results_file}: {e}")
                continue
    
    return model_info

# ================================
# STOCK DATA FUNCTIONS
# ================================

def get_current_price(symbol: str) -> Optional[float]:
    """Get most recent price for a symbol"""
    if _stocks_df is None:
        return None
    
    try:
        latest_price = (
            _stocks_df
            .filter(pl.col('symbol') == symbol)
            .sort('date', descending=True)
            .select('closing_price')
            .head(1)
            .item()
        )
        return float(latest_price)
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        return None

def get_historical_data(symbol: str, days: int = 30) -> List[dict]:
    """Get historical price data"""
    if _stocks_df is None:
        return []
    
    try:
        # Get recent data
        historical = (
            _stocks_df
            .filter(pl.col('symbol') == symbol)
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
                'volume': int(row.get('volume', 1000000))
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        return []

def get_stock_data_for_date_range(symbol: str, start_date: str, end_date: str) -> pl.DataFrame:
    """Get stock data for a specific date range"""
    if _stocks_df is None:
        return pl.DataFrame()
    
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        filtered_data = (
            _stocks_df
            .filter(
                (pl.col('symbol') == symbol) &
                (pl.col('date') >= start_dt) &
                (pl.col('date') <= end_dt)
            )
            .sort('date')
        )
        
        return filtered_data
    except Exception as e:
        logger.error(f"Error getting data for {symbol} from {start_date} to {end_date}: {e}")
        return pl.DataFrame()

# ================================
# MODEL FUNCTIONS
# ================================

def load_pytorch_model(model_path: str, input_size: int = 20):
    """Load PyTorch LSTM model from file"""
    if not os.path.exists(model_path):
        return None
    
    try:
        # Define the model architecture (matches training script)
        class MomentumLSTM(nn.Module):
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
        
        # Load model
        model = MomentumLSTM(input_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def prepare_model_input(symbol: str, prediction_date: str, sequence_length: int = 10) -> Optional[torch.Tensor]:
    """Prepare input data for model prediction"""
    if _stocks_df is None:
        return None
    
    try:
        # Get features (assuming these exist in your data)
        feature_cols = [
            'returns', 'intraday_return', 'daily_range', 'rsi_14', 'macd', 
            'volume_ratio', 'volatility_10d', 'sma_20', 'sma_50', 'volume_ma10'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in _stocks_df.columns]
        
        if len(available_features) < 5:
            logger.warning(f"Only {len(available_features)} features available for {symbol}")
            return None
        
        # Get data up to prediction date
        pred_date = datetime.strptime(prediction_date, '%Y-%m-%d').date()
        
        recent_data = (
            _stocks_df
            .filter(
                (pl.col('symbol') == symbol) &
                (pl.col('date') < pred_date)
            )
            .sort('date')
            .tail(sequence_length)
            .select(available_features)
        )
        
        if len(recent_data) < sequence_length:
            logger.warning(f"Insufficient data for {symbol}: need {sequence_length}, got {len(recent_data)}")
            return None
        
        # Convert to numpy and then to tensor
        data_array = recent_data.to_numpy()
        
        # Handle any NaN values
        if np.isnan(data_array).any():
            data_array = np.nan_to_num(data_array, nan=0.0)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(data_array).unsqueeze(0)  # Shape: [1, sequence_length, features]
        
        return input_tensor
    except Exception as e:
        logger.error(f"Error preparing model input for {symbol}: {e}")
        return None

def generate_lstm_prediction(symbol: str, model_path: str, prediction_date: str) -> Optional[PredictionPoint]:
    """Generate a single prediction using LSTM model"""
    try:
        # Load model
        model = load_pytorch_model(model_path)
        if model is None:
            return None
        
        # Prepare input
        input_tensor = prepare_model_input(symbol, prediction_date)
        if input_tensor is None:
            return None
        
        # Make prediction
        with torch.no_grad():
            price_pred, direction_pred = model(input_tensor)
            predicted_price = float(price_pred.item())
        
        # Get current price for bounds checking
        current_price = get_current_price(symbol)
        if current_price:
            # Ensure prediction is reasonable (within 50% change)
            if predicted_price < current_price * 0.5:
                predicted_price = current_price * 0.5
            elif predicted_price > current_price * 1.5:
                predicted_price = current_price * 1.5
        
        # Calculate confidence intervals (simple approach)
        confidence_width = predicted_price * 0.1  # ±10%
        
        return PredictionPoint(
            date=prediction_date,
            predicted_price=round(predicted_price, 2),
            confidence_low=round(predicted_price - confidence_width, 2),
            confidence_high=round(predicted_price + confidence_width, 2)
        )
    
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {e}")
        return None

def generate_predictions_for_dates(symbol: str, model_type: str, dates: List[str]) -> List[PredictionPoint]:
    """Generate predictions for multiple dates"""
    model_info = load_model_info()
    model_key = f"{symbol}_{model_type}"
    
    if model_key not in model_info:
        logger.error(f"No model found for {symbol} with type {model_type}")
        return []
    
    model_path = model_info[model_key].model_path
    predictions = []
    
    for date in dates:
        prediction = generate_lstm_prediction(symbol, model_path, date)
        if prediction:
            predictions.append(prediction)
    
    return predictions

def compare_models(symbol: str, dates: List[str]) -> Dict[str, List[PredictionPoint]]:
    """Compare predictions from different models for the same dates"""
    model_info = load_model_info()
    results = {}
    
    # Find available models for this symbol
    available_models = []
    for key, info in model_info.items():
        if info.symbol == symbol:
            available_models.append(info.model_type)
    
    # Generate predictions for each model
    for model_type in available_models:
        predictions = generate_predictions_for_dates(symbol, model_type, dates)
        if predictions:
            results[model_type] = predictions
    
    return results

# ================================
# SENTIMENT FUNCTIONS
# ================================

def load_sentiment_data(sentiment_file: str) -> pl.DataFrame:
    """Load sentiment data from parquet file"""
    try:
        if not os.path.exists(sentiment_file):
            logger.warning(f"Sentiment file not found: {sentiment_file}")
            return pl.DataFrame()
        
        df = pl.read_parquet(sentiment_file)
        
        # Process sentiment columns
        df = df.with_columns([
            # Combine title and text sentiment
            pl.when((pl.col("title_fin_polarity") != 0) & (pl.col("text_fin_polarity") != 0))
            .then((pl.col("title_fin_polarity") + pl.col("text_fin_polarity")) / 2)
            .otherwise(pl.col("text_fin_polarity"))
            .alias("company_sentiment"),
            
            # Market sentiment (could be average across all companies)
            pl.col("text_fin_polarity").alias("market_sentiment"),
            
            # Sentiment label
            pl.when(pl.col("text_fin_polarity") > 0.1)
            .then("Positive")
            .when(pl.col("text_fin_polarity") < -0.1)
            .then("Negative")
            .otherwise("Neutral")
            .alias("sentiment_label")
        ])
        
        return df
    except Exception as e:
        logger.error(f"Error loading sentiment data: {e}")
        return pl.DataFrame()

def get_company_sentiment(symbol: str, sentiment_df: pl.DataFrame) -> SentimentData:
    """Get sentiment analysis for a specific company"""
    try:
        if sentiment_df.is_empty():
            return SentimentData(
                company_sentiment=0.0,
                market_sentiment=0.0,
                sentiment_label="Neutral",
                headlines=[]
            )
        
        # Get recent news for this company
        company_news = (
            sentiment_df
            .filter(pl.col("symbol") == symbol)
            .sort("date", descending=True)
            .head(10)
        )
        
        if company_news.is_empty():
            company_sentiment = 0.0
            headlines = []
        else:
            company_sentiment = company_news["company_sentiment"].mean()
            headlines = []
            
            for row in company_news.head(3).iter_rows(named=True):
                headlines.append({
                    'title': row.get('title', 'No title'),
                    'date': row['date'].strftime('%Y-%m-%d') if row.get('date') else 'Unknown',
                    'sentiment': row.get('company_sentiment', 0.0)
                })
        
        # Get overall market sentiment
        market_sentiment = sentiment_df["market_sentiment"].mean()
        
        # Determine sentiment label
        if company_sentiment > 0.1:
            sentiment_label = "Positive"
        elif company_sentiment < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Generate word cloud from latest article
        wordcloud_b64 = generate_wordcloud(company_news.head(1), symbol)
        
        return SentimentData(
            company_sentiment=round(float(company_sentiment), 3),
            market_sentiment=round(float(market_sentiment), 3),
            sentiment_label=sentiment_label,
            headlines=headlines,
            wordcloud_b64=wordcloud_b64
        )
    
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        return SentimentData(
            company_sentiment=0.0,
            market_sentiment=0.0,
            sentiment_label="Neutral",
            headlines=[]
        )

def generate_wordcloud(news_df: pl.DataFrame, symbol: str) -> str:
    """Generate word cloud from news text and convert to base64"""
    try:
        if news_df.is_empty():
            return ""
        
        # Extract text content
        text_content = ""
        for row in news_df.iter_rows(named=True):
            title = row.get('title', '')
            text = row.get('cleaned_text', '')
            text_content += f" {title} {text}"
        
        if not text_content.strip():
            return ""
        
        # Create word cloud
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color='white',
            max_words=50,
            colormap='viridis'
        ).generate(text_content)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Latest News Keywords - {symbol}')
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
# UTILITY FUNCTIONS
# ================================

def get_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate list of dates between start and end date"""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return dates
    except Exception as e:
        logger.error(f"Error generating date range: {e}")
        return []

def get_business_days(start_date: str, days: int) -> List[str]:
    """Get next N business days from start date"""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        business_days = []
        current = start
        
        while len(business_days) < days:
            current += timedelta(days=1)
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() < 5:
                business_days.append(current.strftime('%Y-%m-%d'))
        
        return business_days
    except Exception as e:
        logger.error(f"Error getting business days: {e}")
        return []

def validate_prediction_date(date_str: str) -> bool:
    """Validate that prediction date is reasonable"""
    try:
        pred_date = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now()
        
        # Must be in the future but not more than 30 days ahead
        if pred_date <= today:
            return False
        
        if pred_date > today + timedelta(days=30):
            return False
        
        return True
    except:
        return False

# ================================
# MAIN API FUNCTIONS
# ================================

def get_stock_prediction(symbol: str, model_type: str, prediction_date: str) -> Dict:
    """Main function to get stock prediction"""
    try:
        # Validate inputs
        if not validate_prediction_date(prediction_date):
            return {'error': 'Invalid prediction date'}
        
        available_stocks = get_available_stocks()
        if symbol not in available_stocks:
            return {'error': f'Symbol {symbol} not available'}
        
        # Generate prediction
        prediction = generate_predictions_for_dates(symbol, model_type, [prediction_date])
        
        if not prediction:
            return {'error': 'Could not generate prediction'}
        
        # Get additional data
        current_price = get_current_price(symbol)
        historical_data = get_historical_data(symbol, 30)
        
        return {
            'symbol': symbol,
            'model_type': model_type,
            'current_price': current_price,
            'prediction': prediction[0].__dict__,
            'historical_data': historical_data,
            'generated_at': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in get_stock_prediction: {e}")
        return {'error': str(e)}

def get_model_comparison(symbol: str, prediction_dates: List[str]) -> Dict:
    """Compare multiple models for the same symbol and dates"""
    try:
        # Validate inputs
        for date in prediction_dates:
            if not validate_prediction_date(date):
                return {'error': f'Invalid prediction date: {date}'}
        
        available_stocks = get_available_stocks()
        if symbol not in available_stocks:
            return {'error': f'Symbol {symbol} not available'}
        
        # Get predictions from all available models
        model_predictions = compare_models(symbol, prediction_dates)
        
        if not model_predictions:
            return {'error': 'No predictions generated'}
        
        # Format response
        response = {
            'symbol': symbol,
            'prediction_dates': prediction_dates,
            'current_price': get_current_price(symbol),
            'models': {}
        }
        
        for model_type, predictions in model_predictions.items():
            response['models'][model_type] = [p.__dict__ for p in predictions]
        
        return response
    
    except Exception as e:
        logger.error(f"Error in get_model_comparison: {e}")
        return {'error': str(e)}

def get_sentiment_analysis(symbol: str, sentiment_file: str) -> Dict:
    """Get sentiment analysis for a symbol"""
    try:
        # Load sentiment data
        sentiment_df = load_sentiment_data(sentiment_file)
        
        # Get company sentiment
        sentiment_data = get_company_sentiment(symbol, sentiment_df)
        
        return {
            'symbol': symbol,
            'company_sentiment': sentiment_data.company_sentiment,
            'market_sentiment': sentiment_data.market_sentiment,
            'sentiment_label': sentiment_data.sentiment_label,
            'headlines': sentiment_data.headlines,
            'wordcloud_image': sentiment_data.wordcloud_b64,
            'updated_at': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in get_sentiment_analysis: {e}")
        return {'error': str(e)}
