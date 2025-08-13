"""
Simple FastAPI for LSTM Stock Predictions
Function-first approach with minimal classes
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import uvicorn
import polars as pl
from pathlib import Path
import os

# Import our simple data manager functions
from simple_data_manager import (
    initialize_data, get_available_stocks, get_stock_prediction,
    get_model_comparison, get_sentiment_analysis, get_current_price,
    get_historical_data, get_business_days, load_model_info
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LSTM Stock Prediction API",
    description="Simple API for LSTM stock predictions with sentiment analysis",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class PredictionRequest(BaseModel):
    symbol: str
    model_type: str
    prediction_date: str

class ComparisonRequest(BaseModel):
    symbol: str
    prediction_dates: List[str]

class SentimentRequest(BaseModel):
    symbol: str

class HealthResponse(BaseModel):
    status: str
    available_stocks: int
    timestamp: str

# Global variables for simple state management
_initialized = False
_available_stocks = []
_sentiment_file = None

# ================================
# INITIALIZATION
# ================================

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    global _initialized, _available_stocks, _sentiment_file
    
    try:
        logger.info("🚀 Starting LSTM Stock Prediction API...")
        
        # Load stock data
        stock_data_paths = [
            r"C:\Users\Joshh\Projects\Stocks\clean_stock_data.parquet",
            "clean_stock_data.parquet",
            r"C:\Users\Joshh\Projects\Stocks\Data\stocks_df.csv"
        ]
        
        stocks_df = None
        for path in stock_data_paths:
            if os.path.exists(path):
                logger.info(f"📊 Loading stock data from: {path}")
                try:
                    if path.endswith('.parquet'):
                        stocks_df = pl.read_parquet(path)
                    else:
                        stocks_df = pl.read_csv(path)
                    logger.info(f"✅ Loaded {len(stocks_df)} records")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue
        
        if stocks_df is None:
            logger.error("❌ No stock data found")
            return
        
        # Set sentiment file path
        sentiment_paths = [
            r"C:\Users\Joshh\Projects\Stocks\Data\joined_newspaper_data_v5.parquet",
            "Data/joined_newspaper_data_v5.parquet"
        ]
        
        for path in sentiment_paths:
            if os.path.exists(path):
                _sentiment_file = path
                logger.info(f"📰 Found sentiment data: {path}")
                break
        
        # Initialize data manager
        _available_stocks = initialize_data(stocks_df, models_dir="models", sentiment_file=_sentiment_file)
        
        if _available_stocks:
            _initialized = True
            logger.info(f"✅ API initialized with {len(_available_stocks)} stocks")
            logger.info(f"📈 Available stocks: {_available_stocks[:10]}{'...' if len(_available_stocks) > 10 else ''}")
        else:
            logger.warning("⚠️ No stocks with models found")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        _initialized = False

# ================================
# UTILITY ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LSTM Stock Prediction API",
        "version": "3.0.0",
        "status": "running" if _initialized else "not_initialized",
        "endpoints": {
            "health": "/health",
            "stocks": "/stocks",
            "predict": "/predict",
            "compare": "/compare",
            "sentiment": "/sentiment/{symbol}",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if _initialized else "unhealthy",
        available_stocks=len(_available_stocks),
        timestamp=datetime.now().isoformat()
    )

@app.get("/stocks")
async def get_stocks():
    """Get available stock symbols"""
    if not _initialized:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    # Load model info for additional details
    model_info = load_model_info()
    
    stock_details = []
    for symbol in _available_stocks:
        current_price = get_current_price(symbol)
        
        # Find models for this symbol
        available_models = []
        for key, info in model_info.items():
            if info.symbol == symbol:
                available_models.append({
                    'type': info.model_type,
                    'accuracy': round(info.accuracy, 3),
                    'mae': round(info.mae, 3)
                })
        
        stock_details.append({
            'symbol': symbol,
            'current_price': current_price,
            'available_models': available_models
        })
    
    return {
        'stocks': stock_details,
        'count': len(_available_stocks)
    }

# ================================
# PREDICTION ENDPOINTS
# ================================

@app.post("/predict")
async def predict_stock(request: PredictionRequest):
    """Generate single stock prediction"""
    if not _initialized:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    try:
        # Validate symbol