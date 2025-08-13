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

# Import updated data manager functions
from updated_data_manager import (
    initialize_data_manager,
    get_available_stocks_list,
    get_stock_predictions,
    get_sentiment_analysis, 
    get_model_metrics,
    clear_cache,
    get_next_business_days
)

# Configure it to use historical data
SIMULATION_DATE = "2024-10-01"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="JSE Stock Prediction API",
    description="LSTM & ARIMA Stock Prediction with Sentiment Analysis",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global initialization flag
_initialized = False

# Pydantic models
class PredictionPoint(BaseModel):
    date: str
    predicted_price: float
    confidence_low: float
    confidence_high: float

class HistoricalPoint(BaseModel):
    date: str
    close: float
    volume: int

class PredictionResponse(BaseModel):
    symbol: str
    model_type: str
    current_price: Optional[float]
    predictions: List[PredictionPoint]
    historical: List[HistoricalPoint]
    model_metrics: Dict
    confidence: float
    generated_at: str

class SentimentResponse(BaseModel):
    symbol: str
    company_sentiment: float
    market_sentiment: float
    sentiment_label: str
    recent_headlines: List[Dict]
    wordcloud_image: Optional[str]
    updated_at: str

class StocksResponse(BaseModel):
    stocks: List[str]
    count: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    available_stocks: int

# ================================
# INITIALIZATION
# ================================

def load_data_files():
    """Load stock and sentiment data files"""
    try:
        # Load stock data
        stock_file = "stocks_df_Sample.csv"  # Your stock data file
        if not os.path.exists(stock_file):
            # Try alternative paths
            stock_file = r"C:\Users\Joshh\Projects\Stocks\Data\clean_stock_data.parquet"
            if not os.path.exists(stock_file):
                stock_file = "clean_stock_data.parquet"
        
        if stock_file.endswith('.parquet'):
            stocks_df = pl.read_parquet(stock_file)
        else:
            stocks_df = pl.read_csv(stock_file)
        
        logger.info(f"Loaded {len(stocks_df)} stock records")
        
        # Load sentiment data
        sentiment_df = None
        sentiment_file = r"C:\Users\Joshh\Projects\Stocks\Data\sentiment_data_FINAL.parquet"
        
        if os.path.exists(sentiment_file):
            sentiment_df = pl.read_parquet(sentiment_file)
            logger.info(f"Loaded {len(sentiment_df)} sentiment records")
        else:
            logger.warning("Sentiment data file not found")
        
        return stocks_df, sentiment_df
    
    except Exception as e:
        logger.error(f"Error loading data files: {e}")
        return None, None

@app.on_event("startup")
async def startup_event():
    """Initialize data manager on startup"""
    global _initialized
    
    try:
        logger.info("🚀 Starting Stock Prediction API...")
        
        # Load data files
        stocks_df, sentiment_df = load_data_files()
        
        if stocks_df is None:
            logger.error("❌ Failed to load stock data")
            return
        
        # Initialize data manager
        available_stocks = initialize_data_manager(stocks_df, sentiment_df, "models")
        
        if len(available_stocks) > 0:
            logger.info(f"✅ API initialized with {len(available_stocks)} stocks")
            logger.info(f"📈 Available stocks: {available_stocks[:10]}{'...' if len(available_stocks) > 10 else ''}")
            _initialized = True
        else:
            logger.warning("⚠️ No stocks with trained models found")
    
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        _initialized = False

# ================================
# UTILITY FUNCTIONS
# ================================

def check_initialization():
    """Check if API is properly initialized"""
    if not _initialized:
        raise HTTPException(
            status_code=503,
            detail="API not initialized. Check data files and model directory."
        )

def get_current_date():
    """Get current date or simulation date"""
    if SIMULATION_DATE:
        return datetime.strptime(SIMULATION_DATE, '%Y-%m-%d').date()
    else:
        return datetime.now().date()

def validate_prediction_dates(dates: List[str]) -> List[str]:
    """Validate and filter prediction dates"""
    valid_dates = []
    today = get_current_date()
    
    for date_str in dates:
        try:
            pred_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Must be future date but not more than 30 days ahead
            if today < pred_date <= today + timedelta(days=30):
                valid_dates.append(date_str)
        except ValueError:
            continue
    
    return valid_dates

# ================================
# API ENDPOINTS
# ================================

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "JSE Stock Prediction API",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "stocks": "/stocks",
            "predict": "/predict/{symbol}",
            "sentiment": "/sentiment/{symbol}",
            "metrics": "/metrics/{symbol}",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        if not _initialized:
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                version="3.0.0",
                available_stocks=0
            )
        
        stocks = await get_available_stocks_list()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="3.0.0",
            available_stocks=len(stocks)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
            version="3.0.0",
            available_stocks=0
        )

@app.get("/stocks", response_model=StocksResponse, tags=["Data"])
async def get_stocks():
    """Get list of available stock symbols"""
    check_initialization()
    
    try:
        stocks = await get_available_stocks_list()
        
        return StocksResponse(
            stocks=stocks,
            count=len(stocks)
        )
    
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available stocks")

@app.get("/predict/{symbol}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_stock_price(
    symbol: str,
    model: str = "lstm",
    dates: Optional[str] = None,
    days: int = 7
):
    """
    Generate stock price predictions
    
    Parameters:
    - symbol: Stock symbol
    - model: Model type ('lstm' or 'arima')
    - dates: Comma-separated prediction dates (YYYY-MM-DD) or None for auto-generated dates
    - days: Number of days to predict (if dates not provided)
    """
    check_initialization()
    
    try:
        symbol = symbol.upper()
        model = model.lower()
        
        # Validate model type
        if model not in ["lstm", "arima"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model type. Choose: 'lstm' or 'arima'"
            )
        
        # Validate symbol
        available_stocks = await get_available_stocks_list()
        if symbol not in available_stocks:
            raise HTTPException(
                status_code=404,
                detail=f"Stock {symbol} not found. Available: {available_stocks[:5]}..."
            )
        
        # Handle prediction dates
        if dates:
            # Parse provided dates
            date_list = [d.strip() for d in dates.split(',')]
            prediction_dates = validate_prediction_dates(date_list)
            
            if not prediction_dates:
                raise HTTPException(
                    status_code=400,
                    detail="No valid prediction dates provided"
                )
        else:
            # Generate next business days
            today = get_current_date()
            prediction_dates = get_next_business_days(today, min(days, 30))
        
        logger.info(f"Generating {model} predictions for {symbol} on dates: {prediction_dates}")
        
        # Generate predictions with specified model
        result = await get_stock_predictions(symbol, prediction_dates, model)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return PredictionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sentiment/{symbol}", response_model=SentimentResponse, tags=["Sentiment"])
async def get_stock_sentiment(symbol: str):
    """Get sentiment analysis for a stock"""
    check_initialization()
    
    try:
        symbol = symbol.upper()
        
        # Validate symbol
        available_stocks = await get_available_stocks_list()
        if symbol not in available_stocks:
            raise HTTPException(
                status_code=404,
                detail=f"Stock {symbol} not found"
            )
        
        logger.info(f"Getting sentiment analysis for {symbol}")
        
        result = await get_sentiment_analysis(symbol)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return SentimentResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics/{symbol}", tags=["Metrics"])
async def get_stock_metrics(symbol: str):
    """Get model performance metrics for a stock"""
    check_initialization()
    
    try:
        symbol = symbol.upper()
        
        # Validate symbol
        available_stocks = await get_available_stocks_list()
        if symbol not in available_stocks:
            raise HTTPException(
                status_code=404,
                detail=f"Stock {symbol} not found"
            )
        
        logger.info(f"Getting metrics for {symbol}")
        
        result = await get_model_metrics(symbol)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/cache/clear", tags=["System"])
async def clear_all_cache():
    """Clear all cached data"""
    check_initialization()
    
    try:
        await clear_cache()
        return {
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@app.get("/system/info", tags=["System"])
async def get_system_info():
    """Get system information and statistics"""
    check_initialization()
    
    try:
        stocks = await get_available_stocks_list()
        
        return {
            "api_version": "1.0.0",
            "available_stocks": len(stocks),
            "model_types": ["PyTorch LSTM with Attention", "ARIMA Time Series"],
            "max_prediction_days": 30,
            "features": "Technical indicators + Sentiment analysis",
            "sample_stocks": stocks[:10] if stocks else [],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

@app.get("/test", tags=["Development"])
async def test_endpoint():
    """Test endpoint to verify API functionality"""
    return {
        "message": "API is working!",
        "initialized": _initialized,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting JSE Stock Prediction API...")
    print("📡 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    print("🧪 Test endpoint: http://localhost:8000/test")
    print("💹 Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "updated_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
