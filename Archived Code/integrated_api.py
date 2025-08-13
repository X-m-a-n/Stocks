from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
import uvicorn
import polars as pl
from pathlib import Path
import asyncio
import os

# Import your DataManager
from simplified_dashboard_data import DataManager
from data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="JSE Stock Prediction API",
    description="Advanced stock prediction API with ARIMA, LSTM, and Sentiment Analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data manager instance
data_manager: Optional[DataManager] = None

# Pydantic models
class PredictionPoint(BaseModel):
    date: str
    predicted_price: float
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None

class HistoricalPoint(BaseModel):
    date: str
    close: float
    volume: Optional[int] = None

class PredictionResponse(BaseModel):
    symbol: str
    model: str
    current_price: float
    predictions: List[PredictionPoint]
    historical: List[HistoricalPoint]
    confidence: float
    generated_at: str

class MetricsResponse(BaseModel):
    mae: Optional[float] = None
    mape: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None

class SentimentResponse(BaseModel):
    symbol: str
    polarity: float
    subjectivity: float
    sentiment_label: str
    recent_headlines: List[str]
    updated_at: str

class StocksResponse(BaseModel):
    stocks: List[str]
    count: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    data_manager_status: str
    available_stocks: int

# Dependency to get data manager
def get_data_manager() -> DataManager:
    """Dependency to get initialized data manager"""
    global data_manager
    if data_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="Data manager not initialized. Please check that your actual data files exist and are properly formatted."
        )
    return data_manager

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize data manager on startup"""
    global data_manager
    
    try:
        logger.info("🚀 Starting Stock Prediction API...")
        
        # Initialize DataLoader
        try:
            from data_loader import DataLoader
            loader = DataLoader()
            logger.info("✅ DataLoader initialized successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import DataLoader: {e}")
            logger.error("Please ensure data_loader.py is available")
            data_manager = None
            return
        except Exception as e:
            logger.error(f"❌ Failed to initialize DataLoader: {e}")
            data_manager = None
            return
        
        # Try loading from database first
        stocks_df = None
        try:
            logger.info("📊 Attempting to load stock data from database...")
            
            query = """
            SELECT DISTINCT
                trade_date as date,
                clean_symbol as symbol,
                daily_range_high,
                daily_range_low,
                closing_price as close,
                closing_price + price_change as open_price,
                volume
            FROM
                old_trade_data TD
                LEFT JOIN jse_database.instruments I on (TD.CLEAN_symbol = I.symbol AND TD.CURRENCY = I.CURRENCY)
            WHERE
                trade_date BETWEEN :start_date AND :end_date
                AND I.TYPE = 'ORDINARY'
                AND TD.CURRENCY = 'JMD'
            ORDER BY
                trade_date ASC,
                clean_symbol ASC
            """
            
            # Load stock data from database
            stocks_df: pl.DataFrame = loader.fetch_data(
                query=query,
                start_date='2017-01-01',
                end_date='2025-03-31'
            )
            
            if stocks_df is not None and len(stocks_df) > 0:
                logger.info(f"✅ Loaded {len(stocks_df)} records from database")
                logger.info(f"📈 Found {len(stocks_df['symbol'].unique())} unique symbols")
            else:
                logger.warning("⚠️ Database query returned no data, trying fallback...")
                stocks_df = None
                
        except Exception as e:
            logger.warning(f"⚠️ Database loading failed: {e}")
            logger.info("🔄 Falling back to file-based data loading...")
            stocks_df = None
        
        # Fallback to CSV/Parquet files if database fails
        if stocks_df is None:
            logger.info("📁 Attempting to load stock data from csv")
            
            stock_fileLoc = r"C:\Users\Joshh\Projects\Stocks\Data\stock_data_backup.csv"

            if Path(stock_fileLoc).exists():
                try:
                    stocks_df = pl.read_csv(stock_fileLoc)
                    logger.info(f"✅ Loaded {len(stocks_df)} records from {stock_fileLoc}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {stock_fileLoc}: {e}")

            # If still no data found
            if stocks_df is None:
                logger.error("❌ No valid stock data found in any location")
                return

        try:            
            # Validate data
            logger.info("🔍 Validating data...")
            
            # Validate we have actual data
            unique_symbols: list[str] = stocks_df['symbol'].unique().to_list()
            date_stats = stocks_df.select([
                pl.col('date').min().alias('min_date'), 
                pl.col('date').max().alias('max_date')
            ])
            min_date = date_stats['min_date'].item()
            max_date = date_stats['max_date'].item()
            
            logger.info(f"📊 Data validation successful:")
            logger.info(f"   - Records: {len(stocks_df):,}")
            logger.info(f"   - Symbols: {len(unique_symbols)}")
            logger.info(f"   - Date range: {min_date} to {max_date}")
            logger.info(f"   - Sample symbols: {unique_symbols[:10]}")
                
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            data_manager = None
            return
        
        # Initialize DataManager with validated data
        try:
            logger.info("🔧 Initializing DataManager...")
            
            data_manager = DataManager(
                stocks_df=stocks_df,
                base_path="./Models",
                sentiment_fileLoc="./Data/joined_newspaper_data_v5.parquet"
            )
            
            # Initialize and test the system
            await data_manager.initialize()
            
            # Run system test
            test_passed = await data_manager.test_system()
            if test_passed:
                logger.info("✅ DataManager initialized and tested successfully")
            else:
                logger.warning("⚠️ System test had some issues, but continuing...")
            
            # Get final stock count
            available_stocks = await data_manager.get_available_stocks()
            logger.info(f"🎯 API ready with {len(available_stocks)} available stocks")
            
            if len(available_stocks) > 0:
                logger.info(f"📈 Available stocks: {available_stocks[:10]}{'...' if len(available_stocks) > 10 else ''}")
            else:
                logger.warning("⚠️ No stocks available - check model files and data")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize DataManager: {e}")
            data_manager = None
            return
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error("Please check your database connection and data file paths")
        data_manager = None


# API Endpoints
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "JSE Stock Prediction API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "stocks": "/stocks", 
            "predictions": "/predict/{symbol}",
            "sentiment": "/sentiment/{symbol}",
            "metrics": "/metrics/{symbol}",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Comprehensive health check"""
    global data_manager
    
    if data_manager is None:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            data_manager_status="not_initialized",
            available_stocks=0
        )
    
    try:
        available_stocks = await data_manager.get_available_stocks()
        stock_count = len(available_stocks)
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            data_manager_status="initialized",
            available_stocks=stock_count
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            data_manager_status="error",
            available_stocks=0
        )

@app.get("/stocks", response_model=StocksResponse, tags=["Data"])
async def get_available_stocks(dm: DataManager = Depends(get_data_manager)):
    """Get list of available stock symbols"""
    try:
        stocks = await dm.get_available_stocks()
        
        logger.info(f"Returning {len(stocks)} available stocks")
        
        return StocksResponse(
            stocks=stocks,
            count=len(stocks)
        )
    
    except Exception as e:
        logger.error(f"Error fetching available stocks: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available stocks")

@app.get("/predict/{symbol}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_stock_price(
    symbol: str,
    model: str = "arima",
    days: int = 7,
    dm: DataManager = Depends(get_data_manager)
):
    """Generate stock price predictions using trained models"""
    try:
        # Validate inputs
        symbol = symbol.upper()
        model = model.lower()
        
        if model not in ["arima", "lstm", "sentiment"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid model. Choose: arima, lstm, or sentiment"
            )
        
        if days < 1 or days > 30:
            raise HTTPException(
                status_code=400, 
                detail="Days must be between 1 and 30"
            )
        
        available_stocks = await dm.get_available_stocks()
        if symbol not in available_stocks:
            raise HTTPException(
                status_code=404, 
                detail=f"Stock {symbol} not found. Available stocks: {available_stocks[:5]}..."
            )
        
        logger.info(f"Generating {model} prediction for {symbol} ({days} days)")
        
        # Generate predictions using DataManager
        predictions = await dm.generate_predictions(symbol, model, days)
        if not predictions:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate predictions for {symbol} using {model} model"
            )
        
        # Get additional data
        current_price = await dm.get_current_price(symbol)
        if current_price is None:
            raise HTTPException(
                status_code=500,
                detail=f"Could not retrieve current price for {symbol}"
            )
        
        historical_data = await dm.get_historical_data(symbol, days=30)
        confidence = await dm.get_model_confidence(symbol, model)
        
        return PredictionResponse(
            symbol=symbol,
            model=model,
            current_price=current_price,
            predictions=predictions,
            historical=historical_data,
            confidence=confidence,
            generated_at=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.get("/metrics/{symbol}", tags=["Analytics"])
async def get_model_metrics(
    symbol: str, 
    dm: DataManager = Depends(get_data_manager)
) -> Dict[str, MetricsResponse]:
    """Get evaluation metrics for all models of a stock"""
    try:
        symbol = symbol.upper()
        
        available_stocks = await dm.get_available_stocks()
        if symbol not in available_stocks:
            raise HTTPException(
                status_code=404, 
                detail=f"Stock {symbol} not found"
            )
        
        logger.info(f"Fetching metrics for {symbol}")
        
        metrics_data = await dm.get_model_metrics(symbol)
        
        # Convert to response format
        response = {}
        for model_type, metrics in metrics_data.items():
            response[model_type] = MetricsResponse(
                mae=metrics.get('mae'),
                mape=metrics.get('mape'),
                rmse=metrics.get('rmse'),
                r2_score=metrics.get('r2_score')
            )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sentiment/{symbol}", response_model=SentimentResponse, tags=["Analytics"])
async def get_sentiment_analysis(
    symbol: str, 
    dm: DataManager = Depends(get_data_manager)
):
    """Get sentiment analysis for a stock"""
    try:
        symbol = symbol.upper()
        
        available_stocks = await dm.get_available_stocks()
        if symbol not in available_stocks:
            raise HTTPException(
                status_code=404, 
                detail=f"Stock {symbol} not found"
            )
        
        logger.info(f"Fetching sentiment analysis for {symbol}")
        
        sentiment_data = await dm.get_sentiment_analysis(symbol)
        
        if sentiment_data is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze sentiment"
            )
        
        return SentimentResponse(**sentiment_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Cache management endpoints
@app.delete("/cache/clear", tags=["System"])
async def clear_cache(dm: DataManager = Depends(get_data_manager)):
    """Clear all cached data"""
    try:
        await dm.clear_cache()
        logger.info("Cache cleared successfully")
        return {"message": "Cache cleared successfully", "timestamp": datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

# System information endpoint
@app.get("/system/info", tags=["System"])
async def get_system_info(dm: DataManager = Depends(get_data_manager)):
    """Get system information and statistics"""
    try:
        available_stocks = await dm.get_available_stocks()
        
        # Get model availability for each stock
        model_stats = {"arima": 0, "lstm": 0, "total_models": 0}
        
        for stock in available_stocks[:5]:  # Check first 5 stocks to avoid long delays
            models = await dm.get_available_models(stock)
            model_stats["total_models"] += len(models)
            if "arima" in models:
                model_stats["arima"] += 1
            if "lstm" in models:
                model_stats["lstm"] += 1
        
        return {
            "api_version": "2.0.0",
            "available_stocks": len(available_stocks),
            "model_statistics": model_stats,
            "supported_models": ["arima", "lstm", "sentiment"],
            "max_prediction_days": 30,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

# Test endpoint for development
@app.get("/test", tags=["Development"])
async def test_endpoint():
    """Test endpoint to verify API functionality"""
    try:
        return {
            "message": "API is working!",
            "timestamp": datetime.now().isoformat(),
            "data_manager_status": "initialized" if data_manager else "not_initialized"
        }
    except Exception as e:
        return {
            "message": "API test failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🚀 Starting JSE Stock Prediction API...")
    print("📡 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    print("🧪 Test endpoint: http://localhost:8000/test")
    print("💹 Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "integrated_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )