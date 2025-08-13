from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
from datetime import datetime, timedelta
import logging
from data import DataManager
import uvicorn

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="Stock Prediction API",
    description="Advanced stock price prediction API with ARIMA, LSTM, and Sentiment models",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Manager Instance ---
data_manager = DataManager()

# --- Pydantic Models ---
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
    mae: float
    mape: float
    rmse: float
    r2_score: float

class SentimentResponse(BaseModel):
    symbol: str
    polarity: float
    subjectivity: float
    sentiment_label: str
    recent_headlines: List[str]
    updated_at: str

# --- Background Tasks ---
@app.on_event("startup")
async def startup_event():
    """Initialize data manager on startup"""
    logger.info("Starting Stock Prediction API...")
    await data_manager.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Stock Prediction API...")
    await data_manager.cleanup()

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(data_manager.loaded_models),
        "version": "1.0.0"
    }

# --- Stock Data Endpoints ---
@app.get("/stocks")
async def get_available_stocks():
    """Get list of available stock symbols"""
    try:
        stocks = await data_manager.get_available_stocks()
        return {"stocks": stocks, "count": len(stocks)}
    except Exception as e:
        logger.error(f"Error fetching available stocks: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch available stocks")

# --- Prediction Endpoints ---
@app.get("/predict/{symbol}", response_model=PredictionResponse)
async def predict_stock_price(
    symbol: str,
    model: str = "arima",
    days: int = 7
):
    """Generate stock price predictions"""
    try:
        # Validate inputs
        symbol = symbol.upper()
        model = model.lower()
        
        if model not in ["arima", "lstm", "sentiment"]:
            raise HTTPException(status_code=400, detail="Invalid model. Choose: arima, lstm, or sentiment")
        
        if days < 1 or days > 30:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 30")
        
        logger.info(f"Generating {model} prediction for {symbol} ({days} days)")
        
        # Load model and generate predictions
        predictions = await data_manager.generate_predictions(symbol, model, days)
        
        if not predictions:
            raise HTTPException(status_code=404, detail=f"No model found for {symbol}")
        
        # Get historical data for context
        historical_data = await data_manager.get_historical_data(symbol, days=30)
        
        # Get current price
        current_price = await data_manager.get_current_price(symbol)
        
        return PredictionResponse(
            symbol=symbol,
            model=model,
            current_price=current_price,
            predictions=predictions,
            historical=historical_data,
            confidence=await data_manager.get_model_confidence(symbol, model),
            generated_at=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Model Metrics ---
@app.get("/metrics/{symbol}")
async def get_model_metrics(symbol: str) -> Dict[str, MetricsResponse]:
    """Get evaluation metrics for all models of a stock"""
    try:
        symbol = symbol.upper()
        logger.info(f"Fetching metrics for {symbol}")
        
        metrics = await data_manager.get_model_metrics(symbol)
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics found for {symbol}")
        
        return metrics
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Sentiment Analysis ---
@app.get("/sentiment/{symbol}", response_model=SentimentResponse)
async def get_sentiment_analysis(symbol: str):
    """Get sentiment analysis for a stock"""
    try:
        symbol = symbol.upper()
        logger.info(f"Fetching sentiment analysis for {symbol}")
        
        sentiment_data = await data_manager.get_sentiment_analysis(symbol)
        
        if not sentiment_data:
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {symbol}")
        
        return sentiment_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Batch Prediction ---
@app.post("/predict/batch")
async def batch_predict(
    symbols: List[str],
    model: str = "arima",
    days: int = 7
):
    """Generate predictions for multiple stocks"""
    try:
        if len(symbols) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed per batch")
        
        model = model.lower()
        if model not in ["arima", "lstm", "sentiment"]:
            raise HTTPException(status_code=400, detail="Invalid model")
        
        results = {}
        
        for symbol in symbols:
            try:
                symbol = symbol.upper()
                predictions = await data_manager.generate_predictions(symbol, model, days)
                current_price = await data_manager.get_current_price(symbol)
                
                results[symbol] = {
                    "predictions": predictions,
                    "current_price": current_price,
                    "status": "success"
                }
            except Exception as e:
                results[symbol] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        return {
            "results": results,
            "model": model,
            "days": days,
            "generated_at": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Model Management ---
@app.post("/models/{symbol}/retrain")
async def retrain_model(
    symbol: str,
    model: str,
    background_tasks: BackgroundTasks
):
    """Retrain a model for a specific stock"""
    try:
        symbol = symbol.upper()
        model = model.lower()
        
        if model not in ["arima", "lstm", "sentiment"]:
            raise HTTPException(status_code=400, detail="Invalid model")
        
        # Add background task for retraining
        background_tasks.add_task(data_manager.retrain_model, symbol, model)
        
        return {
            "message": f"Retraining {model} model for {symbol} started",
            "symbol": symbol,
            "model": model,
            "status": "in_progress"
        }
    
    except Exception as e:
        logger.error(f"Error starting retraining for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Cache Management ---
@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    try:
        await data_manager.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

# --- Development Server ---
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )