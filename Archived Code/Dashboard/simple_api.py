from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Simple Stock Prediction API",
    description="Simplified stock prediction API for testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample stock symbols
SAMPLE_STOCKS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "META", "NVDA", "BRK-B", "UNH", "JNJ"
]

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

# Helper functions
def generate_dummy_price(symbol: str, base_price: float = None) -> float:
    """Generate a realistic stock price"""
    if base_price is None:
        # Set different base prices for different stocks
        base_prices: dict[str, float] = {
            "AAPL": 175.0, "MSFT": 380.0, "GOOGL": 140.0,
            "AMZN": 155.0, "TSLA": 210.0, "META": 350.0,
            "NVDA": 480.0, "BRK-B": 350.0, "UNH": 520.0, "JNJ": 160.0
        }
        base_price = base_prices.get(symbol, 150.0)
    
    # Add random variation
    return round(base_price * np.random.uniform(0.95, 1.05), 2)

def generate_historical_data(symbol: str, days: int = 30) -> List[Dict]:
    """Generate dummy historical data"""
    historical_data = []
    base_price = generate_dummy_price(symbol)
    base_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        # Random walk for price
        base_price *= np.random.uniform(0.98, 1.02)
        
        historical_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'close': round(base_price, 2),
            'volume': int(np.random.uniform(1000000, 10000000))
        })
    
    return historical_data

def generate_predictions(symbol: str, model: str, days: int, current_price: float) -> List[Dict]:
    """Generate dummy predictions"""
    predictions = []
    price = current_price
    base_date = datetime.now()
    
    # Different volatility for different models
    volatility = {
        'arima': 0.01,
        'lstm': 0.02,
        'sentiment': 0.025
    }
    
    vol = volatility.get(model, 0.015)
    
    for i in range(1, days + 1):
        # Generate price change
        change = np.random.normal(0, vol)
        price = price * (1 + change)
        
        pred_date = base_date + timedelta(days=i)
        
        predictions.append({
            'date': pred_date.strftime('%Y-%m-%d'),
            'predicted_price': round(price, 2),
            'confidence_interval_low': round(price * 0.92, 2),
            'confidence_interval_high': round(price * 1.08, 2)
        })
    
    return predictions

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple Stock Prediction API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/stocks")
async def get_available_stocks():
    """Get list of available stock symbols"""
    return {
        "stocks": SAMPLE_STOCKS,
        "count": len(SAMPLE_STOCKS)
    }

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
        
        if symbol not in SAMPLE_STOCKS:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        logger.info(f"Generating {model} prediction for {symbol} ({days} days)")
        
        # Generate data
        current_price = generate_dummy_price(symbol)
        predictions = generate_predictions(symbol, model, days, current_price)
        historical_data = generate_historical_data(symbol, days=30)
        
        # Model confidence varies by type
        confidence_map = {'arima': 0.85, 'lstm': 0.92, 'sentiment': 0.78}
        confidence = confidence_map.get(model, 0.85)
        
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
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics/{symbol}")
async def get_model_metrics(symbol: str) -> Dict[str, MetricsResponse]:
    """Get evaluation metrics for all models of a stock"""
    try:
        symbol = symbol.upper()
        
        if symbol not in SAMPLE_STOCKS:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        logger.info(f"Fetching metrics for {symbol}")
        
        # Generate dummy metrics
        metrics = {}
        for model in ['arima', 'lstm', 'sentiment']:
            metrics[model] = {
                'mae': round(np.random.uniform(0.5, 2.0), 4),
                'mape': round(np.random.uniform(2.0, 8.0), 2),
                'rmse': round(np.random.uniform(1.0, 3.0), 4),
                'r2_score': round(np.random.uniform(0.7, 0.95), 4)
            }
        
        return metrics
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sentiment/{symbol}", response_model=SentimentResponse)
async def get_sentiment_analysis(symbol: str):
    """Get sentiment analysis for a stock"""
    try:
        symbol = symbol.upper()
        
        if symbol not in SAMPLE_STOCKS:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        logger.info(f"Fetching sentiment analysis for {symbol}")
        
        # Generate dummy sentiment
        polarity = np.random.uniform(-0.5, 0.5)
        subjectivity = np.random.uniform(0.3, 0.8)
        
        if polarity > 0.1:
            sentiment_label = "Positive"
        elif polarity < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        headlines = [
            f"{symbol} shows strong quarterly performance",
            f"Market analysts upgrade {symbol} rating",
            f"{symbol} announces new product launch",
            f"Investors optimistic about {symbol} future",
            f"{symbol} reports better than expected earnings"
        ]
        
        selected_headlines = np.random.choice(headlines, size=3, replace=False).tolist()
        
        return SentimentResponse(
            symbol=symbol,
            polarity=round(polarity, 3),
            subjectivity=round(subjectivity, 3),
            sentiment_label=sentiment_label,
            recent_headlines=selected_headlines,
            updated_at=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Test endpoint
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "API is working!",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Simple Stock Prediction API...")
    print("📡 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    print("🧪 Test endpoint: http://localhost:8000/test")
    
    uvicorn.run(
        "simple_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )