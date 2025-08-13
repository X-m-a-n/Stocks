import os
import json
import logging
from pathlib import Path
import polars as pl
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import logging
from data_loader import DataLoader

# Configure logging
logger = logging.getLogger(__name__)

class DataManager:
    """
    Core data manager that connects to all API endpoints.
    
    This class needs to provide data for:
    - /predict/{symbol} endpoint
    - /sentiment/{symbol} endpoint  
    - /metrics/{symbol} endpoint
    - /stocks endpoint
    """
    
    def __init__(self, stocks_df: pl.DataFrame, base_path: str = "./Models", sentiment_fileLoc = '/Data/joined_newspaper_data_v5.parquet'):
        self.base_path = Path(base_path)
        self.stocks_df = stocks_df
        self.cache = {}
        self.model_cache = {}
        self.sentiment_file = Path(sentiment_fileLoc)
        
        # Extract available stocks from both dataframe and model files
        df_stocks = set(self.stocks_df['symbol'].unique().to_list())
        model_stocks = set(self._detect_available_stocks())
        
        # Only include stocks that have both data and models
        self.available_stocks = sorted(list(df_stocks.intersection(model_stocks)))
        
        logger.info(f"Found {len(self.available_stocks)} stocks with both data and models")
        
        self._validate_dataframe()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directory structure"""
        directories: list[Path] = [
            self.base_path / "ARIMA",  
            self.base_path / "LSTM", 
            self.base_path / "data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _validate_dataframe(self):
        """Ensure dataframe has required columns"""
        required_columns = ['symbol', 'date', 'close']
        missing_columns = [col for col in required_columns if col not in self.stocks_df.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
        # UPDATED: Handle multiple date formats and convert to consistent Date type
        date_dtype = self.stocks_df['date'].dtype

        if date_dtype == pl.Datetime:
            # Convert datetime to date (remove time component)
            self.stocks_df = self.stocks_df.with_columns(
                pl.col('date').dt.date()
            )
        elif date_dtype == pl.Utf8:
            # Parse string dates to Date type
            try:
                self.stocks_df = self.stocks_df.with_columns(
                    pl.col('date').str.to_date()
                )
            except Exception as e:
                # If parsing fails, try different date formats
                try:
                    self.stocks_df = self.stocks_df.with_columns(
                        pl.col('date').str.to_date(format='%Y-%m-%d')
                    )
                except Exception:
                    try:
                        self.stocks_df = self.stocks_df.with_columns(
                            pl.col('date').str.to_date(format='%m/%d/%Y')
                        )
                    except Exception:
                        raise ValueError(f"Could not parse date column. Sample values: {self.stocks_df['date'].head(3).to_list()}")
        elif date_dtype == pl.Date:
            # Already correct type, no conversion needed
            pass
        else:
            # For any other type, try to cast to string first, then to date
            try:
                self.stocks_df = self.stocks_df.with_columns(
                    pl.col('date').cast(pl.Utf8).str.to_date()
                )
            except Exception as e:
                raise ValueError(f"Cannot convert date column from {date_dtype} to Date. Error: {e}")

        # Verify the conversion worked
        if self.stocks_df['date'].dtype != pl.Date:
            raise ValueError(f"Date column conversion failed. Final type: {self.stocks_df['date'].dtype}")

    async def test_system(self):
        """Test all major functions"""
        if not self.available_stocks:
            logger.error("No stocks available for testing")
            return False
        
        test_symbol = self.available_stocks[0]
        logger.info(f"Testing system with symbol: {test_symbol}")
        
        try:
            # Test data access
            current_price = await self.get_current_price(test_symbol)
            logger.info(f"✓ Current price: ${current_price}")
            
            historical_data = await self.get_historical_data(test_symbol, 30)
            logger.info(f"✓ Historical data: {len(historical_data)} records")
            
            # Test model loading and predictions
            available_models = await self.get_available_models(test_symbol)
            logger.info(f"✓ Available models: {available_models}")
            
            for model_type in available_models:
                predictions = await self.generate_predictions(test_symbol, model_type, 5)
                if predictions:
                    logger.info(f"✓ {model_type} predictions: {len(predictions)} days")
                else:
                    logger.warning(f"✗ {model_type} prediction failed")
            
            # Test metrics
            metrics = await self.get_model_metrics(test_symbol)
            logger.info(f"✓ Model metrics available: {list(metrics.keys())}")
            
            logger.info("✓ System test completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"✗ System test failed: {e}")
            return False
            
    # ================================
    # INITIALIZATION FUNCTIONS
    # ================================
    
    async def initialize(self):
        """Initialize the data manager"""
        logger.info("Initializing DataManager...")
        
        # Test that we can load models
        if self.available_stocks:
            test_symbol = self.available_stocks[0]
            available_models = await self.get_available_models(test_symbol)
            logger.info(f"Test symbol {test_symbol} has models: {available_models}")
            
            # Test loading one model
            if available_models:
                test_model = await self.load_model(test_symbol, available_models[0])
                if test_model:
                    logger.info(f"Successfully loaded {available_models[0]} model for {test_symbol}")
                else:
                    logger.warning(f"Failed to load test model")
        
        logger.info(f"DataManager initialized with {len(self.available_stocks)} stocks")
    
    # ================================
    # STOCK DATA FUNCTIONS
    # ================================
    async def get_available_stocks(self) -> List[str]:
        """
        [REQUIRED] Get list of available stock symbols
        
        Called by: GET /stocks endpoint
        Returns: List of stock symbols
        """
        return self.available_stocks


    async def get_current_price(self, symbol: str) -> float:
        """Get the most recent price from historical data"""
        try:
            latest_price = (
                self.stocks_df
                .filter(pl.col('symbol') == symbol)
                .sort('date', descending=True)
                .select('close')
                .head(1)
                .item()
            )
            return float(latest_price)
        
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None

    async def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical data from stored dataframe"""
        try:
            # Get the most recent date for this symbol
            max_date = (
                self.stocks_df
                .filter(pl.col('symbol') == symbol)
                .select(pl.col('date').max())
                .item()
            )
            
            # Calculate start date
            start_date = max_date - timedelta(days=days)
            
            # Filter and format data
            historical_data = (
                self.stocks_df
                .filter(
                    (pl.col('symbol') == symbol) & 
                    (pl.col('date') >= start_date) &
                    (pl.col('date') <= max_date)
                )
                .sort('date')
                .select(['date', 'close', 'volume'])  # volume is optional
            )
            
            # Convert to list of dicts
            result = []
            for row in historical_data.iter_rows(named=True):
                result.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'close': float(row['close']),
                    'volume': int(row.get('volume', 1000000))  # Default if no volume
                })
            
            return result
    
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    # ================================
    # MODEL MANAGEMENT FUNCTIONS
    # ================================
    async def get_available_models(self, symbol: str) -> List[str]:
        """Get list of available model types for a symbol"""
        available_models = []
        
        # Check ARIMA
        arima_path = self.base_path / "ARIMA" / f"{symbol}_arima_model.pkl"
        if arima_path.exists():
            available_models.append('arima')
        
        # Check LSTM
        lstm_path = self.base_path / "LSTM" / f"{symbol}_lstm_model.pkl"
        if lstm_path.exists():
            available_models.append('lstm')
        
        return available_models

    async def load_model(self, symbol: str, model_type: str) -> Optional[Dict]:
        """Load a trained model from storage with validation"""
        cache_key = f"{symbol}_{model_type}"
        
        # Check cache first
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            model_path = self.base_path / model_type.upper() / f"{symbol}_{model_type.lower()}_model.pkl"
            
            if model_path.exists():
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Ensure model has required structure
                if isinstance(model, dict):
                    model['symbol'] = symbol
                    model['model_type'] = model_type
                else:
                    # If model is just the fitted model object, wrap it
                    model = {
                        'fitted_model': model,
                        'symbol': symbol,
                        'model_type': model_type,
                        'created_at': datetime.now().isoformat()
                    }
                
                # Validate model for LSTM
                if model_type == 'lstm':
                    if not await self._validate_lstm_model(model):
                        logger.error(f"LSTM model validation failed for {symbol}")
                        return None
                
                self.model_cache[cache_key] = model
                logger.info(f"Loaded {model_type} model for {symbol}")
                return model
            else:
                logger.warning(f"Model file not found: {model_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {model_type} model for {symbol}: {e}")
            return None
    
    
    async def get_model_confidence(self, symbol: str, model_type: str) -> float:
        """
        [REQUIRED] Get model confidence score
        
        Called by: GET /predict/{symbol} endpoint
        Input: Stock symbol, model type
        Returns: Confidence score between 0.0 and 1.0
        
        TODO: Calculate based on your model metrics
        - Use R² score
        - Use validation accuracy
        - Use recent performance
        """
        model = await self.load_model(symbol, model_type)
        
        if not model or 'metrics' not in model:
            return 0.85  # Default confidence
        
        # Use R² score as confidence measure
        r2_score = model['metrics'].get('r2_score', 0.85)
        return min(0.99, max(0.5, r2_score))
    
    async def get_model_metrics(self, symbol: str) -> Dict[str, Dict]:
        """Get evaluation metrics for available models"""
        metrics = {}
        
        available_models = await self.get_available_models(symbol)
        
        for model_type in available_models:
            try:
                model = await self.load_model(symbol, model_type)
                if model and 'metrics' in model:
                    metrics[model_type] = model['metrics']
                else:
                    # If no metrics stored in model, calculate basic ones
                    metrics[model_type] = await self._calculate_basic_metrics(symbol, model_type)
            except Exception as e:
                logger.error(f"Failed to get metrics for {symbol} {model_type}: {e}")
                metrics[model_type] = {}
        
        return metrics

    async def get_validation_data(self, symbol: str) -> pl.DataFrame:
        """Get validation data (recent 30 days)"""
        try:
            validation_data = (
                self.stocks_df
                .filter(pl.col('symbol') == symbol)
                .sort('date')
                .tail(30)  # Last 30 days for validation
            )
            
            return validation_data
        
        except Exception as e:
            logger.error(f"Failed to get validation data for {symbol}: {e}")
            return pl.DataFrame()

    async def _calculate_basic_metrics(self, symbol: str, model_type: str) -> Dict:
        """Calculate basic metrics if not stored in model"""
        try:
            # Use validation data to calculate metrics
            validation_data = await self.get_validation_data(symbol)
            if len(validation_data) < 5:
                return {'error': 'Insufficient validation data'}
            
            # Generate predictions for validation period
            actual_prices = validation_data['close'].to_list()
            
            # This would need actual model predictions
            # For now, return placeholder
            return {
                'mae': 0.0,
                'mape': 0.0,
                'rmse': 0.0,
                'r2_score': 0.0,
                'note': 'Metrics calculation not implemented'
            }
        except Exception as e:
            return {'error': str(e)}

    def _detect_available_stocks(self) -> List[str]:
        """Detect available stocks from model files"""
        available_stocks = set()
        
        # Check both ARIMA and LSTM directories
        for model_type in ['ARIMA', 'LSTM']:
            model_dir = self.base_path / model_type
            if model_dir.exists():
                for model_file in model_dir.glob("*_model.pkl"):
                    # Extract symbol from filename (e.g., "1GS_arima_model.pkl" -> "1GS")
                    symbol = model_file.stem.split('_')[0]
                    available_stocks.add(symbol)
        
        return sorted(list(available_stocks))

    # PREDICTION FUNCTIONS
    
    async def generate_predictions(self, symbol: str, model_type: str, days: int) -> List[Dict]:
        """Generate price predictions using your trained models"""
        try:
            # Validate inputs
            if symbol not in self.available_stocks:
                raise ValueError(f"Symbol {symbol} not available")
            
            available_models = await self.get_available_models(symbol)
            if model_type not in available_models:
                raise ValueError(f"No {model_type} model available for {symbol}")
            
            model = await self.load_model(symbol, model_type)
            if not model:
                raise ValueError(f"Failed to load {model_type} model for {symbol}")
            
            current_price = await self.get_current_price(symbol)
            if not current_price:
                raise ValueError(f"Could not get current price for {symbol}")
            
            # Route to specific prediction function
            if model_type == 'arima':
                return await self._predict_arima(model, current_price, days)
            elif model_type == 'lstm':
                return await self._predict_lstm(model, current_price, days)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Prediction failed for {symbol} ({model_type}): {e}")
            return []
    
    async def _predict_arima(self, model: Dict, current_price: float, days: int) -> List[Dict]:
        """ARIMA prediction using historical data"""
        try:
            symbol = model['symbol']
            
            # Get recent historical data for context
            recent_data = await self.get_historical_data(symbol, 60)
            prices = [d['close'] for d in recent_data]
            
            if not model.get('fitted_model'):
                # If no pre-fitted model, create simple forecast
                return await self._simple_trend_forecast(symbol, current_price, days)
            
            # Use your fitted ARIMA model
            fitted_model = model['fitted_model']
            forecast = fitted_model.forecast(steps=days)
            
            predictions = []
            base_date = datetime.now()
            
            for i in range(days):
                pred_date = base_date + timedelta(days=i+1)
                pred_price = float(forecast[i]) if hasattr(forecast, '__getitem__') else float(forecast)
                
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_price': pred_price,
                    'confidence_interval_low': pred_price * 0.95,
                    'confidence_interval_high': pred_price * 1.05
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            return await self._simple_trend_forecast(model['symbol'], current_price, days)

    async def _simple_trend_forecast(self, symbol: str, current_price: float, days: int) -> List[Dict]:
        """Fallback: Simple trend-based forecast when no model available"""
        try:
            # Get recent price changes to estimate trend
            recent_data = await self.get_historical_data(symbol, 20)
            
            if len(recent_data) < 2:
                # No trend data available, assume flat
                daily_change = 0.0
            else:
                # Calculate average daily change
                prices = [d['close'] for d in recent_data]
                daily_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                daily_change = sum(daily_changes) / len(daily_changes)
            
            predictions = []
            price = current_price
            base_date = datetime.now()
            
            for i in range(1, days + 1):
                # Apply trend with some noise
                noise = np.random.normal(0, 0.01)  # 1% daily volatility
                price_change = daily_change + noise
                price = price * (1 + price_change)
                
                pred_date = base_date + timedelta(days=i)
                
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_price': round(price, 2),
                    'confidence_interval_low': round(price * 0.95, 2),
                    'confidence_interval_high': round(price * 1.05, 2)
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Simple forecast failed: {e}")
            return []
    
    async def _predict_lstm(self, model: Dict, current_price: float, days: int) -> List[Dict]:
        """LSTM model prediction logic"""
        try:
            symbol = model['symbol']
            
            # Check if we have a trained model
            if 'fitted_model' not in model:
                logger.warning(f"No fitted LSTM model found for {symbol}, using trend forecast")
                return await self._simple_trend_forecast(symbol, current_price, days)
            
            # Get the fitted model and any preprocessing components
            fitted_model = model['fitted_model']
            scaler = model.get('scaler')  # Price scaler if used during training
            
            # Get recent historical data for sequence input
            sequence_length = model.get('sequence_length', 60)  # Default to 60 days
            recent_data = await self.get_historical_data(symbol, sequence_length)
            
            if len(recent_data) < sequence_length:
                logger.warning(f"Insufficient data for LSTM prediction, using trend forecast")
                return await self._simple_trend_forecast(symbol, current_price, days)
            
            # Prepare input sequence
            prices = np.array([d['close'] for d in recent_data])
            
            # Apply scaling if scaler exists
            if scaler:
                prices_scaled = scaler.transform(prices.reshape(-1, 1)).flatten()
            else:
                # If no scaler, normalize prices to help with prediction stability
                prices_scaled = (prices - prices.mean()) / prices.std()
            
            # Reshape for model input (batch_size, sequence_length, features)
            input_sequence = prices_scaled.reshape(1, sequence_length, 1)
            
            predictions = []
            base_date = datetime.now()
            
            # Generate predictions iteratively
            current_sequence = input_sequence.copy()
            
            for i in range(days):
                try:
                    # Predict next value
                    pred_scaled = fitted_model.predict(current_sequence, verbose=0)
                    
                    # Handle different output shapes
                    if pred_scaled.ndim > 1:
                        pred_scaled = pred_scaled[0, 0] if pred_scaled.shape[1] == 1 else pred_scaled[0, -1]
                    else:
                        pred_scaled = pred_scaled[0]
                    
                    # Convert back to actual price
                    if scaler:
                        pred_price = scaler.inverse_transform([[pred_scaled]])[0, 0]
                    else:
                        # Reverse normalization
                        pred_price = (pred_scaled * prices.std()) + prices.mean()
                    
                    # Ensure reasonable price bounds
                    pred_price = max(pred_price, current_price * 0.5)  # Not less than 50% of current
                    pred_price = min(pred_price, current_price * 2.0)  # Not more than 200% of current
                    
                    pred_date = base_date + timedelta(days=i+1)
                    
                    # Calculate confidence intervals based on historical volatility
                    historical_volatility = np.std([d['close'] for d in recent_data[-20:]])  # Last 20 days
                    confidence_width = historical_volatility * 1.96  # 95% confidence interval
                    
                    predictions.append({
                        'date': pred_date.strftime('%Y-%m-%d'),
                        'predicted_price': round(float(pred_price), 2),
                        'confidence_interval_low': round(float(pred_price - confidence_width), 2),
                        'confidence_interval_high': round(float(pred_price + confidence_width), 2)
                    })
                    
                    # Update sequence for next prediction (rolling window)
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1, 0] = pred_scaled
                    
                    # Update current price for bounds checking
                    current_price = pred_price
                    
                except Exception as model_error:
                    logger.error(f"LSTM model prediction error at day {i+1}: {model_error}")
                    # Fallback to trend-based prediction for remaining days
                    remaining_days = days - i
                    if remaining_days > 0:
                        fallback_predictions = await self._simple_trend_forecast(symbol, current_price, remaining_days)
                        # Adjust dates to continue from where we left off
                        for j, pred in enumerate(fallback_predictions):
                            pred['date'] = (base_date + timedelta(days=i+1+j)).strftime('%Y-%m-%d')
                        predictions.extend(fallback_predictions)
                    break
            
            return predictions
        
        except Exception as e:
            logger.error(f"LSTM prediction failed for {symbol}: {e}")
            return await self._simple_trend_forecast(symbol, current_price, days)


    # Additional helper function for better error handling
    async def _validate_lstm_model(self, model: Dict) -> bool:
        """Validate LSTM model structure"""
        try:
            required_keys = ['fitted_model', 'symbol']
            missing_keys = [key for key in required_keys if key not in model]
            
            if missing_keys:
                logger.error(f"LSTM model missing required keys: {missing_keys}")
                return False
            
            fitted_model = model['fitted_model']
            
            # Check if model has predict method
            if not hasattr(fitted_model, 'predict'):
                logger.error("LSTM fitted_model does not have predict method")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"LSTM model validation failed: {e}")
            return False



    # ================================
    # SENTIMENT ANALYSIS FUNCTIONS
    # ================================
    
    async def get_sentiment_analysis(self, symbol: str) -> Optional[Dict]:
        """
        [REQUIRED] Get sentiment analysis for a stock
        
        Called by: GET /sentiment/{symbol} endpoint
        Input: Stock symbol
        Returns: {"symbol": str, "polarity": float, "subjectivity": float, "sentiment_label": str, "recent_headlines": List[str], "updated_at": str}
        """
        cache_key = f"sentiment_{symbol}"
        
        # Check cache (30-minute expiry)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=30):
                return cached_data
        
        sentiment_data = await self._analyze_news_sentiment(symbol)
        
        self.cache[cache_key] = (sentiment_data, datetime.now())
        return sentiment_data
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict:
        df: pl.DataFrame = pl.read_parquet(self.sentiment_file)

        df = df.with_columns([
            pl.when((pl.col("title_fin_polarity") != 0) & (pl.col("text_fin_polarity") != 0))
            .then((pl.col("title_fin_polarity") + pl.col("text_fin_polarity")) / 2)
            .otherwise(pl.col("text_fin_polarity"))
            .alias("avg_polarity"),

            pl.when((pl.col("title_fin_subjectivity") != 0) & (pl.col("text_fin_subjectivity") != 0))
            .then((pl.col("title_fin_subjectivity") + pl.col("text_fin_subjectivity")) / 2)
            .otherwise(pl.col("text_fin_subjectivity"))
            .alias("avg_subjectivity")
        ])

        drop_cols: list[str] = [
            'publication_date',
            'organizations',
            'sentiment_polarity',
            'sentiment_subjectivity',
            'title_polarity',
            'title_subjectivity',
            'title_fin_polarity',
            'title_fin_subjectivity',
            'text_fin_polarity',
            'text_fin_subjectivity',
        ]
        df = df.drop([col for col in drop_cols if col in df.columns])

        df = df.with_columns([
            pl.when(pl.col("avg_polarity") > 0.1)
            .then("Positive")
            .otherwise(
                pl.when(pl.col("avg_polarity") < -0.1)
                    .then("Negative")
                    .otherwise("Neutral")
            )
            .alias("sentiment_label")
        ])
        
        # TODO: Fetch real headlines
        df = df.sort(pl.col("date"))
        
        news: list[dict] = (
            df.filter(pl.col("symbol") == symbol)
            .select(["title", "sentiment_label", "date", "avg_subjectivity", "avg_polarity"])
            .tail(4)
            .to_dicts()
        )
        
        # Compute overall (mean) polarity and subjectivity from recent rows
        avg_polarity = sum(row["avg_polarity"] for row in news) / len(news)
        avg_subjectivity = sum(row["avg_subjectivity"] for row in news) / len(news)

        if avg_polarity > 0.1:
            overall_label = "Positive"
        elif avg_polarity < -0.1:
            overall_label = "Negative"
        else:
            overall_label = "Neutral"

        return {
            "symbol": symbol.upper(),
            "polarity": round(avg_polarity, 3),
            "subjectivity": round(avg_subjectivity, 3),
            "sentiment_label": overall_label,
            "recent_headlines": [item["title"] for item in news],
            "updated_at": datetime.now().isoformat()
        }
        
    # UTILITY FUNCTIONS    
    
    async def clear_cache(self):
        self.cache.clear()
        self.model_cache.clear()
        logger.info("All caches cleared")


"""
API ENDPOINT CONNECTIONS:
- GET /stocks → get_available_stocks()
- GET /predict/{symbol} → generate_predictions()
- GET /sentiment/{symbol} → get_sentiment_analysis()  
- GET /metrics/{symbol} → get_model_metrics()
- POST /models/{symbol}/retrain → retrain_model()
- DELETE /cache/clear → clear_cache()
"""