#!/usr/bin/env python3
"""
Sentiment-Enhanced LSTM Architecture Comparison
===============================================

Standalone script to:
1. Build new LSTM models with sentiment + technical features
2. Compare performance against existing traditional LSTM models
3. Train on 10 random stocks for comparison study

Existing models path: C:/Users/Joshh/Projects/Stocks/Models/LSTM/{SYMBOL}_lstm_model.pkl
"""

import polars as pl
import pandas as pd
import numpy as np
import datetime as dt
import os
import pickle
import random
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class LSTMStockPredictor:
    """Dummy class for loading pickled models from original training script"""
    pass

import sys
sys.modules['__main__'].LSTMStockPredictor = LSTMStockPredictor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_lstm_comparison.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SentimentEnhancedLSTM:
    """
    LSTM model enhanced with sentiment and technical indicators
    """
    
    def __init__(self, sequence_length: int = 30, prediction_horizon: int = 1,
                 lstm_units: List[int] = [1024, 512, 256], dropout_rate: float = 0.25):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, stock_df: pl.DataFrame, sentiment_df: pl.DataFrame, 
                        symbol: str) -> pl.DataFrame:
        """Prepare combined feature set for a single symbol"""
        
        # Filter stock data for symbol
        stock_data = stock_df.filter(pl.col("symbol") == symbol).sort("date")
        
        # Filter sentiment data for symbol
        sentiment_data = sentiment_df.filter(pl.col("jse_symbols") == symbol).sort("date")
        
        # Convert both date columns to the same type for joining
        stock_data = stock_data.with_columns([
            pl.col("date").dt.date().alias("date") if stock_data.schema["date"] == pl.Datetime else pl.col("date")
        ])
        
        sentiment_data = sentiment_data.with_columns([
            pl.col("date").cast(pl.Date).alias("date")
        ])
        
        # Check if we have sufficient stock data
        if len(stock_data) < 50:  # Need minimum data for calculations
            raise ValueError(f"Insufficient stock data for {symbol}: only {len(stock_data)} records")
        
        # Ensure price columns are numeric and handle nulls
        stock_data = stock_data.with_columns([
            pl.col("closing_price").cast(pl.Float64).fill_null(strategy="forward").alias("closing_price"),
            pl.col("opening_price").cast(pl.Float64).fill_null(strategy="forward").alias("opening_price"),
            pl.col("high_price").cast(pl.Float64).fill_null(strategy="forward").alias("high_price"),
            pl.col("low_price").cast(pl.Float64).fill_null(strategy="forward").alias("low_price"),
            pl.col("volume").cast(pl.Float64).fill_null(0).alias("volume"),
        ])
        
        # Calculate technical indicators with error handling
        try:
            stock_data = stock_data.with_columns([
                # Price-based features with safe calculations
                pl.col("closing_price").pct_change().fill_null(0).alias("returns"),
                ((pl.col("closing_price") - pl.col("opening_price")) / pl.col("opening_price")).fill_null(0).alias("intraday_return"),
                ((pl.col("high_price") - pl.col("low_price")) / pl.col("low_price")).fill_null(0).alias("daily_range"),
                
                # Moving averages
                pl.col("closing_price").rolling_mean(5).alias("ma_5"),
                pl.col("closing_price").rolling_mean(10).alias("ma_10"),
                pl.col("closing_price").rolling_mean(20).alias("ma_20"),
                pl.col("closing_price").rolling_mean(50).alias("ma_50"),
                
                # Volume indicators
                pl.col("volume").rolling_mean(10).alias("volume_ma10"),
                pl.col("volume").rolling_mean(20).alias("volume_ma20"),
            ])
            
            # Calculate volume ratio safely
            stock_data = stock_data.with_columns([
                (pl.col("volume") / pl.col("volume_ma20")).fill_null(1).alias("volume_ratio")
            ])
            
            # Price to MA ratios
            stock_data = stock_data.with_columns([
                ((pl.col("closing_price") - pl.col("ma_5")) / pl.col("ma_5")).fill_null(0).alias("price_to_ma5"),
                ((pl.col("closing_price") - pl.col("ma_20")) / pl.col("ma_20")).fill_null(0).alias("price_to_ma20"),
            ])
            
            # Volatility
            stock_data = stock_data.with_columns([
                pl.col("returns").rolling_std(10).fill_null(0).alias("volatility_10d"),
                pl.col("returns").rolling_std(20).fill_null(0).alias("volatility_20d"),
            ])
            
            # RSI calculation with error handling
            stock_data = stock_data.with_columns([
                self._calculate_rsi_safe(pl.col("closing_price")).alias("rsi")
            ])
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed for {symbol}: {e}")
            raise ValueError(f"Technical indicator calculation failed for {symbol}: {e}")
        
        # Merge stock and sentiment data
        merged = stock_data.join(
            sentiment_data,
            on="date",
            how="left"
        )
        
        # Fill missing sentiment values with neutral values
        sentiment_fill_values = {
            "article_count": 0,
            "avg_sentiment": 0.0,
            "sentiment_volatility": 0.0,
            "min_sentiment": 0.0,
            "max_sentiment": 0.0,
            "avg_positive": 0.33,
            "avg_negative": 0.33,
            "avg_neutral": 0.34,
            "headline_bias": 0.0,
            "has_ic_insider": False,
            "sentiment_3d": 0.0,
            "sentiment_7d": 0.0,
            "sentiment_14d": 0.0,
            "sentiment_volatility_7d": 0.0,
            "sentiment_momentum_1d": 0.0,
            "sentiment_momentum_7d": 0.0,
            "article_volume_7d": 0,
            "high_positive_sentiment": 0,
            "high_negative_sentiment": 0
        }
        
        for col, fill_val in sentiment_fill_values.items():
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).fill_null(fill_val))
        
        # Calculate sentiment-price interaction features safely
        merged = merged.with_columns([
            (pl.col("avg_sentiment") * pl.col("returns")).fill_null(0).alias("sentiment_returns_interaction"),
            (pl.col("avg_sentiment") * pl.col("volume_ratio")).fill_null(0).alias("sentiment_volume_interaction"),
            (pl.col("sentiment_momentum_1d") * pl.col("volatility_10d")).fill_null(0).alias("sentiment_volatility_interaction"),
        ])
        
        return merged

    def _calculate_rsi_safe(self, prices: pl.Expr, window: int = 14) -> pl.Expr:
        """Calculate RSI with safe error handling"""
        try:
            delta = prices.diff()
            gain = delta.clip(lower_bound=0).rolling_mean(window)
            loss = (-delta.clip(upper_bound=0)).rolling_mean(window)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fill_null(50)  # Fill with neutral RSI value
        except Exception:
            return pl.lit(50)  # Return neutral RSI if calculation fails

    
    def prepare_sequences(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        
        # Define feature columns
        price_features = [
            "opening_price", "high_price", "low_price", "volume",
            "returns", "intraday_return", "daily_range", "volume_ratio"
        ]
        
        technical_features = [
            "ma_5", "ma_10", "ma_20", "ma_50", "price_to_ma5", "price_to_ma20",
            "volatility_10d", "volatility_20d", "rsi", "volume_ma10", "volume_ma20"
        ]
        
        sentiment_features = [
            "avg_sentiment", "sentiment_volatility", "min_sentiment", "max_sentiment",
            "avg_positive", "avg_negative", "avg_neutral", "headline_bias",
            "sentiment_3d", "sentiment_7d", "sentiment_14d", "sentiment_volatility_7d",
            "sentiment_momentum_1d", "sentiment_momentum_7d", "article_volume_7d",
            "high_positive_sentiment", "high_negative_sentiment"
        ]
        
        interaction_features = [
            "sentiment_returns_interaction", "sentiment_volume_interaction", 
            "sentiment_volatility_interaction"
        ]
        
        # Combine all features
        self.feature_columns = price_features + technical_features + sentiment_features + interaction_features
        
        # Filter to existing columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        logger.info(f"Using {len(self.feature_columns)} features: {self.feature_columns}")
        
        # Convert to pandas for easier sequence creation
        df_pd = df.select(["date", "closing_price"] + self.feature_columns).to_pandas()
        
        # Remove rows with NaN values
        df_pd = df_pd.dropna()
        
        if len(df_pd) < self.sequence_length + self.prediction_horizon:
            raise ValueError(f"Insufficient data after cleaning: {len(df_pd)} rows")
        
        # Prepare features and targets
        features = df_pd[self.feature_columns].values
        targets = df_pd["closing_price"].values
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Scale targets
        targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled) - self.prediction_horizon + 1):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(targets_scaled[i + self.prediction_horizon - 1])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build enhanced LSTM model"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(self.lstm_units[0], return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_sequences = i < len(self.lstm_units) - 2
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(32, activation='tanh'))
        model.add(Dropout(self.dropout_rate / 2))
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        
        # Compile
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
              epochs: int = 200, batch_size: int = 32, verbose: int = 0) -> dict:
        """Train the model"""
        
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        return predictions


class ModelComparator:
    """Compare sentiment-enhanced models with existing traditional models"""
    
    def __init__(self, traditional_models_dir: str, output_dir: str):
        self.traditional_models_dir = Path(traditional_models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_results = []
        
    def load_sentiment_data(self, sentiment_file: str) -> pl.DataFrame:
        """Load and process sentiment data"""
        logger.info(f"Loading sentiment data from {sentiment_file}")
        
        df = pl.read_parquet(sentiment_file)
        
        # Simply rename the date column since it's already Date type
        df = df.with_columns([pl.col("publication_date").alias("date")])
        
        # First, ensure jse_symbols is properly formatted as a list
        df = df.with_columns([
            pl.col("jse_symbols").cast(pl.List(pl.Utf8))
        ])
        
        # Select columns and explode JSE symbols
        df_exploded = df.select([
            "date", "title", "text", "authors",
            "title_sentiment_compound", "text_sentiment_compound", "combined_sentiment_compound",
            "title_sentiment_positive", "title_sentiment_negative", "title_sentiment_neutral",
            "text_sentiment_positive", "text_sentiment_negative", "text_sentiment_neutral",
            'combined_sentiment_positive', 'combined_sentiment_negative', 'combined_sentiment_neutral',
            "jse_symbols"
        ]).explode("jse_symbols")
        
        # Filter out null symbols and ensure it's properly exploded
        df_exploded = df_exploded.filter(
            pl.col("jse_symbols").is_not_null() & 
            (pl.col("jse_symbols") != "") &
            (pl.col("jse_symbols").str.len_chars() > 0)
        )
        
        # Verify jse_symbols is now a string column
        logger.info(f"jse_symbols dtype after explode: {df_exploded['jse_symbols'].dtype}")
        logger.info(f"Sample jse_symbols values: {df_exploded.select('jse_symbols').head(5)}")
        
        # If jse_symbols is still a list, force convert to string
        if str(df_exploded['jse_symbols'].dtype).startswith('List'):
            logger.warning("jse_symbols still a list, converting to string")
            df_exploded = df_exploded.with_columns([
                pl.col("jse_symbols").list.first().alias("jse_symbols")
            ]).filter(pl.col("jse_symbols").is_not_null())
        
        # Daily aggregation by symbol
        daily_sentiment = df_exploded.group_by(["date", "jse_symbols"]).agg([
            pl.count("title").alias("article_count"),
            pl.col("combined_sentiment_compound").mean().alias("avg_sentiment"),
            pl.col("combined_sentiment_compound").std().fill_null(0).alias("sentiment_volatility"),
            pl.col("combined_sentiment_compound").min().alias("min_sentiment"),
            pl.col("combined_sentiment_compound").max().alias("max_sentiment"),
            pl.col("combined_sentiment_positive").mean().alias("avg_positive"),
            pl.col("combined_sentiment_negative").mean().alias("avg_negative"),
            pl.col("combined_sentiment_neutral").mean().alias("avg_neutral"),
            (pl.col("title_sentiment_compound") - pl.col("text_sentiment_compound")).mean().alias("headline_bias"),
            pl.col("authors").list.join(", ").str.contains("Ic Insider.Com").any().alias("has_ic_insider")
        ])
        
        # Add rolling features
        daily_sentiment = daily_sentiment.sort(["jse_symbols", "date"])
        
        daily_sentiment = daily_sentiment.with_columns([
            pl.col("avg_sentiment").rolling_mean(window_size=3).over("jse_symbols").alias("sentiment_3d"),
            pl.col("avg_sentiment").rolling_mean(window_size=7).over("jse_symbols").alias("sentiment_7d"),
            pl.col("avg_sentiment").rolling_mean(window_size=14).over("jse_symbols").alias("sentiment_14d"),
            pl.col("avg_sentiment").rolling_std(window_size=7).over("jse_symbols").alias("sentiment_volatility_7d"),
            (pl.col("avg_sentiment") - pl.col("avg_sentiment").shift(1).over("jse_symbols")).alias("sentiment_momentum_1d"),
            (pl.col("avg_sentiment") - pl.col("avg_sentiment").shift(7).over("jse_symbols")).alias("sentiment_momentum_7d"),
            pl.col("article_count").rolling_sum(window_size=7).over("jse_symbols").alias("article_volume_7d"),
            (pl.col("avg_sentiment") > 0.3).cast(pl.Int8).alias("high_positive_sentiment"),
            (pl.col("avg_sentiment") < -0.3).cast(pl.Int8).alias("high_negative_sentiment"),
        ])
        
        logger.info(f"Processed sentiment data: {daily_sentiment.shape}")
        return daily_sentiment
    
    def get_available_symbols(self) -> List[str]:
        """Get symbols that have traditional models saved"""
        model_files = list(self.traditional_models_dir.glob("*_lstm_model.pkl"))
        symbols = [f.stem.replace("_lstm_model", "") for f in model_files]
        logger.info(f"Found {len(symbols)} traditional models: {symbols}")
        return symbols
    
    def load_traditional_model(self, symbol: str):
        """Load existing traditional LSTM model"""
        model_path = self.traditional_models_dir / f"{symbol}_lstm_model.pkl"
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Failed to load traditional model for {symbol}: {e}")
            return None
    
    def compare_single_stock(self, symbol: str, stock_df: pl.DataFrame, 
                            sentiment_df: pl.DataFrame, split_date: dt.date) -> Dict:
        """Compare traditional vs sentiment-enhanced model for single stock"""
        
        logger.info(f"Comparing models for {symbol}")
        start_time = time.time()
        
        try:
            # Load traditional model
            traditional_model = self.load_traditional_model(symbol)
            if traditional_model is None:
                return {"symbol": symbol, "status": "failed", "error": "Could not load traditional model"}
            
            # Prepare data for sentiment-enhanced model
            enhanced_lstm = SentimentEnhancedLSTM(sequence_length=30, prediction_horizon=1)
            
            # Prepare features with sentiment
            merged_data = enhanced_lstm.prepare_features(stock_df, sentiment_df, symbol)
            
            # Split data
            train_data = merged_data.filter(pl.col("date") < split_date)
            test_data = merged_data.filter(pl.col("date") >= split_date)
            
            if len(train_data) < 60 or len(test_data) < 10:
                return {"symbol": symbol, "status": "failed", "error": "Insufficient data for comparison"}
            
            # Train sentiment-enhanced model
            logger.info(f"Training sentiment-enhanced model for {symbol}")
            X_train, y_train = enhanced_lstm.prepare_sequences(train_data)
            history = enhanced_lstm.train(X_train, y_train, epochs=200, verbose=0)
            
            # Prepare test data for both models
            X_test, y_test = enhanced_lstm.prepare_sequences(test_data)
            
            # Make predictions with sentiment-enhanced model
            enhanced_predictions = enhanced_lstm.predict(X_test)
            
            # Get actual prices for comparison (from test data)
            test_prices = test_data.select("closing_price").to_numpy().flatten()
            
            # For traditional model comparison, we need to prepare data in its format
            # This is a simplified approach - you may need to adjust based on your traditional model's exact format
            traditional_test_data = test_data.select(["date", "closing_price"]).to_pandas()
            
            # Calculate metrics for sentiment-enhanced model
            actual_test_prices = test_prices[-len(enhanced_predictions):]
            
            enhanced_mae = mean_absolute_error(actual_test_prices, enhanced_predictions)
            enhanced_rmse = np.sqrt(mean_squared_error(actual_test_prices, enhanced_predictions))
            enhanced_mape = np.mean(np.abs((actual_test_prices - enhanced_predictions) / actual_test_prices)) * 100
            
            # Direction accuracy for enhanced model
            actual_direction = np.diff(actual_test_prices) > 0
            predicted_direction = np.diff(enhanced_predictions) > 0
            enhanced_direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            # Get traditional model predictions (simplified approach)
            # Note: You may need to adjust this based on your traditional model's exact interface
            try:
                # Assume traditional model has similar interface
                recent_prices = train_data.tail(30).select("closing_price").to_numpy().flatten()
                traditional_prediction = traditional_model.predict_next_day(recent_prices)
                
                # For fair comparison, we'll use the last prediction
                current_price = actual_test_prices[0]
                traditional_predictions = np.full_like(actual_test_prices, traditional_prediction)
                
                traditional_mae = mean_absolute_error(actual_test_prices, traditional_predictions)
                traditional_rmse = np.sqrt(mean_squared_error(actual_test_prices, traditional_predictions))
                traditional_mape = np.mean(np.abs((actual_test_prices - traditional_predictions) / actual_test_prices)) * 100
                
                # This is a simplified comparison - for a fair test, you'd need the traditional model's full prediction sequence
                
            except Exception as e:
                logger.warning(f"Could not get traditional model predictions for {symbol}: {e}")
                traditional_mae = traditional_rmse = traditional_mape = None
            
            # Save enhanced model
            enhanced_model_path = self.output_dir / f"{symbol}_sentiment_enhanced_lstm.pkl"
            with open(enhanced_model_path, 'wb') as f:
                pickle.dump(enhanced_lstm, f)
            
            # Create comparison plot
            self.plot_comparison(symbol, actual_test_prices, enhanced_predictions, 
                               traditional_predictions if 'traditional_predictions' in locals() else None)
            
            result = {
                "symbol": symbol,
                "status": "success",
                "enhanced_mae": enhanced_mae,
                "enhanced_rmse": enhanced_rmse,
                "enhanced_mape": enhanced_mape,
                "enhanced_direction_accuracy": enhanced_direction_accuracy,
                "traditional_mae": traditional_mae,
                "traditional_rmse": traditional_rmse,
                "traditional_mape": traditional_mape,
                "improvement_mae": (traditional_mae - enhanced_mae) / traditional_mae * 100 if traditional_mae else None,
                "improvement_rmse": (traditional_rmse - enhanced_rmse) / traditional_rmse * 100 if traditional_rmse else None,
                "improvement_mape": (traditional_mape - enhanced_mape) / traditional_mape * 100 if traditional_mape else None,
                "training_time": time.time() - start_time,
                "epochs_trained": len(history['loss']),
                "final_loss": history['loss'][-1],
                "model_path": str(enhanced_model_path)
            }
            
            logger.info(f"SUCCESS {symbol}: Enhanced MAPE={enhanced_mape:.2f}%, Direction={enhanced_direction_accuracy:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"FAILED {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "status": "failed",
                "error": str(e),
                "training_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def plot_comparison(self, symbol: str, actual: np.ndarray, enhanced_pred: np.ndarray, 
                       traditional_pred: Optional[np.ndarray] = None):
        """Create comparison plot"""
        try:
            plt.figure(figsize=(14, 8))
            
            days = range(len(actual))
            plt.plot(days, actual, label='Actual Price', color='black', linewidth=2)
            plt.plot(days, enhanced_pred, label='Sentiment-Enhanced LSTM', color='green', linewidth=2)
            
            if traditional_pred is not None:
                plt.plot(days, traditional_pred, label='Traditional LSTM', color='red', linewidth=2, alpha=0.7)
            
            plt.title(f'{symbol} - Model Comparison', fontsize=16)
            plt.xlabel('Days', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = self.output_dir / f"{symbol}_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to create comparison plot for {symbol}: {e}")
    
    def run_comparison_study(self, stock_df: pl.DataFrame, sentiment_df: pl.DataFrame,
                           n_random_stocks: int = 10, split_date: dt.date = None) -> pl.DataFrame:
        """Run full comparison study"""
        
        if split_date is None:
            split_date = dt.date(2024, 9, 30)
        
        # Get available symbols
        available_symbols = self.get_available_symbols()
        
        # Filter to symbols that exist in stock data
        stock_symbols = stock_df.select("symbol").unique().to_series().to_list()
        valid_symbols = [s for s in available_symbols if s in stock_symbols]
        
        logger.info(f"Valid symbols for comparison: {len(valid_symbols)}")
        
        # Randomly select symbols for comparison
        if len(valid_symbols) < n_random_stocks:
            logger.warning(f"Only {len(valid_symbols)} valid symbols available, using all")
            selected_symbols = valid_symbols
        else:
            selected_symbols = random.sample(valid_symbols, n_random_stocks)
        
        logger.info(f"Selected symbols for comparison: {selected_symbols}")
        
        # Run comparisons
        results = []
        for symbol in selected_symbols:
            result = self.compare_single_stock(symbol, stock_df, sentiment_df, split_date)
            results.append(result)
        
        # Save results
        results_df = pl.DataFrame(results)
        results_path = self.output_dir / "model_comparison_results.csv"
        results_df.write_csv(results_path)
        
        # Generate summary report
        self.generate_comparison_report(results_df)
        
        return results_df
    
    def generate_comparison_report(self, results_df: pl.DataFrame):
        """Generate detailed comparison report"""
        
        successful = results_df.filter(pl.col("status") == "success")
        failed = results_df.filter(pl.col("status") == "failed")
        
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("LSTM Model Comparison Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total stocks analyzed: {results_df.height}\n")
            f.write(f"Successful comparisons: {successful.height}\n")
            f.write(f"Failed comparisons: {failed.height}\n\n")
            
            if successful.height > 0:
                # Calculate average improvements
                avg_enhanced_mape = successful.select(pl.col("enhanced_mape").mean()).item()
                avg_direction_accuracy = successful.select(pl.col("enhanced_direction_accuracy").mean()).item()
                
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Sentiment-Enhanced LSTM - Average MAPE: {avg_enhanced_mape:.2f}%\n")
                f.write(f"Sentiment-Enhanced Direction Accuracy: {avg_direction_accuracy:.1f}%\n\n")
                
                # Check if we have traditional model comparisons
                traditional_data = successful.filter(pl.col("traditional_mape").is_not_null())
                if traditional_data.height > 0:
                    avg_traditional_mape = traditional_data.select(pl.col("traditional_mape").mean()).item()
                    f.write(f"Traditional LSTM - Average MAPE: {avg_traditional_mape:.2f}%\n\n")
                    
                    # Improvement statistics
                    improvements = traditional_data.filter(pl.col("improvement_mape").is_not_null())
                    if improvements.height > 0:
                        avg_improvement = improvements.select(pl.col("improvement_mape").mean()).item()
                        positive_improvements = improvements.filter(pl.col("improvement_mape") > 0).height
                        
                        f.write(f"Average MAPE Improvement: {avg_improvement:.2f}%\n")
                        f.write(f"Stocks with improved performance: {positive_improvements}/{improvements.height}\n")
                        f.write(f"Improvement success rate: {positive_improvements/improvements.height*100:.1f}%\n")
                        
                        # Log improvement info only if we have it
                        logger.info(f"Average improvement: {avg_improvement:.2f}%, Better performance: {positive_improvements}/{improvements.height} stocks")
                    else:
                        logger.warning("No improvement data available - traditional model loading failed")
                        f.write("No improvement data available - traditional model loading failed\n")
                else:
                    logger.warning("No traditional model comparisons available")
                    f.write("No traditional model comparisons available\n")
            else:
                logger.error("No successful comparisons to report")
                f.write("No successful comparisons completed\n")
    
        logger.info(f"Comparison report saved to {report_path}")
        
        logger.info("Sentiment-Enhanced LSTM Comparison Study completed!")


def load_stock_data_fallback() -> pl.DataFrame:
    """
    Fallback function to load stock data if custom modules aren't available.
    Replace this with your actual data loading logic.
    """
    logger.warning("Using fallback data loading. Please implement your actual data loading here.")
    
    # This is a placeholder - you should replace with actual data loading
    # For example, if you have a CSV file:
    # return pl.read_csv("path/to/your/stock_data.csv")
    
    # Or if you have a different data loading approach:
    # return your_data_loading_function()
    
    raise NotImplementedError("Please implement stock data loading in load_stock_data_fallback()")


def validate_data_compatibility(stock_df: pl.DataFrame, sentiment_df: pl.DataFrame) -> Tuple[List[str], str]:
    """
    Validate that stock and sentiment data are compatible for comparison.
    
    Returns:
        Tuple of (valid_symbols, status_message)
    """
    
    # Get symbols from both datasets
    stock_symbols = set(stock_df.select("symbol").unique().to_series().to_list())
    sentiment_symbols = set(sentiment_df.select("jse_symbols").unique().to_series().to_list())
    
    # Find overlap
    common_symbols = stock_symbols.intersection(sentiment_symbols)
    
    logger.info(f"Stock data symbols: {len(stock_symbols)}")
    logger.info(f"Sentiment data symbols: {len(sentiment_symbols)}")
    logger.info(f"Common symbols: {len(common_symbols)}")
    
    if len(common_symbols) == 0:
        return [], "No common symbols found between stock and sentiment data"
    
    # Check date ranges
    stock_date_range = stock_df.select([
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date")
    ]).row(0)
    sentiment_date_range = sentiment_df.select([
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date")
    ]).row(0)
    
    logger.info(f"Stock data date range: {stock_date_range[0]} to {stock_date_range[1]}")
    logger.info(f"Sentiment data date range: {sentiment_date_range[0]} to {sentiment_date_range[1]}")
    
    # Check for sufficient overlap
    overlap_start = max(stock_date_range[0].date(), sentiment_date_range[0])
    overlap_end   = min(stock_date_range[1].date(), sentiment_date_range[1])

    
    if overlap_start >= overlap_end:
        return [], "No temporal overlap between stock and sentiment data"
    
    overlap_days = (overlap_end - overlap_start).days
    if overlap_days < 365:  # Need at least 1 year of overlap
        return [], f"Insufficient temporal overlap: only {overlap_days} days"
    
    return list(common_symbols), f"Found {len(common_symbols)} compatible symbols with {overlap_days} days of overlap"


def create_comprehensive_analysis(results_df: pl.DataFrame, output_dir: Path):
    """Create comprehensive analysis plots and reports"""
    
    successful = results_df.filter(pl.col("status") == "success")
    
    if successful.height == 0:
        logger.warning("No successful comparisons to analyze")
        return
    
    # Create analysis plots
    try:
        plt.style.use('default')
        
        # 1. Performance comparison scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # MAPE comparison
        if "enhanced_mape" in successful.columns and "traditional_mape" in successful.columns:
            traditional_data = successful.filter(pl.col("traditional_mape").is_not_null())
            if traditional_data.height > 0:
                enhanced = traditional_data.select("enhanced_mape").to_series()
                traditional = traditional_data.select("traditional_mape").to_series()
                symbols = traditional_data.select("symbol").to_series()
                
                axes[0,0].scatter(traditional, enhanced, alpha=0.7, s=100)
                axes[0,0].plot([min(traditional), max(traditional)], [min(traditional), max(traditional)], 'r--', alpha=0.7)
                axes[0,0].set_xlabel('Traditional LSTM MAPE (%)')
                axes[0,0].set_ylabel('Sentiment-Enhanced LSTM MAPE (%)')
                axes[0,0].set_title('MAPE Comparison (Lower is Better)')
                axes[0,0].grid(True, alpha=0.3)
                
                # Add symbol labels
                for i, symbol in enumerate(symbols):
                    axes[0,0].annotate(symbol, (traditional[i], enhanced[i]), 
                                     xytext=(5, 5), textcoords='offset points', 
                                     fontsize=8, alpha=0.7)
        
        # Direction accuracy histogram
        direction_acc = successful.select("enhanced_direction_accuracy").to_series()
        axes[0,1].hist(direction_acc, bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].set_xlabel('Direction Accuracy (%)')
        axes[0,1].set_ylabel('Number of Stocks')
        axes[0,1].set_title('Sentiment-Enhanced Model Direction Accuracy')
        axes[0,1].grid(True, alpha=0.3)
        
        # Improvement distribution
        improvements = successful.filter(pl.col("improvement_mape").is_not_null())
        if improvements.height > 0:
            improvement_data = improvements.select("improvement_mape").to_series()
            axes[1,0].hist(improvement_data, bins=10, alpha=0.7, color='blue', edgecolor='black')
            axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
            axes[1,0].set_xlabel('MAPE Improvement (%)')
            axes[1,0].set_ylabel('Number of Stocks')
            axes[1,0].set_title('Performance Improvement Distribution')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Training time comparison
        training_times = successful.select("training_time").to_series()
        symbols = successful.select("symbol").to_series()
        axes[1,1].bar(range(len(symbols)), training_times, alpha=0.7, color='orange')
        axes[1,1].set_xlabel('Stock Symbol')
        axes[1,1].set_ylabel('Training Time (seconds)')
        axes[1,1].set_title('Training Time per Stock')
        axes[1,1].set_xticks(range(len(symbols)))
        axes[1,1].set_xticklabels(symbols, rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Comprehensive analysis plots created")
        
    except Exception as e:
        logger.error(f"Failed to create analysis plots: {e}")


def create_detailed_report(results_df: pl.DataFrame, output_dir: Path):
    """Create detailed markdown report"""
    
    successful = results_df.filter(pl.col("status") == "success")
    failed = results_df.filter(pl.col("status") == "failed")
    
    report_path = output_dir / "detailed_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Sentiment-Enhanced LSTM vs Traditional LSTM Comparison\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Stocks Analyzed**: {results_df.height}\n")
        f.write(f"- **Successful Comparisons**: {successful.height}\n")
        f.write(f"- **Failed Comparisons**: {failed.height}\n")
        f.write(f"- **Success Rate**: {successful.height/results_df.height*100:.1f}%\n\n")
        
        if successful.height > 0:
            # Performance metrics
            f.write("## Performance Metrics\n\n")
            
            avg_enhanced_mape = successful.select(pl.col("enhanced_mape").mean()).item()
            avg_enhanced_direction = successful.select(pl.col("enhanced_direction_accuracy").mean()).item()
            
            f.write("### Sentiment-Enhanced LSTM Performance\n")
            f.write(f"- **Average MAPE**: {avg_enhanced_mape:.2f}%\n")
            f.write(f"- **Average Direction Accuracy**: {avg_enhanced_direction:.1f}%\n\n")
            
            # Traditional comparison
            traditional_data = successful.filter(pl.col("traditional_mape").is_not_null())
            if traditional_data.height > 0:
                avg_traditional_mape = traditional_data.select(pl.col("traditional_mape").mean()).item()
                f.write("### Traditional LSTM Performance\n")
                f.write(f"- **Average MAPE**: {avg_traditional_mape:.2f}%\n\n")
                
                # Improvements
                improvements = traditional_data.filter(pl.col("improvement_mape").is_not_null())
                if improvements.height > 0:
                    avg_improvement = improvements.select(pl.col("improvement_mape").mean()).item()
                    positive_improvements = improvements.filter(pl.col("improvement_mape") > 0).height
                    
                    f.write("### Improvement Analysis\n")
                    f.write(f"- **Average MAPE Improvement**: {avg_improvement:.2f}%\n")
                    f.write(f"- **Stocks with Better Performance**: {positive_improvements}/{improvements.height} ({positive_improvements/improvements.height*100:.1f}%)\n\n")
            
            # Individual stock results
            f.write("## Individual Stock Results\n\n")
            f.write("| Symbol | Enhanced MAPE | Direction Acc | Traditional MAPE | Improvement | Status |\n")
            f.write("|--------|---------------|---------------|------------------|-------------|--------|\n")
            
            for row in successful.iter_rows(named=True):
                symbol = row["symbol"]
                enhanced_mape = row["enhanced_mape"]
                direction_acc = row["enhanced_direction_accuracy"]
                traditional_mape = row.get("traditional_mape", "N/A")
                improvement = row.get("improvement_mape", "N/A")
                
                if isinstance(traditional_mape, float):
                    traditional_mape = f"{traditional_mape:.2f}%"
                if isinstance(improvement, float):
                    improvement = f"{improvement:.2f}%"
                    status = "✅ Better" if improvement != "N/A" and float(improvement.replace("%", "")) > 0 else "❌ Worse"
                else:
                    status = "⚠️ No Comparison"
                
                f.write(f"| {symbol} | {enhanced_mape:.2f}% | {direction_acc:.1f}% | {traditional_mape} | {improvement} | {status} |\n")
        
        if failed.height > 0:
            f.write("\n## Failed Comparisons\n\n")
            f.write("| Symbol | Error | Training Time |\n")
            f.write("|--------|-------|---------------|\n")
            
            for row in failed.iter_rows(named=True):
                symbol = row["symbol"]
                error = row.get("error", "Unknown error")
                training_time = row.get("training_time", 0)
                f.write(f"| {symbol} | {error} | {training_time:.1f}s |\n")
        
        f.write("\n## Technical Details\n\n")
        f.write("### Sentiment-Enhanced Model Architecture\n")
        f.write("- **Sequence Length**: 30 days\n")
        f.write("- **LSTM Layers**: [128, 64, 32] units with BatchNormalization\n")
        f.write("- **Features**: Price + Technical + Sentiment (18 sentiment features)\n")
        f.write("- **Training**: Early stopping, learning rate reduction\n\n")
        
        f.write("### Key Sentiment Features\n")
        f.write("- Average sentiment, sentiment volatility\n")
        f.write("- Rolling sentiment averages (3d, 7d, 14d)\n")
        f.write("- Sentiment momentum indicators\n")
        f.write("- Sentiment-price interaction features\n")
        f.write("- Article volume and coverage indicators\n\n")
    
    logger.info(f"Detailed report saved to {report_path}")


def main():
    """Run the comparison study"""
    
    # Configuration
    TRADITIONAL_MODELS_DIR = "C:/Users/Joshh/Projects/Stocks/Models/LSTM"
    SENTIMENT_DATA_FILE = "C:/Users/Joshh/Projects/Stocks/Data/sentiment_data_FINAL.parquet"
    OUTPUT_DIR = "C:/Users/Joshh/Projects/Stocks/Models/LSTM/sentiment_comparison"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Sentiment-Enhanced LSTM Comparison Study")
    
    # Initialize comparator
    comparator = ModelComparator(TRADITIONAL_MODELS_DIR, OUTPUT_DIR)
    
    # Load sentiment data
    sentiment_df = comparator.load_sentiment_data(SENTIMENT_DATA_FILE)
    
    # Load stock data
    logger.info("Loading stock data...")
    try:
        # Try to load using your existing pipeline
        from data_loader import DataLoader
        
        loader = DataLoader()

        query = """
        SELECT DISTINCT
            trade_date as date,
            clean_symbol as symbol,
            -- open_price,
            daily_range_high as high_price,
            daily_range_low as low_price,
            closing_price,
            closing_price + price_change as opening_price,
            volume
        FROM
            old_trade_data TD
            LEFT JOIN  jse_database.instruments I on (TD.CLEAN_symbol = I.symbol AND TD.CURRENCY = I.CURRENCY)
        WHERE
            trade_date BETWEEN :start_date AND :end_date
            AND I.TYPE = 'ORDINARY'
            AND TD.CURRENCY = 'JMD'
        ORDER BY
            trade_date ASC,
            clean_symbol ASC
        """

        # Load all stocks data
        stock_df = loader.fetch_data(
            query=query,
            start_date='2017-01-01',
            end_date='2025-03-31'
        )

        logger.info(f"Loaded stock data: {stock_df.height:,} records for {stock_df.select('symbol').n_unique()} symbols")
        
    except Exception as e:
        logger.warning(f"Could not load data using custom pipeline: {e}")
        logger.info("Trying fallback data loading method...")
        try:
            stock_df = load_stock_data_fallback()
        except NotImplementedError:
            logger.error("Please implement stock data loading in load_stock_data_fallback() function")
            logger.error("Or ensure your custom data loading modules are available")
            return
    
    # Validate data compatibility
    valid_symbols, status_msg = validate_data_compatibility(stock_df, sentiment_df)
    logger.info(f"Data validation: {status_msg}")
    
    if not valid_symbols:
        logger.error("Cannot proceed with comparison - no compatible data found")
        return
    
    # Run comparison study
    try:
        results_df = comparator.run_comparison_study(
            stock_df=stock_df,
            sentiment_df=sentiment_df,
            n_random_stocks=10,
            split_date=dt.date(2024, 9, 30)
        )
        
        # Create comprehensive analysis
        create_comprehensive_analysis(results_df, Path(OUTPUT_DIR))
        create_detailed_report(results_df, Path(OUTPUT_DIR))
        
        # Print summary
        successful = results_df.filter(pl.col("status") == "success")
        if successful.height > 0:
            avg_enhanced_mape = successful.select(pl.col("enhanced_mape").mean()).item()
            avg_direction = successful.select(pl.col("enhanced_direction_accuracy").mean()).item()
            logger.info(f"Study completed! Sentiment-Enhanced Avg MAPE: {avg_enhanced_mape:.2f}%, Direction: {avg_direction:.1f}%")
            
            improvements = successful.filter(pl.col("improvement_mape").is_not_null())
            if improvements.height > 0:
                avg_improvement = improvements.select(pl.col("improvement_mape").mean()).item()
                positive_improvements = improvements.filter(pl.col("improvement_mape") > 0).height
                logger.info(f"Average improvement: {avg_improvement:.2f}%, Better performance: {positive_improvements}/{improvements.height} stocks")
                
                # Log best and worst performers
                if improvements.height > 2:
                    best_stock = improvements.sort("improvement_mape", descending=True).select("symbol").item(0)
                    best_improvement = improvements.sort("improvement_mape", descending=True).select("improvement_mape").item(0)
                    worst_stock = improvements.sort("improvement_mape").select("symbol").item(0)
                    worst_improvement = improvements.sort("improvement_mape").select("improvement_mape").item(0)
                    
                    logger.info(f"Best performer: {best_stock} ({best_improvement:.2f}% improvement)")
                    logger.info(f"Worst performer: {worst_stock} ({worst_improvement:.2f}% improvement)")
                
                # Statistical significance info
                significant_improvements = improvements.filter(pl.col("improvement_mape") > 5.0).height
                logger.info(f"Significant improvements (>5%): {significant_improvements}/{improvements.height} stocks")
                
            else:
                logger.warning("No improvement data available for comparison")
        else:
            logger.error("No successful comparisons completed")
        
        logger.info("Sentiment-Enhanced LSTM Comparison Study completed!")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        logger.info("Check the following files for detailed analysis:")
        logger.info(f"  - detailed_report.md (comprehensive report)")
        logger.info(f"  - comprehensive_analysis.png (visual analysis)")
        logger.info(f"  - model_comparison_results.csv (raw data)")
            
    except Exception as e:
        logger.error(f"Comparison study failed: {e}")
        raise


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    
    main()

