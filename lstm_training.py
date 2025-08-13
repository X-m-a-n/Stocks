import polars as pl
import numpy as np
import datetime as dt
import os
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Custom Imports
from data_loader import DataLoader
from jse_companies import jse_company_names
from technical_indicators import add_all_indicators, filter_buy_signals, get_signals_summary
from data import MarketDataPipeline

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LSTMStockPredictor:
    """
    LSTM-based stock price prediction model
    """
    
    def __init__(self, prediction_days: int = 30, lstm_units: int = 128, dropout_rate: float = 0.3):
        self.prediction_days = prediction_days
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        
    def prepare_data(self, df: pl.DataFrame, price_col: str = "closing_price", 
                    symbol: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Filter for specific symbol if provided
        if symbol:
            data = df.filter(pl.col("symbol") == symbol).sort("date")
        else:
            data = df.sort("date")
        
        # Extract closing prices and convert to float
        prices = data[price_col].cast(pl.Float64).to_numpy().reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create training sequences
        x_train, y_train = [], []
        
        for i in range(self.prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i-self.prediction_days:i, 0])
            y_train.append(scaled_data[i, 0])
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        self.model = Sequential([
            LSTM(units=self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(units=self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(units=self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
             epochs: int = 250, batch_size: int = 30, validation_split: float = 0.15,
             verbose: int = 0) -> dict:
        if self.model is None:
            input_shape = (x_train.shape[1], x_train.shape[2])
            self.build_model(input_shape)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=0.00001)
        ]
        
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(x_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions.flatten()
    
    def prepare_test_data(self, train_df: pl.DataFrame, test_df: pl.DataFrame,
                         price_col: str = "closing_price", symbol: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if symbol:
            train_data = train_df.filter(pl.col("symbol") == symbol).sort("date")
            test_data = test_df.filter(pl.col("symbol") == symbol).sort("date")
        else:
            train_data = train_df.sort("date")
            test_data = test_df.sort("date")
        
        actual_prices = test_data[price_col].cast(pl.Float64).to_numpy()
        
        combined_prices = pl.concat([
            train_data.select([price_col]),
            test_data.select([price_col])
        ])[price_col].cast(pl.Float64).to_numpy().reshape(-1, 1)
        
        scaled_combined = self.scaler.transform(combined_prices)
        model_inputs = scaled_combined[len(scaled_combined) - len(test_data) - self.prediction_days:]
        
        x_test = []
        for i in range(self.prediction_days, len(model_inputs)):
            x_test.append(model_inputs[i-self.prediction_days:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_test, actual_prices
    
    def plot_predictions(self, actual_prices: np.ndarray, predicted_prices: np.ndarray,
                        symbol: str = "Stock", save_path: Optional[str] = None) -> None:
        try:
            plt.ioff()  # Turn off interactive mode
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(actual_prices, color='black', label=f"Actual {symbol} Price", linewidth=2)
            ax.plot(predicted_prices, color='green', label=f"Predicted {symbol} Price", linewidth=2)
            ax.set_title(f"{symbol} Stock Price Prediction", fontsize=16)
            ax.set_xlabel('Time (Days)', fontsize=12)
            ax.set_ylabel(f'{symbol} Price', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close(fig)  # Explicitly close the figure
        except Exception as e:
            logger.warning(f"Failed to create plot for {symbol}: {e}")
    
    def predict_next_day(self, recent_data: np.ndarray) -> float:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(recent_data) < self.prediction_days:
            raise ValueError(f"Need at least {self.prediction_days} days of recent data")
        
        recent_data_float = np.array(recent_data, dtype=np.float64)
        recent_scaled = self.scaler.transform(recent_data_float[-self.prediction_days:].reshape(-1, 1))
        prediction_input = recent_scaled.reshape(1, self.prediction_days, 1)
        prediction = self.model.predict(prediction_input, verbose=0)
        prediction = self.scaler.inverse_transform(prediction)
        
        return float(prediction[0][0])


class SimpleBatchTrainer:
    """
    Simplified batch trainer for LSTM models
    """
    
    def __init__(self, output_dir: str = "models", results_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_symbols(self, df: pl.DataFrame) -> List[str]:
        """Get all unique stock symbols from dataframe"""
        return df.select("symbol").unique().to_series().to_list()
    
    def train_single_stock(self, df: pl.DataFrame, symbol: str, 
                          split_date: dt.date, **kwargs) -> Dict:
        """Train LSTM model for a single stock"""
        try:
            logger.info(f"Training model for {symbol}")
            start_time = time.time()
            
            # Filter data for this symbol
            symbol_data = df.filter(pl.col("symbol") == symbol).sort("date")
            
            # Check minimum data requirements
            if len(symbol_data) < 100:
                raise ValueError(f"Insufficient data: {len(symbol_data)} days")
            
            # Split data
            train_df = symbol_data.filter(pl.col("date") < split_date)
            test_df = symbol_data.filter(pl.col("date") >= split_date)
            
            if len(train_df) < kwargs.get('prediction_days', 30) * 2:
                raise ValueError(f"Insufficient training data: {len(train_df)} days")
            
            if len(test_df) < 5:
                raise ValueError(f"Insufficient test data: {len(test_df)} days")
            
            # Initialize and train predictor
            predictor = LSTMStockPredictor(
                prediction_days=kwargs.get('prediction_days', 30),
                lstm_units=kwargs.get('lstm_units', 128),
                dropout_rate=kwargs.get('dropout_rate', 0.3)
            )
            
            x_train, y_train = predictor.prepare_data(train_df, "closing_price")
            history = predictor.train(
                x_train, y_train,
                epochs=kwargs.get('epochs', 250),
                batch_size=kwargs.get('batch_size', 30),
                validation_split=kwargs.get('validation_split', 0.15),
                verbose=0
            )
            
            # Make predictions
            x_test, actual_prices = predictor.prepare_test_data(train_df, test_df, "closing_price")
            predicted_prices = predictor.predict(x_test)
            
            # Calculate metrics
            actual_prices = np.array(actual_prices, dtype=np.float64)
            predicted_prices = np.array(predicted_prices, dtype=np.float64)
            
            mae = np.mean(np.abs(actual_prices - predicted_prices))
            rmse = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            
            # Direction accuracy
            actual_direction = np.diff(actual_prices) > 0
            predicted_direction = np.diff(predicted_prices) > 0
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            # Next day prediction
            recent_prices = test_df.tail(kwargs.get('prediction_days', 30))["closing_price"].to_numpy()
            next_day_prediction = predictor.predict_next_day(recent_prices)
            current_price = float(recent_prices[-1])
            predicted_change = ((next_day_prediction - current_price) / current_price) * 100
            
            # Save model and plot (with thread safety)
            model_path = self.output_dir / f"{symbol}_lstm_model.pkl"
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(predictor, f)
            except Exception as e:
                logger.warning(f"Failed to save model for {symbol}: {e}")
            
            # Create plot in thread-safe way
            plot_path = self.results_dir / f"{symbol}_predictions.png"
            try:
                predictor.plot_predictions(actual_prices, predicted_prices, symbol, str(plot_path))
            except Exception as e:
                logger.warning(f"Failed to create plot for {symbol}: {e}")
            
            training_time = time.time() - start_time
            
            result = {
                'symbol': symbol,
                'status': 'success',
                'training_days': len(train_df),
                'test_days': len(test_df),
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'direction_accuracy': direction_accuracy,
                'final_loss': history['loss'][-1],
                'epochs_trained': len(history['loss']),
                'current_price': current_price,
                'next_day_prediction': next_day_prediction,
                'predicted_change_pct': predicted_change,
                'training_time': training_time,
                'model_path': str(model_path)
            }
            
            logger.info(f"SUCCESS {symbol}: MAPE={mape:.2f}%, Direction={direction_accuracy:.1f}%, Time={training_time:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"FAILED {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def train_all_stocks(self, df: pl.DataFrame, symbols: Optional[List[str]] = None,
                        split_date: Optional[dt.date] = None, parallel: bool = True,
                        max_workers: int = 10, **training_kwargs) -> pl.DataFrame:
        """Train LSTM models for all stocks"""
        
        if split_date is None:
            split_date = dt.date(2024, 9, 30)
        
        if symbols is None:
            symbols = self.get_all_symbols(df)
        
        logger.info(f"Training LSTM models for {len(symbols)} stocks")
        logger.info(f"Split date: {split_date}")
        
        results = []
        
        if parallel and len(symbols) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.train_single_stock, df, symbol, split_date, **training_kwargs): symbol 
                    for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    result = future.result()
                    results.append(result)
        else:
            for symbol in symbols:
                result = self.train_single_stock(df, symbol, split_date, **training_kwargs)
                results.append(result)
        
        # Save results
        results_df = pl.DataFrame(results)
        results_path = self.results_dir / "training_results.csv"
        results_df.write_csv(results_path)
        
        # Generate simple summary
        self.generate_summary(results_df)
        
        return results_df
    
    def generate_summary(self, results_df: pl.DataFrame) -> None:
        """Generate simple summary report"""
        successful = results_df.filter(pl.col("status") == "success")
        failed = results_df.filter(pl.col("status") == "failed")
        
        report_path = self.results_dir / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write("LSTM Training Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total stocks: {results_df.height}\n")
            f.write(f"Successful: {successful.height}\n")
            f.write(f"Failed: {failed.height}\n")
            f.write(f"Success rate: {(successful.height/results_df.height*100):.1f}%\n\n")
            
            if successful.height > 0:
                avg_mape = successful.select(pl.col("mape").mean()).item()
                avg_direction = successful.select(pl.col("direction_accuracy").mean()).item()
                f.write(f"Average MAPE: {avg_mape:.2f}%\n")
                f.write(f"Average Direction Accuracy: {avg_direction:.1f}%\n")
        
        logger.info(f"Summary saved to {report_path}")


def main():
    """
    Main execution function
    """
    logger.info("Starting LSTM batch training for JSE stocks")
    
    # Initialize components
    pipeline = MarketDataPipeline(DataLoader(), jse_company_names)
    trainer = SimpleBatchTrainer(
        output_dir="C:/Users/Joshh/Projects/Stocks/Models/LSTM",
        results_dir="C:/Users/Joshh/Projects/Stocks/Models/LSTM/analysis_results"
    )
    
    # Load data
    logger.info("Loading stock data...")
    df = pipeline.load_stock_data(start_date='2017-01-01', end_date='2025-03-31')
    logger.info(f"Loaded {df.height:,} records for {df.select('symbol').n_unique()} stocks")
    
    # Add technical indicators
    logger.info("Adding technical indicators...")
    technical_df = add_all_indicators(df, price_col="closing_price", symbol='all')
    
    # Train models for all stocks
    split_date = dt.date(2024, 9, 30)
    
    # Training parameters
    training_params = {
        'prediction_days': 30,
        'lstm_units': 128,
        'dropout_rate': 0.3,
        'epochs': 250,
        'batch_size': 30,
        'validation_split': 0.15
    }
    
    results_df = trainer.train_all_stocks(
        technical_df,
        split_date=split_date,
        parallel=True,
        max_workers=10,
        **training_params
    )
    
    # Print quick summary
    successful = results_df.filter(pl.col("status") == "success")
    if successful.height > 0:
        avg_mape = successful.select(pl.col("mape").mean()).item()
        avg_direction = successful.select(pl.col("direction_accuracy").mean()).item()
        logger.info(f"Training completed! Avg MAPE: {avg_mape:.2f}%, Avg Direction Accuracy: {avg_direction:.1f}%")
    
    logger.info("Batch training completed successfully!")


if __name__ == "__main__":
    main()