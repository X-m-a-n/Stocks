import multiprocessing as mp
import polars as pl
import numpy as np
import pickle
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
import logging
from datetime import datetime
import traceback

# Statistical libraries for ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Your existing imports
from data_loader import DataLoader
from jse_companies import jse_company_names
from data import MarketDataPipeline


class ARIMAStockPredictor:
    """
    ARIMA-based stock price prediction model
    """
    
    def __init__(self, auto_order: bool = True):
        self.auto_order = auto_order
        self.fitted_model = None
        self.order = None
        self.is_trained = False
        
    def find_optimal_order(self, data: np.ndarray, max_p: int = 3, max_q: int = 3, 
                          max_d: int = 2) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using grid search with AIC"""
        best_aic = np.inf
        best_order = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order or (1, 1, 1)
    
    def fit(self, data: np.ndarray, order: Optional[Tuple[int, int, int]] = None) -> None:
        """Fit ARIMA model to the data"""
        if order is None and self.auto_order:
            order = self.find_optimal_order(data, max_p=3, max_q=3, max_d=2)
        elif order is None:
            order = (1, 1, 1)
            
        self.order = order
        self.fitted_model = ARIMA(data, order=order).fit()
        self.is_trained = True
    
    def predict_next_day(self) -> Dict:
        """Predict the next day's stock price"""
        if not self.is_trained:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.get_forecast(steps=1)
        forecast_values = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int(alpha=0.05)
        
        # Handle both numpy arrays and pandas Series
        if hasattr(forecast_values, 'values'):
            forecast_array = forecast_values.values[0]
        else:
            forecast_array = forecast_values[0]
            
        if hasattr(confidence_intervals, 'iloc'):
            lower_ci = confidence_intervals.iloc[0, 0]
            upper_ci = confidence_intervals.iloc[0, 1]
        else:
            lower_ci = confidence_intervals[0, 0]
            upper_ci = confidence_intervals[0, 1]
        
        return {
            'predicted_price': float(forecast_array),
            'lower_bound': float(lower_ci),
            'upper_bound': float(upper_ci)
        }
    
    def backtest_on_test_set(self, train_data: np.ndarray, test_data: np.ndarray) -> Dict:
        """Perform backtesting on held-out test set"""
        predictions = []
        actual_values = []
        
        # Train initial model on training data
        try:
            temp_model = ARIMA(train_data, order=self.order)
            temp_fitted = temp_model.fit()
        except:
            temp_model = ARIMA(train_data, order=(1, 1, 1))
            temp_fitted = temp_model.fit()
        
        # Predict each point in test set
        for i in range(len(test_data)):
            try:
                if i == 0:
                    current_train = train_data
                else:
                    current_train = np.concatenate([train_data, test_data[:i]])
                
                current_model = ARIMA(current_train, order=self.order)
                current_fitted = current_model.fit()
                forecast = current_fitted.forecast(steps=1)
                predictions.append(float(forecast[0]))
                actual_values.append(float(test_data[i]))
                
            except Exception:
                # Fallback to persistence model
                last_value = current_train[-1] if i == 0 else test_data[i-1]
                predictions.append(float(last_value))
                actual_values.append(float(test_data[i]))
        
        # Calculate metrics
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        mae = np.mean(np.abs(predictions - actual_values))
        rmse = np.sqrt(np.mean((predictions - actual_values) ** 2))
        mape = np.mean(np.abs((predictions - actual_values) / np.maximum(actual_values, 1e-8))) * 100
        
        # Direction accuracy
        if len(actual_values) > 1:
            actual_direction = np.sign(np.diff(actual_values))
            pred_direction = np.sign(np.diff(predictions))
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 0.0
        
        return {
            'predictions': predictions,
            'actual': actual_values,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'test_size': len(test_data)
        }


def prepare_stock_data(df: pl.DataFrame, symbol: str, split_date: str = "2024-09-30") -> Dict:
    """
    Prepare data for a specific stock symbol with date-based train/test split
    """
    try:
        stock_data = df.filter(pl.col("symbol") == symbol).sort("date")
        
        if len(stock_data) < 100:
            raise ValueError(f"Insufficient data for {symbol}: {len(stock_data)} observations")
        
        # Convert split_date to proper format
        split_date_obj = datetime.strptime(split_date, "%Y-%m-%d").date()
        
        # Split data based on date
        train_data_df = stock_data.filter(pl.col("date") < split_date_obj)
        test_data_df = stock_data.filter(pl.col("date") >= split_date_obj)
        
        if len(train_data_df) < 50:
            raise ValueError(f"Insufficient training data for {symbol}: {len(train_data_df)} observations before {split_date}")
        if len(test_data_df) < 5:
            raise ValueError(f"Insufficient test data for {symbol}: {len(test_data_df)} observations after {split_date}")
        
        # Extract prices
        train_prices = train_data_df.select("closing_price").to_series().cast(pl.Float64)
        if train_prices.null_count() > 0:
            train_prices = train_prices.fill_null(strategy="forward")
        train_data = train_prices.to_numpy()
        
        test_prices = test_data_df.select("closing_price").to_series().cast(pl.Float64)
        if test_prices.null_count() > 0:
            test_prices = test_prices.fill_null(strategy="forward")
        test_data = test_prices.to_numpy()
        
        # Check for valid price data
        if np.any(train_data <= 0) or np.any(test_data <= 0):
            raise ValueError(f"Invalid price data for {symbol}")
        
        return {
            'symbol': symbol,
            'train_data': train_data,
            'test_data': test_data,
            'split_date': split_date,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_start': train_data_df.select("date").min().item(),
            'train_end': train_data_df.select("date").max().item(),
            'test_start': test_data_df.select("date").min().item(),
            'test_end': test_data_df.select("date").max().item()
        }
        
    except Exception as e:
        raise Exception(f"Error preparing data for {symbol}: {str(e)}")


def train_single_arima_model(args: Tuple) -> Dict:
    """
    Train ARIMA model for a single stock
    """
    symbol, split_date, output_dir, results_dir = args
    
    result = {
        'symbol': symbol,
        'success': False,
        'error': None,
        'model_path': None,
        'metrics': {},
        'prediction': {},
        'training_time': 0
    }
    
    try:
        start_time = datetime.now()
        
        # Load data for this symbol (recreate from global data)
        pipeline = MarketDataPipeline(DataLoader(), jse_company_names)
        df = pipeline.load_stock_data(start_date='2017-01-01', end_date='2025-03-31')
        
        # Prepare data
        data_dict = prepare_stock_data(df, symbol, split_date)
        train_data = data_dict['train_data']
        test_data = data_dict['test_data']
        
        # Initialize and train ARIMA predictor
        arima_predictor = ARIMAStockPredictor(auto_order=True)
        arima_predictor.fit(train_data)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        next_day_pred = arima_predictor.predict_next_day()
        current_price = float(train_data[-1])
        
        # Backtest on test set
        backtest_results = arima_predictor.backtest_on_test_set(train_data, test_data)
        
        # Save model
        model_filename = f"{symbol}_arima_model.pkl"
        model_path = Path(output_dir) / model_filename
        
        with open(model_path, 'wb') as f:
            model_data = {
                'fitted_model': arima_predictor.fitted_model,
                'order': arima_predictor.order,
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'split_date': split_date,
                'train_size': len(train_data),
                'test_size': len(test_data)
            }
            pickle.dump(model_data, f)
        
        # Prepare results
        result.update({
            'success': True,
            'model_path': str(model_path),
            'training_time': training_time,
            'arima_order': arima_predictor.order,
            'data_info': {
                'train_size': len(train_data),
                'test_size': len(test_data),
                'split_date': split_date,
                'train_period': f"{data_dict['train_start']} to {data_dict['train_end']}",
                'test_period': f"{data_dict['test_start']} to {data_dict['test_end']}"
            },
            'metrics': {
                'mae': float(backtest_results['mae']),
                'rmse': float(backtest_results['rmse']),
                'mape': float(backtest_results['mape']),
                'direction_accuracy': float(backtest_results['direction_accuracy'])
            },
            'prediction': {
                'current_price': current_price,
                'predicted_price': float(next_day_pred['predicted_price']),
                'lower_bound': float(next_day_pred['lower_bound']),
                'upper_bound': float(next_day_pred['upper_bound'])
            }
        })
        
        # Save individual results
        results_filename = f"{symbol}_arima_results.json"
        results_path = Path(results_dir) / results_filename
        
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ {symbol}: ARIMA{arima_predictor.order}, MAPE: {backtest_results['mape']:.2f}%, Dir: {backtest_results['direction_accuracy']:.1f}%")
        
        return result
        
    except Exception as e:
        error_msg = f"Error training {symbol}: {str(e)}"
        print(f"✗ {error_msg}")
        
        result.update({
            'error': error_msg,
            'traceback': traceback.format_exc()
        })
        
        return result


def main():
    """
    Main function to run the ARIMA batch training
    """
    # Configuration
    SPLIT_DATE = "2024-09-30"
    OUTPUT_DIR = "C:/Users/Joshh/Projects/Stocks/Models/ARIMA"
    RESULTS_DIR = "C:/Users/Joshh/Projects/Stocks/Models/ARIMA/analysis_results"
    MAX_WORKERS = 10
    
    print("🔄 Loading stock data...")
    
    # Load stock data
    pipeline = MarketDataPipeline(DataLoader(), jse_company_names)
    df = pipeline.load_stock_data(start_date='2017-01-01', end_date='2025-03-31')
    
    print(f"✅ Loaded {len(df)} records for {df.select('symbol').n_unique()} unique stocks")
    
    # Create directories
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get unique symbols and prepare arguments
    symbols = df.select("symbol").unique().to_series().to_list()
    
    print(f"🚀 Starting ARIMA model training for {len(symbols)} stocks...")
    print(f"📅 Split Date: {SPLIT_DATE} (Train: before, Test: after)")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📁 Results: {RESULTS_DIR}")
    
    # Prepare arguments for multiprocessing
    args_list = []
    for symbol in symbols:
        try:
            # Quick validation
            prepare_stock_data(df, symbol, SPLIT_DATE)
            args_list.append((symbol, SPLIT_DATE, OUTPUT_DIR, RESULTS_DIR))
        except Exception as e:
            print(f"⚠️  Skipping {symbol}: {str(e)}")
    
    print(f"✅ {len(args_list)} stocks ready for training")
    
    if len(args_list) == 0:
        print("❌ No stocks to train!")
        return
    
    # Train models using multiprocessing
    start_time = datetime.now()
    
    with mp.Pool(processes=min(MAX_WORKERS, len(args_list))) as pool:
        results = pool.map(train_single_arima_model, args_list)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 ARIMA MODEL TRAINING SUMMARY")
    print("="*80)
    print(f"📊 Results:")
    print(f"   • Total stocks: {len(symbols)}")
    print(f"   • Successful: {len(successful)}")
    print(f"   • Failed: {len(failed)}")
    print(f"   • Success rate: {len(successful)/len(args_list)*100:.1f}%")
    print(f"   • Total time: {total_time:.1f} seconds")
    
    if successful:
        mape_values = [r['metrics']['mape'] for r in successful]
        direction_values = [r['metrics']['direction_accuracy'] for r in successful]
        
        print(f"\n📈 Performance on Test Set (after {SPLIT_DATE}):")
        print(f"   • Mean MAPE: {np.mean(mape_values):.2f}%")
        print(f"   • Median MAPE: {np.median(mape_values):.2f}%")
        print(f"   • Best MAPE: {np.min(mape_values):.2f}%")
        print(f"   • Mean Direction Accuracy: {np.mean(direction_values):.1f}%")
        
        # Top 5 performers
        top_models = sorted(successful, key=lambda x: x['metrics']['mape'])[:5]
        print(f"\n🏆 Top 5 Models:")
        for i, model in enumerate(top_models, 1):
            mape = model['metrics']['mape']
            direction = model['metrics']['direction_accuracy']
            print(f"   {i}. {model['symbol']}: {mape:.2f}% MAPE, {direction:.1f}% Direction")
    
    if failed:
        print(f"\n❌ Failed Models ({len(failed)}):")
        for model in failed[:5]:
            print(f"   • {model['symbol']}: {model['error']}")
        if len(failed) > 5:
            print(f"   ... and {len(failed)-5} more")
    
    # Save summary
    summary = {
        'total_stocks': len(symbols),
        'successful_models': len(successful),
        'failed_models': len(failed),
        'split_date': SPLIT_DATE,
        'training_time': total_time,
        'results': results
    }
    
    summary_path = Path(RESULTS_DIR) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Training completed! Summary saved to {summary_path}")
    
    return summary


if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    mp.set_start_method('spawn', force=True)
    
    results = main()