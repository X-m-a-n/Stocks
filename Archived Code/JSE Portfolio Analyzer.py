import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor
import ta_py as ta
import pathlib
from typing import Tuple, List, Dict
from datetime import datetime

def load_stock_data(data_folder: str) -> pl.DataFrame:
    """
    Load all stock data files from a folder into a single dataframe
    
    Parameters:
    data_folder (str): Path to folder containing stock data files
    
    Returns:
    polars.DataFrame: Combined stock data
    """
    # Get all CSV files in folder
    data_path = pathlib.Path(data_folder)
    data_files = list(data_path.glob('*.csv'))
    
    if not data_files:
        raise ValueError(f"No CSV files found in {data_folder}")
        
    # Read and combine all files
    dfs = []
    for file in data_files:
        try:
            df = pl.read_csv(
                file,
                try_parse_dates=True,
                columns=[
                    'date', 'symbol', 'last_price', 'closing_price', 
                    'price_change', 'bid', 'ask', 'volume', 'daily_range_low', 
                    'daily_range_high', 'year_range_low', 'year_range_high'
                ]
            )
            
            # Convert columns to appropriate types
            df = df.with_columns([
                pl.col(['last_price', 'closing_price', 'price_change', 'bid', 'ask', 
                       'daily_range_low', 'daily_range_high', 'year_range_low', 
                       'year_range_high']).cast(pl.Float64),
                pl.col('volume').cast(pl.Int64)
            ])
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid data files could be processed")
        
    # Combine all dataframes
    combined_df = pl.concat(dfs)
    
    # Sort by date and symbol
    combined_df = combined_df.sort(['date', 'symbol'])
    
    print(f"Loaded {len(data_files)} files with {len(combined_df)} total rows")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Unique symbols: {len(combined_df['symbol'].unique())}")
    
    return combined_df

def prepare_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create technical indicators and prepare features
    
    Parameters:
    df (pl.DataFrame): Input dataframe with stock data
    
    Returns:
    pl.DataFrame: Dataframe with added technical indicators
    """
    # First, ensure we're working with clean data
    df = df.drop_nulls(subset=['closing_price', 'daily_range_high', 'daily_range_low', 'volume'])
    
    # Convert polars series to numpy arrays for ta_py
    closing_prices = df['closing_price'].to_numpy()
    high_prices = df['daily_range_high'].to_numpy()
    low_prices = df['daily_range_low'].to_numpy()
    volumes = df['volume'].to_numpy()
    
    # Calculate technical indicators
    sma_20 = ta.sma(closing_prices, 20)
    sma_50 = ta.sma(closing_prices, 50)
    macd = ta.macd(closing_prices, 12, 26)
    rsi = ta.rsi(closing_prices, 14)
    
    # Calculate stochastic oscillator
    def calculate_stochastic(high, low, close, period=14):
        k_list = []
        for i in range(period - 1, len(close)):
            window_high = np.max(high[i-period+1:i+1])
            window_low = np.min(low[i-period+1:i+1])
            if window_high == window_low:
                k = 50
            else:
                k = 100 * (close[i] - window_low) / (window_high - window_low)
            k_list.append(k)
        return np.array(k_list)
    
    stoch = calculate_stochastic(high_prices, low_prices, closing_prices, 14)
    
    # Find the correct alignment point
    max_offset = 50  # Maximum lookback period
    min_length = min(len(sma_20) - max_offset, 
                    len(sma_50) - max_offset,
                    len(macd) - max_offset, 
                    len(rsi) - max_offset,
                    len(stoch))
    
    # Align all indicators to the same length from the end
    data_dict = {
        'sma_20': sma_20[-min_length:],
        'sma_50': sma_50[-min_length:],
        'macd': macd[-min_length:],
        'rsi': rsi[-min_length:],
        'stoch': stoch[-min_length:]
    }
    
    # Create indicators DataFrame
    indicators_df = pl.DataFrame(data_dict)
    
    # Trim original dataframe to match
    df_trimmed = df.slice(len(df) - min_length, len(df))
    
    # Add indicators to the trimmed dataframe
    for col in indicators_df.columns:
        df_trimmed = df_trimmed.with_columns([
            pl.Series(name=col, values=indicators_df[col])
        ])
    
    # Calculate returns
    df_trimmed = df_trimmed.with_columns([
        pl.col('closing_price').pct_change().alias('returns')
    ])
    
    # Final cleanup
    df_trimmed = df_trimmed.drop_nulls()
    
    return df_trimmed

def train_model(df: pl.DataFrame, symbol: str) -> Tuple[XGBRegressor, StandardScaler]:
    """
    Train prediction model for a given stock symbol
    
    Parameters:
    df (pl.DataFrame): Input dataframe with stock data
    symbol (str): Stock symbol to train model for
    
    Returns:
    Tuple[XGBRegressor, StandardScaler]: Trained model and feature scaler
    """
    # Filter data for symbol
    stock_df = df.filter(pl.col('symbol') == symbol)
    
    if len(stock_df) == 0:
        raise ValueError(f"No data found for symbol {symbol}")
    
    # Prepare features
    stock_df = prepare_features(stock_df)
    
    # Define features and target
    features = ['sma_20', 'sma_50', 'macd', 'rsi', 'stoch']
    target = 'returns'
    
    # Verify all features exist
    missing_features = [f for f in features if f not in stock_df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Split into features and target
    X = stock_df.select(features).to_numpy()
    y = stock_df.select(target).to_numpy().ravel()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize model 
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Train using time series split
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = mean_squared_error(y_val, pred, squared=False)
        scores.append(score)
    
    print(f"Average RMSE for {symbol}: {np.mean(scores):.4f}")
    
    # Final fit on all data
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_signals(
    model: XGBRegressor, 
    scaler: StandardScaler, 
    current_data: pl.DataFrame, 
    features: List[str],
    return_threshold: float = 0.05
) -> pl.DataFrame:
    """
    Generate trading signals based on model predictions
    
    Parameters:
    model (XGBRegressor): Trained model
    scaler (StandardScaler): Fitted feature scaler
    current_data (pl.DataFrame): Current market data
    features (List[str]): List of feature names
    return_threshold (float): Minimum return threshold for buy signals
    
    Returns:
    pl.DataFrame: DataFrame with predictions and signals
    """
    # Generate predictions
    X = current_data.select(features).to_numpy()
    X_scaled = scaler.transform(X)
    pred_returns = model.predict(X_scaled)
    
    # Create base DataFrame
    result_df = current_data.select(['date', 'closing_price']).unique(['date', 'closing_price'])
    
    # Add predicted returns
    result_df = result_df.with_columns([
        pl.Series(name='predicted_return', values=pred_returns[-len(result_df):] * 100)
    ])
    
    # Add signals (-1, 0, 1)
    signals = np.where(pred_returns > return_threshold, 1,
                      np.where(pred_returns < -return_threshold, -1, 0))
    
    result_df = result_df.with_columns([
        pl.Series(name='signal', values=signals[-len(result_df):])
    ])
    
    # Add signal descriptions
    result_df = result_df.with_columns([
        pl.when(pl.col('signal') == 1)
          .then(pl.lit('BUY'))
          .when(pl.col('signal') == -1)
          .then(pl.lit('SELL'))
          .otherwise(pl.lit('HOLD'))
          .alias('recommendation')
    ])
    
    # Format predicted return as percentage
    result_df = result_df.with_columns([
        pl.col('predicted_return').round(2).alias('predicted_return')
    ])
    
    return result_df

def analyze_all_stocks(df: pl.DataFrame) -> pl.DataFrame:
    """
    Analyze all stocks in the dataset and return combined results
    
    Parameters:
    df (pl.DataFrame): Input dataframe with all stock data
    
    Returns:
    pl.DataFrame: Combined analysis results for all stocks
    """
    # Get unique symbols
    symbols = df['symbol'].unique().to_list()
    print(f"\nAnalyzing {len(symbols)} stocks...")
    
    all_results = []
    
    for symbol in symbols:
        try:
            print(f"\nProcessing {symbol}...")
            
            # Train model for current symbol
            model, scaler = train_model(df, symbol)
            
            # Get current data for predictions
            current_data = prepare_features(df.filter(pl.col('symbol') == symbol))
            
            # Define features
            features = ['sma_20', 'sma_50', 'macd', 'rsi', 'stoch']
            
            # Generate predictions and signals
            result_df = predict_signals(
                model, 
                scaler, 
                current_data, 
                features,
                return_threshold=0.05
            )
            
            # Add symbol column
            result_df = result_df.with_columns([
                pl.lit(symbol).alias('symbol')
            ])
            
            # Get latest result only
            latest_result = result_df.sort('date', descending=True).head(1)
            
            # Calculate additional metrics
            signal_stats = result_df.group_by('recommendation').agg([
                pl.len().alias('signal_count')
            ])
            
            # Get counts for each signal type
            buy_count = signal_stats.filter(pl.col('recommendation') == 'BUY')['signal_count'].get(0, 0)
            sell_count = signal_stats.filter(pl.col('recommendation') == 'SELL')['signal_count'].get(0, 0)
            hold_count = signal_stats.filter(pl.col('recommendation') == 'HOLD')['signal_count'].get(0, 0)
            
            # Add statistics to latest result
            latest_result = latest_result.with_columns([
                pl.lit(buy_count).alias('total_buy_signals'),
                pl.lit(sell_count).alias('total_sell_signals'),
                pl.lit(hold_count).alias('total_hold_signals'),
                pl.col('predicted_return').mean().over('symbol').alias('avg_predicted_return'),
                pl.col('predicted_return').std().over('symbol').alias('return_volatility')
            ])
            
            all_results.append(latest_result)
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    if not all_results:
        raise ValueError("No valid results generated for any stock")
        
    # Combine all results
    combined_results = pl.concat(all_results)
    
    # Sort by predicted return (descending)
    combined_results = combined_results.sort('predicted_return', descending=True)
    
    return combined_results

def main():
    """Main function to run the stock predictor"""
    try:
        # Load data
        data_folder = r'C:\Users\michaelsjo\Desktop\Stocks\Data\eod_trade_summary'  # Update this path
        df = load_stock_data(data_folder)
        
        # Analyze all stocks
        results_df = analyze_all_stocks(df)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"stock_analysis_results_{timestamp}.csv"
        
        # Save results to CSV
        results_df.write_csv(output_file)
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total stocks analyzed: {len(results_df)}")
        print("\nTop 5 Stocks by Predicted Return:")
        print(results_df.head(5).select([
            'symbol', 
            'date', 
            'closing_price', 
            'predicted_return', 
            'recommendation'
        ]))
        
        # Signal distribution across all stocks
        print("\nOverall Signal Distribution:")
        signal_dist = results_df.group_by('recommendation').agg([
            pl.len().alias('count'),
            pl.mean('predicted_return').round(2).alias('avg_return_%')
        ]).sort('recommendation')
        
        print(signal_dist)
        
        # Print volatility summary
        print("\nVolatility Summary:")
        print(f"Average volatility across stocks: {results_df['return_volatility'].mean():.2f}")
        print(f"Most volatile stock: {results_df.sort('return_volatility', descending=True)['symbol'][0]}")
        print(f"Least volatile stock: {results_df.sort('return_volatility')['symbol'][0]}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()