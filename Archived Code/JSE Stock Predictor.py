import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor
import ta_py as ta
import pathlib
from typing import Tuple, List, Union

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
    
    print(f"Initial data length: {len(closing_prices)}")
    
    # Calculate technical indicators with proper offset handling
    sma_20 = ta.sma(closing_prices, 20)
    sma_50 = ta.sma(closing_prices, 50)
    macd = ta.macd(closing_prices, 12, 26)
    rsi = ta.rsi(closing_prices, 14)
    
    # Manual calculation of stochastic oscillator since ta_py's stoch function is having issues
    def calculate_stochastic(high, low, close, period=14):
        k_list = []
        for i in range(period - 1, len(close)):
            window_high = np.max(high[i-period+1:i+1])
            window_low = np.min(low[i-period+1:i+1])
            if window_high == window_low:
                k = 50  # Default to middle value if high == low
            else:
                k = 100 * (close[i] - window_low) / (window_high - window_low)
            k_list.append(k)
        return np.array(k_list)
    
    stoch = calculate_stochastic(high_prices, low_prices, closing_prices, 14)
    
    print(f"Lengths after calculation:")
    print(f"SMA-20: {len(sma_20)}")
    print(f"SMA-50: {len(sma_50)}")
    print(f"MACD: {len(macd)}")
    print(f"RSI: {len(rsi)}")
    print(f"Stoch: {len(stoch)}")
    
    # Find the correct alignment point
    max_offset = 50  # Maximum lookback period
    min_length = min(len(sma_20) - max_offset, 
                    len(sma_50) - max_offset,
                    len(macd) - max_offset, 
                    len(rsi) - max_offset,
                    len(stoch))
    
    print(f"Aligned length will be: {min_length}")
    
    # Align all indicators to the same length from the end
    data_dict = {
        'sma_20': sma_20[-min_length:],
        'sma_50': sma_50[-min_length:],
        'macd': macd[-min_length:],
        'rsi': rsi[-min_length:],
        'stoch': stoch[-min_length:]
    }
    
    # Verify lengths before creating DataFrame
    for name, series in data_dict.items():
        print(f"Final length of {name}: {len(series)}")
        
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
    print(f"Final DataFrame shape: {df_trimmed.shape}")
    
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
    print(f"\nProcessing symbol: {symbol}")
    
    # Filter data for symbol
    stock_df = df.filter(pl.col('symbol') == symbol)
    print(f"Data points for {symbol}: {len(stock_df)}")
    
    if len(stock_df) == 0:
        raise ValueError(f"No data found for symbol {symbol}")
    
    # Prepare features
    try:
        stock_df = prepare_features(stock_df)
        print(f"DataFrame shape after feature preparation: {stock_df.shape}")
    except Exception as e:
        print(f"Error in feature preparation: {str(e)}")
        raise
    
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
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
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
    
    print(f"Average RMSE: {np.mean(scores):.4f}")
    
    return model, scaler

def predict_signals(
    model: XGBRegressor, 
    scaler: StandardScaler, 
    current_data: pl.DataFrame, 
    features: List[str],
    return_threshold: float = 0.075  # 7.5% return threshold
) -> pl.DataFrame:
    """
    Generate trading signals based on model predictions
    
    Parameters:
    model (XGBRegressor): Trained model
    scaler (StandardScaler): Fitted feature scaler
    current_data (pl.DataFrame): Current market data
    features (List[str]): List of feature names
    return_threshold (float): Minimum return threshold for buy signals (default: 7.5%)
    
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

def main():
    """Main function to run the stock predictor"""
    try:
        # Load data
        data_folder = r'C:\Users\michaelsjo\Desktop\Stocks\Data\eod_trade_summary'
        df = load_stock_data(data_folder)
        
        # Train model for specific symbol
        symbol = 'WIG'
        print(f"\nAnalyzing {symbol}...")
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
            return_threshold=0.05  #return threshold
        )
        
        # Sort by date and get truly latest dates
        result_df = result_df.sort('date', descending=True)
        
        # Print latest predictions
        print("\nLatest Trading Signals (Last 5 Trading Days):")
        latest_signals = result_df.head(5).sort('date')  # Get first 5 from descending sorted df
        
        print(latest_signals.select([
            'date',
            'closing_price',
            'predicted_return',
            'recommendation'
        ]))
        
        # Calculate signal distribution
        print("\nSignal Distribution:")
        signal_counts = result_df.group_by('recommendation').agg([
            pl.len().alias('count'),
            pl.mean('predicted_return').round(2).alias('avg_return_%')
        ]).sort('recommendation')
        
        print(signal_counts)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total predictions: {len(result_df)}")
        print(f"Average predicted return: {result_df['predicted_return'].mean():.2f}%")
        print(f"Max predicted return: {result_df['predicted_return'].max():.2f}%")
        print(f"Min predicted return: {result_df['predicted_return'].min():.2f}%")

        # Write results to file for analysis
        out_file = f'C:/Users/michaelsjo/Desktop/Stocks/Data/Results/{symbol}.xlsx'
        result_df.write_excel(out_file)
        
        # Print date range of predictions
        print(f"\nPrediction Date Range:")
        print(f"From: {result_df['date'].min()}")
        print(f"To: {result_df['date'].max()}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()