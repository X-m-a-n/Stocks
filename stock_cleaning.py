import polars as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_and_format_data(data_path: str, output_path: str = None):
    """
    Clean and format stock data for LSTM training - Focus on most important features
    """
    
    print("🔧 DATA PREPROCESSING FOR LSTM TRAINING")
    print("="*60)
    
    # Load data
    print("📁 Loading data...")
    try:
        if data_path.endswith('.parquet'):
            df = pl.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            try:
                df = pl.read_csv(data_path)
            except:
                df = pl.read_csv(data_path, encoding="utf8-lossy")
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    print(f"✅ Data loaded: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏢 Symbols: {df['symbol'].n_unique()}")
    
    # Convert date column
    print("\n📅 Processing dates...")
    try:
        if df["date"].dtype == pl.Datetime:
            pass  # Already datetime
        else:
            df = df.with_columns(pl.col("date").str.to_datetime().alias("date"))
    except:
        try:
            df = df.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d").alias("date"))
        except:
            print("⚠️ Could not parse dates - keeping original format")
    
    # Sort data
    df = df.sort(["symbol", "date"])
    
    # Generate targets
    print("\n🎯 Generating targets...")
    df = df.with_columns([
        # Next day closing price
        pl.col("closing_price").shift(-1).over("symbol").alias("next_day_price"),
        
        # Price change percentage
        ((pl.col("closing_price").shift(-1).over("symbol") - pl.col("closing_price")) / 
         pl.col("closing_price") * 100).alias("next_day_change_pct"),
    ])
    
    # Generate direction categories
    df = df.with_columns([
        pl.when(pl.col("next_day_change_pct") > 5.0)
        .then(pl.lit("big_up"))
        .when(pl.col("next_day_change_pct") > 1.0)
        .then(pl.lit("up"))
        .when(pl.col("next_day_change_pct") < -1.0)
        .then(pl.lit("down"))
        .otherwise(pl.lit("neutral"))
        .alias("direction_category"),
        
        # Numeric direction mapping
        pl.when(pl.col("next_day_change_pct") > 5.0).then(3)  # big_up
        .when(pl.col("next_day_change_pct") > 1.0).then(2)    # up
        .when(pl.col("next_day_change_pct") < -1.0).then(0)   # down
        .otherwise(1)                                         # neutral
        .alias("direction_numeric")
    ])
    
    # Clean extreme values and invalid data
    print("\n🧹 Cleaning data...")
    initial_count = len(df)
    
    # Remove invalid targets
    df = df.filter(
        pl.col("next_day_price").is_not_null() &
        pl.col("next_day_change_pct").is_not_null() &
        pl.col("next_day_change_pct").is_finite() &
        (pl.col("next_day_change_pct").abs() <= 100) &  # Remove extreme outliers
        pl.col("closing_price").is_not_null() &
        (pl.col("closing_price") > 0)  # Remove zero/negative prices
    )
    
    cleaned_count = len(df)
    removed_count = initial_count - cleaned_count
    print(f"🗑️ Removed {removed_count:,} invalid rows ({removed_count/initial_count*100:.2f}%)")
    
    # Define the most important features based on correlation analysis
    print("\n🎯 Selecting most important features...")
    
    # Top features from your correlation analysis (17.47% to 5.66% correlation)
    TOP_FEATURES = [
        # Core momentum indicators (highest correlation)
        'intraday_return',      # 17.47% - Strongest predictor
        'daily_range',          # 12.53% - Volatility signal
        'returns',              # Base return signal
        
        # Technical indicators (9-10% correlation)
        'rsi_14',               # 9.06% - RSI momentum
        'macd',                 # MACD momentum
        'volume_ratio',         # 8.09% - Volume momentum
        
        # Price and volatility features
        'volatility_10d',       # Short-term volatility
        'volatility_20d',       # Medium-term volatility
        'sma_20',              # Short-term trend
        'sma_50',              # Medium-term trend
        
        # Volume indicators
        'volume_ma10',         # Volume moving average
        'volume_ma20',         # Volume moving average
        'volume',              # Raw volume
        
        # Additional technical indicators
        'signal_line',         # MACD signal
        'opening_price',       # Gap analysis
        'high_price',          # Daily high
        'low_price',           # Daily low
        
        # Sentiment features (if available)
        'avg_sentiment',
        'entity_specific_sentiment',
        'market_sentiment',
        'sentiment_volatility',
        'volume_weighted_sentiment',
        'sentiment_momentum',
        
        # Technical signal features (if available)
        'macd_buy_signal',
        'macd_sell_signal',
        'rsi_buy_signal',
        'rsi_sell_signal',
        'combined_buy_signal',
        'combined_sell_signal',
        
        # News and interaction features (if available)
        'market_news_volume_interaction',
        'sentiment_volume_interaction',
        'entity_sentiment_volume_interaction',
        'news_volume_trading_interaction'
    ]
    
    # Filter to only available features
    available_features = []
    for feature in TOP_FEATURES:
        if feature in df.columns:
            available_features.append(feature)
    
    print(f"📊 Found {len(available_features)} important features from {len(TOP_FEATURES)} desired")
    print("🎯 Selected features:", available_features[:15], "..." if len(available_features) > 15 else "")
    
    # Check and clean selected features
    print("\n🔍 Analyzing selected feature quality...")
    good_features = []
    
    for col in available_features:
        # Get column statistics
        null_count = df[col].null_count()
        null_pct = null_count / len(df) * 100
        col_dtype = df[col].dtype
        
        # Handle different data types
        if col_dtype == pl.Boolean:
            # Convert boolean to int8 for easier processing
            df = df.with_columns(pl.col(col).cast(pl.Int8).alias(col))
            print(f"   🔄 Converted {col} from boolean to int8")
        elif col_dtype in [pl.Float64, pl.Float32]:
            # Check for infinite values in float columns
            df = df.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
            # Recount nulls after inf replacement
            null_count = df[col].null_count()
            null_pct = null_count / len(df) * 100
        
        # Keep features with reasonable data quality
        if null_pct <= 40:  # Allow up to 40% missing for important features
            good_features.append(col)
        else:
            print(f"   ⚠️ Removed {col}: {null_pct:.1f}% missing/invalid")
    
    print(f"✅ Kept {len(good_features)} high-quality important features")
    
    # Fill missing values for selected features
    print("\n🔧 Filling missing values for selected features...")
    for col in good_features:
        # Get column data type
        col_dtype = df[col].dtype
        
        # Forward fill then backward fill for time series (by symbol)
        df = df.with_columns(
            pl.col(col).forward_fill().backward_fill().over("symbol").alias(col)
        )
        
        # Fill remaining nulls based on data type
        if col_dtype == pl.Boolean:
            # For boolean columns, fill with False
            df = df.with_columns(
                pl.col(col).fill_null(False).alias(col)
            )
        elif col_dtype in [pl.Float64, pl.Float32]:
            # For float columns, fill with median by symbol, then overall median
            median_overall = df[col].median()
            if median_overall is not None and not np.isnan(median_overall):
                df = df.with_columns(
                    pl.col(col).fill_null(median_overall).alias(col)
                )
            else:
                df = df.with_columns(
                    pl.col(col).fill_null(0.0).alias(col)
                )
        elif col_dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            # For integer columns, fill with 0
            df = df.with_columns(
                pl.col(col).fill_null(0).alias(col)
            )
        else:
            # For other types, try to fill with 0
            try:
                df = df.with_columns(
                    pl.col(col).fill_null(0).alias(col)
                )
            except:
                print(f"   ⚠️ Could not fill nulls for {col} (type: {col_dtype})")
                continue
    
    # Final data quality check
    print("\n✅ Verifying data quality...")
    for col in good_features:
        null_count = df[col].null_count()
        if null_count > 0:
            col_dtype = df[col].dtype
            print(f"⚠️ {col} still has {null_count} null values - filling appropriately")
            
            if col_dtype == pl.Boolean:
                df = df.with_columns(pl.col(col).fill_null(False).alias(col))
            elif col_dtype in [pl.Float64, pl.Float32]:
                df = df.with_columns(pl.col(col).fill_null(0.0).alias(col))
            elif col_dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                df = df.with_columns(pl.col(col).fill_null(0).alias(col))
            else:
                try:
                    df = df.with_columns(pl.col(col).fill_null(0).alias(col))
                except:
                    print(f"   ❌ Could not fill nulls for {col}")
                    good_features.remove(col)
    
    # Create final dataset - avoid duplicate columns
    print("\n📋 Creating final dataset...")
    
    # Core columns (always needed)
    core_cols = ["date", "symbol"]
    
    # Price columns (avoid duplicates)
    price_cols = []
    for col in ["closing_price", "opening_price", "high_price", "low_price"]:
        if col in df.columns and col not in price_cols:
            price_cols.append(col)
    
    # Feature columns (avoid duplicates with price columns)
    feature_cols = []
    for col in good_features:
        if col not in core_cols + price_cols:
            feature_cols.append(col)
    
    # Target columns
    target_cols = ["next_day_price", "next_day_change_pct", "direction_category", "direction_numeric"]
    
    # Combine all columns
    final_cols = core_cols + price_cols + feature_cols + target_cols
    
    print(f"📊 Final dataset structure:")
    print(f"   Core: {len(core_cols)} columns")
    print(f"   Price: {len(price_cols)} columns") 
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Targets: {len(target_cols)} columns")
    print(f"   Total: {len(final_cols)} columns")
    
    # Select final columns
    df_final = df.select(final_cols)
    
    # Direction distribution
    direction_dist = df_final.group_by("direction_category").agg(
        pl.count().alias("count")
    ).with_columns(
        (pl.col("count") / pl.col("count").sum() * 100).alias("percentage")
    ).sort("count", descending=True)
    
    print(f"\n📊 Direction distribution:")
    for row in direction_dist.iter_rows(named=True):
        print(f"   {row['direction_category']}: {row['count']:,} ({row['percentage']:.2f}%)")
    
    # Symbol statistics
    symbol_stats = df_final.group_by("symbol").agg([
        pl.count().alias("total_rows"),
        pl.col("next_day_change_pct").std().alias("volatility"),
        pl.col("closing_price").mean().alias("avg_price")
    ]).sort("total_rows", descending=True)
    
    print(f"\n🏢 Top 10 symbols by data points:")
    for row in symbol_stats.head(10).iter_rows(named=True):
        print(f"   {row['symbol']}: {row['total_rows']} rows, "
              f"vol: {row['volatility']:.2f}%, "
              f"avg: ${row['avg_price']:.2f}")
    
    # Combine feature columns for metadata
    all_model_features = price_cols + feature_cols
    
    # Save processed data
    if output_path:
        print(f"\n💾 Saving processed data to: {output_path}")
        if output_path.endswith('.parquet'):
            df_final.write_parquet(output_path)
        else:
            df_final.write_csv(output_path)
    
    # Save metadata
    metadata = {
        'processing_date': datetime.now().isoformat(),
        'original_shape': [initial_count, len(df.columns)],
        'final_shape': [len(df_final), len(df_final.columns)],
        'rows_removed': removed_count,
        'features_kept': all_model_features,  # All features available for modeling
        'top_features_available': feature_cols,  # Most important features
        'price_features': price_cols,
        'direction_distribution': direction_dist.to_pandas().to_dict('records'),
        'symbols_count': df_final['symbol'].n_unique(),
        'date_range': [str(df_final['date'].min()), str(df_final['date'].max())]
    }
    
    # Save metadata
    with open('preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ PREPROCESSING COMPLETED")
    print(f"📊 Final dataset: {df_final.shape}")
    print(f"🎯 Model features: {len(all_model_features)}")
    print(f"⭐ Top features: {len(feature_cols)}")
    print(f"🏢 Symbols: {df_final['symbol'].n_unique()}")
    print(f"📁 Files saved:")
    print(f"   • {output_path}")
    print(f"   • preprocessing_metadata.json")
    
    return df_final, all_model_features, metadata

def create_scaled_features(df, features, method='robust'):
    """
    Create scaled versions of features for comparison
    """
    print(f"\n⚖️ Creating {method} scaled features...")
    
    # Choose scaler
    if method == 'standard':
        scaler_class = StandardScaler
    elif method == 'robust':
        scaler_class = RobustScaler
    elif method == 'minmax':
        scaler_class = MinMaxScaler
    else:
        raise ValueError("Method must be 'standard', 'robust', or 'minmax'")
    
    df_scaled = df.clone()
    
    # Scale features by symbol
    symbols = df['symbol'].unique().to_list()
    
    for symbol in symbols:
        symbol_mask = df['symbol'] == symbol
        symbol_data = df.filter(symbol_mask)
        
        if len(symbol_data) < 10:  # Skip symbols with too little data
            continue
        
        # Convert to pandas for scaling
        symbol_pd = symbol_data.to_pandas()
        
        # Scale features
        scaler = scaler_class()
        scaled_features = scaler.fit_transform(symbol_pd[features])
        
        # Update scaled dataframe
        for i, feature in enumerate(features):
            # Create scaled column name
            scaled_col = f"{feature}_scaled"
            if scaled_col not in df_scaled.columns:
                df_scaled = df_scaled.with_columns(pl.lit(0.0).alias(scaled_col))
            
            # Update values for this symbol
            symbol_indices = df.with_row_index().filter(pl.col('symbol') == symbol)['index'].to_list()
            
            # This is a simplified approach - in practice you'd want to handle this more efficiently
            scaled_values = scaled_features[:, i]
            
            # Update the scaled dataframe (this is a workaround for polars)
            symbol_df_scaled = df_scaled.filter(pl.col('symbol') == symbol).with_columns(
                pl.lit(scaled_values).alias(scaled_col)
            )
    
    return df_scaled

if __name__ == "__main__":
    # Configuration
    data_path = r"C:\Users\Joshh\Projects\Stocks\Data\stocks_df.csv"
    output_path = r"C:\Users\Joshh\Projects\Stocks\clean_stock_data.parquet"
    
    # Run preprocessing
    df_clean, features, metadata = clean_and_format_data(data_path, output_path)
    
    if df_clean is not None:
        print(f"\n🎉 SUCCESS! Clean data ready for LSTM training")
        print(f"📈 Use this file for training: {output_path}")
        print(f"🔧 Available features: {len(features)}")