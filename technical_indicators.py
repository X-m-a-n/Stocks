import polars as pl
import numpy as np
from typing import Optional, Union, List


def calculate_sma(df: pl.DataFrame, column: str = "close", window: int = 20, 
                 output_col: Optional[str] = None, symbol_col: Optional[str] = None) -> pl.DataFrame:
    """
    Calculate Simple Moving Average using Polars
    
    Args:
        df: Polars DataFrame with price data
        column: Column name for price data (default: "close")
        window: Period for moving average (default: 20)
        output_col: Name for output column (default: f"sma_{window}")
        symbol_col: Column name for symbol grouping (default: None)
    
    Returns:
        DataFrame with SMA column added
    """
    if output_col is None:
        output_col = f"sma_{window}"
    
    if symbol_col is not None:
        return df.with_columns(
            pl.col(column).rolling_mean(window_size=window).over(symbol_col).alias(output_col)
        )
    else:
        return df.with_columns(
            pl.col(column).rolling_mean(window_size=window).alias(output_col)
        )


def calculate_ema(df: pl.DataFrame, column: str = "close", span: int = 20, 
                 output_col: Optional[str] = None, symbol_col: Optional[str] = None) -> pl.DataFrame:
    """
    Calculate Exponential Moving Average using Polars
    
    Args:
        df: Polars DataFrame with price data
        column: Column name for price data (default: "close")
        span: Span for EMA calculation (default: 20)
        output_col: Name for output column (default: f"ema_{span}")
        symbol_col: Column name for symbol grouping (default: None)
    
    Returns:
        DataFrame with EMA column added
    """
    if output_col is None:
        output_col = f"ema_{span}"
    
    # Calculate alpha for EMA
    alpha = 2.0 / (span + 1.0)
    
    if symbol_col is not None:
        return df.with_columns(
            pl.col(column).ewm_mean(alpha=alpha).over(symbol_col).alias(output_col)
        )
    else:
        return df.with_columns(
            pl.col(column).ewm_mean(alpha=alpha).alias(output_col)
        )


def calculate_rsi(df: pl.DataFrame, column: str = "close", window: int = 14, 
                 output_col: Optional[str] = None, symbol_col: Optional[str] = None) -> pl.DataFrame:
    """
    Calculate Relative Strength Index using Polars
    
    Args:
        df: Polars DataFrame with price data
        column: Column name for price data (default: "close")
        window: Period for RSI calculation (default: 14)
        output_col: Name for output column (default: f"rsi_{window}")
        symbol_col: Column name for symbol grouping (default: None)
    
    Returns:
        DataFrame with RSI column added
    """
    if output_col is None:
        output_col = f"rsi_{window}"
    
    if symbol_col is not None:
        return df.with_columns([
            # Calculate price differences
            pl.col(column).diff().over(symbol_col).alias("delta"),
        ]).with_columns([
            # Calculate gains and losses
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).alias("loss"),
        ]).with_columns([
            # Calculate average gains and losses
            pl.col("gain").rolling_mean(window_size=window).over(symbol_col).alias("avg_gain"),
            pl.col("loss").rolling_mean(window_size=window).over(symbol_col).alias("avg_loss"),
        ]).with_columns([
            # Calculate RSI
            (100 - (100 / (1 + (pl.col("avg_gain") / pl.col("avg_loss"))))).alias(output_col)
        ]).drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])
    else:
        return df.with_columns([
            # Calculate price differences
            pl.col(column).diff().alias("delta"),
        ]).with_columns([
            # Calculate gains and losses
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).alias("loss"),
        ]).with_columns([
            # Calculate average gains and losses
            pl.col("gain").rolling_mean(window_size=window).alias("avg_gain"),
            pl.col("loss").rolling_mean(window_size=window).alias("avg_loss"),
        ]).with_columns([
            # Calculate RSI
            (100 - (100 / (1 + (pl.col("avg_gain") / pl.col("avg_loss"))))).alias(output_col)
        ]).drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])


def calculate_macd(df: pl.DataFrame, column: str = "close", fast_period: int = 12, 
                  slow_period: int = 26, signal_period: int = 9, symbol_col: Optional[str] = None) -> pl.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) using Polars
    
    Args:
        df: Polars DataFrame with price data
        column: Column name for price data (default: "close")
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        symbol_col: Column name for symbol grouping (default: None)
    
    Returns:
        DataFrame with MACD, Signal Line, and Histogram columns added
    """
    # Calculate alpha values for EMAs
    alpha_fast = 2.0 / (fast_period + 1.0)
    alpha_slow = 2.0 / (slow_period + 1.0)
    alpha_signal = 2.0 / (signal_period + 1.0)
    
    if symbol_col is not None:
        return df.with_columns([
            # Calculate fast and slow EMAs
            pl.col(column).ewm_mean(alpha=alpha_fast).over(symbol_col).alias(f"ema_{fast_period}"),
            pl.col(column).ewm_mean(alpha=alpha_slow).over(symbol_col).alias(f"ema_{slow_period}"),
        ]).with_columns([
            # Calculate MACD line
            (pl.col(f"ema_{fast_period}") - pl.col(f"ema_{slow_period}")).alias("macd")
        ]).with_columns([
            # Calculate Signal line
            pl.col("macd").ewm_mean(alpha=alpha_signal).over(symbol_col).alias("signal_line")
        ]).with_columns([
            # Calculate MACD Histogram
            (pl.col("macd") - pl.col("signal_line")).alias("macd_histogram")
        ]).drop([f"ema_{fast_period}", f"ema_{slow_period}"])
    else:
        return df.with_columns([
            # Calculate fast and slow EMAs
            pl.col(column).ewm_mean(alpha=alpha_fast).alias(f"ema_{fast_period}"),
            pl.col(column).ewm_mean(alpha=alpha_slow).alias(f"ema_{slow_period}"),
        ]).with_columns([
            # Calculate MACD line
            (pl.col(f"ema_{fast_period}") - pl.col(f"ema_{slow_period}")).alias("macd")
        ]).with_columns([
            # Calculate Signal line
            pl.col("macd").ewm_mean(alpha=alpha_signal).alias("signal_line")
        ]).with_columns([
            # Calculate MACD Histogram
            (pl.col("macd") - pl.col("signal_line")).alias("macd_histogram")
        ]).drop([f"ema_{fast_period}", f"ema_{slow_period}"])


def generate_ma_signals(df: pl.DataFrame, short_ma_col: str, long_ma_col: str) -> pl.DataFrame:
    """
    Generate buy/sell signals based on moving average crossovers
    
    Args:
        df: Polars DataFrame with moving averages
        short_ma_col: Column name for short-term moving average
        long_ma_col: Column name for long-term moving average
    
    Returns:
        DataFrame with buy and sell signal columns added
    """
    return df.with_columns([
        (pl.col(short_ma_col) > pl.col(long_ma_col)).alias("ma_buy_signal"),
        (pl.col(short_ma_col) < pl.col(long_ma_col)).alias("ma_sell_signal")
    ])


def generate_macd_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate buy/sell signals based on MACD crossovers
    
    Args:
        df: Polars DataFrame with MACD and Signal Line
    
    Returns:
        DataFrame with MACD buy and sell signal columns added
    """
    return df.with_columns([
        (pl.col("macd") > pl.col("signal_line")).alias("macd_buy_signal"),
        (pl.col("macd") < pl.col("signal_line")).alias("macd_sell_signal")
    ])


def generate_rsi_signals(df: pl.DataFrame, rsi_col: str = "rsi_14", 
                        oversold: float = 30, overbought: float = 70) -> pl.DataFrame:
    """
    Generate buy/sell signals based on RSI levels
    
    Args:
        df: Polars DataFrame with RSI
        rsi_col: Column name for RSI (default: "rsi_14")
        oversold: RSI level for oversold condition (default: 30)
        overbought: RSI level for overbought condition (default: 70)
    
    Returns:
        DataFrame with RSI buy and sell signal columns added
    """
    return df.with_columns([
        (pl.col(rsi_col) < oversold).alias("rsi_buy_signal"),
        (pl.col(rsi_col) > overbought).alias("rsi_sell_signal")
    ])


def generate_combined_signals(df: pl.DataFrame, 
                            signal_cols_buy: list, 
                            signal_cols_sell: list) -> pl.DataFrame:
    """
    Generate combined buy/sell signals from multiple indicators
    
    Args:
        df: Polars DataFrame with individual signals
        signal_cols_buy: List of buy signal column names
        signal_cols_sell: List of sell signal column names
    
    Returns:
        DataFrame with combined signal columns added
    """
    # Create expressions for combined signals
    buy_expr = pl.col(signal_cols_buy[0])
    for col in signal_cols_buy[1:]:
        buy_expr = buy_expr & pl.col(col)
    
    sell_expr = pl.col(signal_cols_sell[0])
    for col in signal_cols_sell[1:]:
        sell_expr = sell_expr & pl.col(col)
    
    return df.with_columns([
        buy_expr.alias("combined_buy_signal"),
        sell_expr.alias("combined_sell_signal")
    ])


def add_all_indicators(df: pl.DataFrame, price_col: str = "close", 
                      symbol: Union[str, List[str]] = "all", symbol_col: str = "symbol") -> pl.DataFrame:
    """
    Add all technical indicators and signals to a DataFrame
    
    Args:
        df: Polars DataFrame with price data
        price_col: Column name for price data (default: "close")
        symbol: Symbol(s) to process - "all" for all symbols, or specific symbol(s) (default: "all")
        symbol_col: Column name for symbol grouping (default: "symbol")
    
    Returns:
        DataFrame with all indicators and signals added
    """
    # Filter data based on symbol parameter
    if symbol != "all":
        if isinstance(symbol, str):
            filtered_df = df.filter(pl.col(symbol_col) == symbol)
        elif isinstance(symbol, list):
            filtered_df = df.filter(pl.col(symbol_col).is_in(symbol))
        else:
            raise ValueError("symbol must be 'all', a string, or a list of strings")
    else:
        filtered_df = df
    
    # Determine if we need symbol grouping
    symbol_grouping = symbol_col if symbol_col in filtered_df.columns else None
    
    # Add moving averages
    result = calculate_sma(filtered_df, price_col, 20, "sma_20", symbol_grouping)
    result = calculate_sma(result, price_col, 50, "sma_50", symbol_grouping)
    result = calculate_ema(result, price_col, 20, "ema_20", symbol_grouping)
    
    # Add RSI
    result = calculate_rsi(result, price_col, 14, "rsi_14", symbol_grouping)
    
    # Add MACD
    result = calculate_macd(result, price_col, symbol_col=symbol_grouping)
    
    # Generate signals
    result = generate_ma_signals(result, "sma_20", "sma_50")
    result = generate_macd_signals(result)
    result = generate_rsi_signals(result, "rsi_14")
    
    # Generate combined signals
    result = generate_combined_signals(
        result, 
        ["ma_buy_signal", "macd_buy_signal"],
        ["ma_sell_signal", "macd_sell_signal"]
    )
    
    return result


def get_signals_summary(df: pl.DataFrame, symbol_col: str = "symbol") -> pl.DataFrame:
    """
    Get a summary of current signals by symbol
    
    Args:
        df: DataFrame with technical indicators
        symbol_col: Column name for symbol grouping
    
    Returns:
        Summary DataFrame with latest signals by symbol
    """
    return df.group_by(symbol_col).agg([
        pl.col("date").max().alias("latest_date"),
        pl.col("closing_price").last().alias("latest_price"),
        pl.col("sma_20").last().alias("sma_20"),
        pl.col("sma_50").last().alias("sma_50"),
        pl.col("rsi_14").last().alias("rsi_14"),
        pl.col("combined_buy_signal").last().alias("buy_signal"),
        pl.col("combined_sell_signal").last().alias("sell_signal")
    ]).sort("latest_date", descending=True)


def filter_buy_signals(df: pl.DataFrame, symbol_col: str = "symbol") -> pl.DataFrame:
    """
    Filter symbols with current buy signals
    
    Args:
        df: DataFrame with technical indicators
        symbol_col: Column name for symbol grouping
    
    Returns:
        DataFrame filtered to symbols with buy signals
    """
    latest_data = df.group_by(symbol_col).agg([
        pl.col("date").max().alias("latest_date"),
        pl.col("combined_buy_signal").last().alias("latest_buy_signal")
    ])
    
    symbols_with_buy = latest_data.filter(
        pl.col("latest_buy_signal") == True
    )[symbol_col].to_list()
    
    return df.filter(pl.col(symbol_col).is_in(symbols_with_buy))


# Example usage function
def example_usage():
    """
    Example of how to use the technical indicators with sample data
    """
    # Create sample data (replace with actual data loading)
    import datetime
    
    dates = pl.date_range(
        start=datetime.date(2023, 1, 1),
        end=datetime.date(2024, 1, 1),
        interval="1d"
    )
    
    # Generate sample price data
    np.random.seed(42)
    n = len(dates)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    # Create DataFrame
    df = pl.DataFrame({
        "date": dates,
        "close": prices,
        "volume": np.random.randint(1000, 10000, n)
    })
    
    # Add all indicators
    df_with_indicators = add_all_indicators(df, "close")
    
    # Display sample results
    print("Sample data with technical indicators:")
    print(df_with_indicators.tail(10).select([
        "date", "close", "sma_20", "sma_50", "ema_20", 
        "rsi_14", "macd", "signal_line", "combined_buy_signal", "combined_sell_signal"
    ]))
    
    return df_with_indicators

if __name__ == "__main__":
    example_usage()


# Usage examples for utility functions:
"""
# Example usage of utility functions:

# 1. Get latest signals summary
signals_summary = get_signals_summary(technical_df)
print(signals_summary)

# 2. Filter symbols with buy signals
buy_signals_df = filter_buy_signals(technical_df)
print("Symbols with buy signals:")
print(buy_signals_df.select(["symbol", "date", "closing_price", "combined_buy_signal"]).tail(10))

# 3. Get count of symbols with buy/sell signals
latest_signals = get_signals_summary(technical_df)
buy_count = latest_signals.filter(pl.col("buy_signal") == True).height
sell_count = latest_signals.filter(pl.col("sell_signal") == True).height
print(f"Symbols with buy signals: {buy_count}")
print(f"Symbols with sell signals: {sell_count}")
"""