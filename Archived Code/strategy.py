# --- strategy.py ---
"""Module for defining trading strategies."""

import polars as pl

class Strategy:
    """Base class for trading strategies."""
    
    def __init__(self, name, parameters=None):
        """
        Initialize the strategy.
        
        Args:
            name (str): Strategy name
            parameters (dict, optional): Strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
    
    def generate_signals(self, data):
        """
        Generate trading signals based on strategy rules.
        Must be implemented by subclasses.
        
        Args:
            data (pl.DataFrame): Market data
            
        Returns:
            pl.DataFrame: Data with additional columns for signals
        """
        raise NotImplementedError("Subclasses must implement generate_signals method")

class MACrossoverStrategy(Strategy):
    """Moving Average Crossover Strategy."""
    
    def generate_signals(self, data, price_col='close'):
        """
        Generate signals based on moving average crossovers.
        
        Args:
            data (pl.DataFrame): Market data
            price_col (str): Column name for price data
            
        Returns:
            pl.DataFrame: Data with signal columns
        """
        short_window = self.parameters.get('short_window', 50)
        long_window = self.parameters.get('long_window', 200)
        
        # Calculate moving averages
        data = data.with_columns([
            pl.col(price_col).rolling_mean(short_window).alias(f"ma_{short_window}"),
            pl.col(price_col).rolling_mean(long_window).alias(f"ma_{long_window}")
        ])
        
        # Generate signals based on crossovers (1 for long, -1 for short, 0 for no position)
        data = data.with_columns([
            (pl.col(f"ma_{short_window}") > pl.col(f"ma_{long_window}")).cast(pl.Int8).alias("signal")
        ])
        
        # Calculate position changes (1 for buy, -1 for sell, 0 for hold)
        data = data.with_columns([
            pl.col("signal").diff().fill_null(0).alias("position_change")
        ])
        
        return data

class RSIStrategy(Strategy):
    """Relative Strength Index (RSI) Strategy."""
    
    def generate_signals(self, data, price_col='close'):
        """
        Generate signals based on RSI.
        
        Args:
            data (pl.DataFrame): Market data
            price_col (str): Column name for price data
            
        Returns:
            pl.DataFrame: Data with signal columns
        """
        period = self.parameters.get('period', 14)
        oversold = self.parameters.get('oversold', 30)
        overbought = self.parameters.get('overbought', 70)
        
        # Calculate price changes
        data = data.with_columns([
            pl.col(price_col).diff().alias("price_change")
        ])
        
        # Calculate gains and losses
        data = data.with_columns([
            pl.when(pl.col("price_change") > 0).then(pl.col("price_change")).otherwise(0).alias("gains"),
            pl.when(pl.col("price_change") < 0).then(-pl.col("price_change")).otherwise(0).alias("losses")
        ])
        
        # Calculate average gains and losses
        data = data.with_columns([
            pl.col("gains").rolling_mean(period).alias("avg_gain"),
            pl.col("losses").rolling_mean(period).alias("avg_loss")
        ])
        
        # Calculate RS and RSI
        data = data.with_columns([
            (pl.col("avg_gain") / pl.col("avg_loss").clip_min(0.0001)).alias("rs"),
            (100 - (100 / (1 + (pl.col("avg_gain") / pl.col("avg_loss").clip_min(0.0001))))).alias("rsi")
        ])
        
        # Generate signals
        data = data.with_columns([
            pl.when(pl.col("rsi") < oversold).then(1)
              .when(pl.col("rsi") > overbought).then(-1)
              .otherwise(0).alias("signal")
        ])
        
        # Calculate position changes
        data = data.with_columns([
            pl.col("signal").diff().fill_null(0).alias("position_change")
        ])
        
        return data

class BollingerBandsStrategy(Strategy):
    """Bollinger Bands Strategy."""
    
    def generate_signals(self, data, price_col='close'):
        """
        Generate signals based on Bollinger Bands.
        
        Args:
            data (pl.DataFrame): Market data
            price_col (str): Column name for price data
            
        Returns:
            pl.DataFrame: Data with signal columns
        """
        period = self.parameters.get('period', 20)
        std_dev = self.parameters.get('std_dev', 2)
        
        # Calculate moving average
        data = data.with_columns([
            pl.col(price_col).rolling_mean(period).alias("middle_band")
        ])
        
        # Calculate standard deviation
        data = data.with_columns([
            pl.col(price_col).rolling_std(period).alias("std_dev")
        ])
        
        # Calculate upper and lower bands
        data = data.with_columns([
            (pl.col("middle_band") + (pl.col("std_dev") * std_dev)).alias("upper_band"),
            (pl.col("middle_band") - (pl.col("std_dev") * std_dev)).alias("lower_band")
        ])
        
        # Generate signals
        data = data.with_columns([
            pl.when(pl.col(price_col) < pl.col("lower_band")).then(1)
              .when(pl.col(price_col) > pl.col("upper_band")).then(-1)
              .otherwise(0).alias("signal")
        ])
        
        # Calculate position changes
        data = data.with_columns([
            pl.col("signal").diff().fill_null(0).alias("position_change")
        ])
        
        return data

# Factory pattern for creating strategies
def create_strategy(strategy_type, **kwargs):
    """
    Factory function to create strategy instances.
    
    Args:
        strategy_type (str): Type of strategy
        **kwargs: Strategy parameters
        
    Returns:
        Strategy: Strategy instance
    """
    strategies = {
        'ma_crossover': MACrossoverStrategy,
        'rsi': RSIStrategy,
        'bollinger_bands': BollingerBandsStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    name = kwargs.pop('name', strategy_type)
    return strategies[strategy_type](name, kwargs)
