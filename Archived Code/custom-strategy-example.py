# --- custom_strategy_example.py ---
"""Example of creating and using a custom strategy."""

import polars as pl
from backtesting_framework.strategy import Strategy, create_strategy
from backtesting_framework.data_loader import DataLoader
from backtesting_framework.backtest_engine import BacktestEngine
from backtesting_framework.performance import PerformanceAnalyzer
from backtesting_framework.utils import plot_performance
import matplotlib.pyplot as plt

class MACDStrategy(Strategy):
    """Moving Average Convergence Divergence (MACD) Strategy."""
    
    def generate_signals(self, data, price_col='close'):
        """
        Generate signals based on MACD indicator.
        
        Args:
            data (pl.DataFrame): Market data
            price_col (str): Column name for price data
            
        Returns:
            pl.DataFrame: Data with signal columns
        """
        # Get strategy parameters
        fast_period = self.parameters.get('fast_period', 12)
        slow_period = self.parameters.get('slow_period', 26)
        signal_period = self.parameters.get('signal_period', 9)
        
        # Calculate EMAs
        data = data.with_columns([
            pl.col(price_col).ewm_mean(span=fast_period).alias(f"ema_{fast_period}"),
            pl.col(price_col).ewm_mean(span=slow_period).alias(f"ema_{slow_period}")
        ])
        
        # Calculate MACD line
        data = data.with_columns([
            (pl.col(f"ema_{fast_period}") - pl.col(f"ema_{slow_period}")).alias("macd_line")
        ])
        
        # Calculate signal line
        data = data.with_columns([
            pl.col("macd_line").ewm_mean(span=signal_period).alias("signal_line")
        ])
        
        # Calculate MACD histogram
        data = data.with_columns([
            (pl.col("macd_line") - pl.col("signal_line")).alias("macd_histogram")
        ])
        
        # Generate signals
        # Buy when MACD line crosses above signal line
        # Sell when MACD line crosses below signal line
        data = data.with_columns([
            pl.when(pl.col("macd_line") > pl.col("signal_line")).then(1)
              .when(pl.col("macd_line") < pl.col("signal_line")).then(-1)
              .otherwise(0).alias("signal")
        ])
        
        # Calculate position changes
        data = data.with_columns([
            pl.col("signal").diff().fill_null(0).alias("position_change")
        ])
        
        return data

def register_custom_strategy():
    """
    Register the custom strategy with the strategy factory.
    
    This allows the strategy to be created using the create_strategy function.
    """
    from backtesting_framework.strategy import strategies
    strategies['macd'] = MACDStrategy
    
    # Now we can create the strategy with:
    # create_strategy('macd', fast_period=12, slow_period=26, signal_period=9)
    
    return MACDStrategy

def run_custom_strategy_backtest(symbol="AAPL", start_date=None, end_date=None):
    """
    Run a backtest with the custom MACD strategy.
    
    Args:
        symbol (str): Trading symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    """
    # Register the custom strategy
    register_custom_strategy()
    
    # Load data
    data_loader = DataLoader()
    
    # If dates not provided, get available range
    if start_date is None or end_date is None:
        min_date, max_date = data_loader.get_data_date_range(symbol, "price_history")
        start_date = start_date or min_date
        end_date = end_date or max_date
    
    market_data = data_loader.fetch_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        table="price_history"
    )
    
    print(f"Loaded {len(market_data)} data points for {symbol}")
    
    # Create custom strategy
    strategy = create_strategy("macd", fast_period=12, slow_period=26, signal_period=9)
    
    # Alternative method without using factory:
    # strategy = MACDStrategy("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9})
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=100000)
    
    # Run backtest
    performance, trades = engine.run(market_data, strategy)
    trades_df = engine.get_trades_df()
    
    # Analyze performance
    analyzer = PerformanceAnalyzer(initial_capital=100000)
    metrics = analyzer.calculate_metrics(performance)
    trade_metrics = analyzer.calculate_trade_metrics(trades_df)
    
    # Print performance summary
    print(f"\nStrategy: {strategy.name}")
    print(f"Parameters: {strategy.parameters}")
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTrade Metrics:")
    for key, value in trade_metrics.items():
        print(f"  {key}: {value}")
    
    # Plot results
    fig = plot_performance(performance, f"{symbol} - MACD Strategy")
    plt.show()
    
    return performance, trades_df, metrics, trade_metrics

def compare_strategies(symbol="AAPL", start_date=None, end_date=None):
    """
    Compare multiple strategies on the same data.
    
    Args:
        symbol (str): Trading symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    """
    # Register the custom strategy
    register_custom_strategy()
    
    # Load data
    data_loader = DataLoader()
    
    # If dates not provided, get available range
    if start_date is None or end_date is None:
        min_date, max_date = data_loader.get_data_date_range(symbol, "price_history")
        start_date = start_date or min_date
        end_date = end_date or max_date
    
    market_data = data_loader.fetch_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        table="price_history"
    )
    
    print(f"Loaded {len(market_data)} data points for {symbol}")
    
    # Define strategies to compare
    strategies = [
        create_strategy("ma_crossover", short_window=50, long_window=200),
        create_strategy("rsi", period=14, oversold=30, overbought=70),
        create_strategy("macd", fast_period=12, slow_period=26, signal_period=9),
        create_strategy("bollinger_bands", period=20, std_dev=2)
    ]
    
    # Initialize backtest engine and performance analyzer
    engine = BacktestEngine(initial_capital=100000)
    analyzer = PerformanceAnalyzer(initial_capital=100000)
    
    # Run backtests and collect results
    results = []
    
    for strategy in strategies:
        print(f"\nRunning backtest with {strategy.name}...")
        performance, _ = engine.run(market_data, strategy)
        metrics = analyzer.calculate_metrics(performance)
        
        results.append({
            "strategy": strategy.name,
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "final_value": performance["value"].tail(1)[0]
        })
    
    # Print comparison
    print("\nStrategy Comparison:")
    print(f"{'Strategy':<20} {'Total Return':<15} {'Annual Return':<15} {'Max Drawdown':<15} {'Sharpe Ratio':<15} {'Final Value':<15}")
    print("-" * 95)
    
    for result in results:
        print(f"{result['strategy']:<20} {result['total_return']:>14.2%} {result['annual_return']:>14.2%} {result['max_drawdown']:>14.2%} {result['sharpe_ratio']:>14.2f} ${result['final_value']:>14,.2f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        performance, _ = engine.run(market_data, strategy)
        plt.plot(performance["date"], performance["value"], label=strategy.name)
    
    plt.title(f"Strategy Comparison - {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # Run backtest with custom strategy
    run_custom_strategy_backtest()
    
    # Compare multiple strategies
    # compare_strategies()
