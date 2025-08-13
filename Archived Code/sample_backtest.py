# --- sample_backtest.py ---
"""Example backtest script."""

from backtesting_framework.data_loader import DataLoader
from backtesting_framework.strategy import create_strategy
from backtesting_framework.backtest_engine import BacktestEngine
from backtesting_framework.performance import PerformanceAnalyzer
from backtesting_framework.utils import plot_performance, save_results_to_json
import matplotlib.pyplot as plt

def run_sample_backtest():
    """Run a sample backtest."""
    # Initialize components
    data_loader = DataLoader()
    
    # Check available symbols
    available_symbols = data_loader.get_available_symbols("price_history")
    if not available_symbols:
        print("No symbols found in database. Please import data first.")
        return
    
    print(f"Available symbols: {', '.join(available_symbols)}")
    symbol = available_symbols[0]  # Use first available symbol
    
    # Get date range for the symbol
    min_date, max_date = data_loader.get_data_date_range(symbol, "price_history")
    print(f"Available date range for {symbol}: {min_date} to {max_date}")
    
    # Fetch data
    market_data = data_loader.fetch_data(
        symbol=symbol,
        start_date=min_date,
        end_date=max_date,
        table="price_history"
    )
    
    print(f"Loaded {len(market_data)} data points")
    
    # Create strategy
    strategy = create_strategy("ma_crossover", short_window=50, long_window=200)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    performance, trades = engine.run(market_data, strategy)
    trades_df = engine.get_trades_df()
    
    # Analyze performance
    analyzer = PerformanceAnalyzer(initial_capital=100000)
    metrics = analyzer.calculate_metrics(performance)
    trade_metrics = analyzer.calculate_trade_metrics(trades_df)
    
    # Print metrics
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\nTrade Metrics:")
    for key, value in trade_metrics.items():
        print(f"  {key}: {value}")
    
    # Plot results
    fig = plot_performance(performance, f"{symbol} - {strategy.name}")
    plt.show()
    
    return performance, trades_df, metrics, trade_metrics

if __name__ == "__main__":
    run_sample_backtest()
