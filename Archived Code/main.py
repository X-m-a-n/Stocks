# --- main.py ---
"""Main entry point for the backtesting framework."""

import polars as pl
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from data_loader import DataLoader
from strategy import create_strategy
from backtest_engine import BacktestEngine
from performance import PerformanceAnalyzer
from utils import plot_performance, save_results_to_json, create_report
from config import BACKTEST_CONFIG

def run_backtest(symbol, start_date, end_date, strategy_type, strategy_params, 
                 table_name="price_history", output_dir="results"):
    """
    Run a complete backtest.
    
    Args:
        symbol (str): Trading symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        strategy_type (str): Type of strategy
        strategy_params (dict): Strategy parameters
        table_name (str): Database table name
        output_dir (str): Directory to save results
        
    Returns:
        dict: Backtest results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_loader = DataLoader()
    strategy = create_strategy(strategy_type, **strategy_params)
    backtest_engine = BacktestEngine(
        initial_capital=BACKTEST_CONFIG['initial_capital'],
        commission=BACKTEST_CONFIG['commission'],
        slippage=BACKTEST_CONFIG['slippage']
    )
    performance_analyzer = PerformanceAnalyzer(initial_capital=BACKTEST_CONFIG['initial_capital'])
    
    print(f"Running backtest for {symbol} from {start_date} to {end_date}")
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.parameters}")
    
    # Fetch data
    market_data = data_loader.fetch_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        table=table_name
    )
    
    print(f"Loaded {len(market_data)} data points")
    
    # Run backtest
    performance_data, trades = backtest_engine.run(market_data, strategy)
    trades_df = pl.DataFrame(trades)
    
    # Analyze performance
    performance_metrics = performance_analyzer.calculate_metrics(performance_data)
    trade_metrics = performance_analyzer.calculate_trade_metrics(trades_df)
    
    # Combine metrics
    all_metrics = {**performance_metrics, **trade_metrics}
    
    # Plot performance
    fig = plot_performance(performance_data, f"{symbol} - {strategy.name}")
    
    # Create report
    report = create_report(performance_data, all_metrics, trades_df, strategy.name)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(output_dir) / f"{symbol}_{strategy_type}_{timestamp}"
    
    # Save portfolio values
    performance_data.write_csv(f"{results_path}_performance.csv")
    
    # Save trades
    if len(trades_df) > 0:
        trades_df.write_csv(f"{results_path}_trades.csv")
    
    # Save report
    with open(f"{results_path}_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    # Save figure
    fig.savefig(f"{results_path}_chart.png")
    
    # Save metrics to JSON
    save_results_to_json(all_metrics, trades, f"{results_path}_results.json")
    
    # Save to database if needed
    # data_loader.save_results(performance_data, "backtest_performance")
    
    print(f"Backtest completed. Results saved to {results_path}")
    
    return {
        "performance": performance_data,
        "trades": trades_df,
        "metrics": all_metrics,
        "report": report
    }

def main():
    """Command line interface for the backtesting framework."""
    parser = argparse.ArgumentParser(description="Backtesting Framework")
    
    # Required arguments
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategy", type=str, required=True, 
                        choices=["ma_crossover", "rsi", "bollinger_bands"],
                        help="Strategy type")
    
    # Optional arguments
    parser.add_argument("--table", type=str, default="historical_summary", 
                        help="Database table name")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    
    # Strategy parameters
    parser.add_argument("--params", type=str, default="{}", 
                        help="Strategy parameters as JSON string")
    
    args = parser.parse_args()
    
    # Parse strategy parameters
    try:
        strategy_params = json.loads(args.params)
    except json.JSONDecodeError:
        print("Error: Strategy parameters must be a valid JSON string")
        return
    
    # Run backtest
    run_backtest(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_type=args.strategy,
        strategy_params=strategy_params,
        table_name=args.table,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
