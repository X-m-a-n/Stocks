# --- utils.py ---
"""Utility functions for the backtesting framework."""

import polars as pl
import matplotlib.pyplot as plt
import json
from pathlib import Path

def create_date_range(start_date, end_date, freq='D'):
    """Create a date range with the specified frequency."""
    return pl.date_range(start=start_date, end=end_date, interval=freq)

def plot_performance(performance_data, title="Backtest Performance", date_col='date', value_col='value'):
    """
    Plot performance chart.
    
    Args:
        performance_data (pl.DataFrame): Performance data
        title (str): Chart title
        date_col (str): Column name for date data
        value_col (str): Column name for portfolio value data
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to pandas for easier plotting
    df = performance_data.select([date_col, value_col]).to_pandas()
    df.set_index(date_col, inplace=True)
    
    # Plot portfolio value
    ax.plot(df.index, df[value_col], label="Portfolio Value")
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
    
    return fig

def save_results_to_json(metrics, trades, file_path):
    """
    Save backtest results to a JSON file.
    
    Args:
        metrics (dict): Performance metrics
        trades (list): List of trades
        file_path (str): Path to save the JSON file
    """
    # Convert complex types to strings for JSON serialization
    for key, value in metrics.items():
        if isinstance(value, (pl.Series, pl.DataFrame)):
            metrics[key] = value.to_dict()
    
    # Create result dictionary
    results = {
        "metrics": metrics,
        "trades": [trade for trade in trades]
    }
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"Results saved to {file_path}")

def create_report(performance_data, metrics, trades_df, strategy_name):
    """
    Create a detailed backtest report.
    
    Args:
        performance_data (pl.DataFrame): Performance data
        metrics (dict): Performance metrics
        trades_df (pl.DataFrame): Trades data
        strategy_name (str): Name of the strategy
        
    Returns:
        str: Report as string
    """
    # Convert values for display
    metrics_display = {k: f"{v:.2%}" if isinstance(v, float) and k.endswith(("return", "drawdown", "rate")) else v for k, v in metrics.items()}
    
    # Create report
    report = [
        f"# Backtest Report: {strategy_name}",
        "",
        "## Performance Metrics",
        f"- Initial Capital: ${performance_data['value'].head(1)[0]:.2f}",
        f"- Final Value: ${performance_data['value'].tail(1)[0]:.2f}",
        f"- Total Return: {metrics_display['total_return']}",
        f"- Annual Return: {metrics_display['annual_return']}",
        f"- Max Drawdown: {metrics_display['max_drawdown']}",
        f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
        f"- Win Rate: {metrics_display['win_rate']}",
        "",
        "## Trade Statistics",
        f"- Total Trades: {metrics['total_trades']}",
        f"- Winning Trades: {metrics['winning_trades']}",
        f"- Losing Trades: {metrics['losing_trades']}",
        f"- Trade Win Rate: {metrics['win_rate']:.2%}",
        f"- Average Profit: ${metrics['average_profit']:.2f}",
        f"- Average Win: ${metrics['average_win']:.2f}",
        f"- Average Loss: ${metrics['average_loss']:.2f}",
        f"- Profit Factor: {metrics['profit_factor']:.2f}",
        f"- Total Commission: ${metrics['total_commission']:.2f}",
        "",
        "## Performance Chart",
        "[Portfolio Value Chart]",
        "",
        "## Trade Log (First 5 trades)",
        str(trades_df.head(5))
    ]
    
    return "\n".join(report)
