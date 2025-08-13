# --- performance.py ---
"""Module for analyzing backtest performance."""

import polars as pl
import numpy as np
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    """Analyzes backtest performance."""
    
    def __init__(self, initial_capital, risk_free_rate=0.0):
        """
        Initialize the performance analyzer.
        
        Args:
            initial_capital (float): Initial capital
            risk_free_rate (float, optional): Annual risk-free rate
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self, performance_data, date_col='date', value_col='value'):
        """
        Calculate performance metrics.
        
        Args:
            performance_data (pl.DataFrame): Performance data
            date_col (str): Column name for date data
            value_col (str): Column name for portfolio value data
            
        Returns:
            dict: Performance metrics
        """
        # Calculate returns
        performance_data = performance_data.with_columns([
            (pl.col(value_col) / performance_data[value_col].head(1)[0] - 1).alias("cumulative_return"),
            (pl.col(value_col) / pl.col(value_col).shift(1) - 1).fill_null(0).alias("daily_return")
        ])
        
        # Calculate total return
        total_return = performance_data[value_col].tail(1)[0] / self.initial_capital - 1
        
        # Calculate annualized return
        first_date = performance_data[date_col].head(1)[0]
        last_date = performance_data[date_col].tail(1)[0]
        days = (last_date - first_date).total_seconds() / (24 * 60 * 60)
        annual_return = (1 + total_return) ** (365.25 / days) - 1
        
        # Calculate drawdown
        running_max = pl.Series(np.maximum.accumulate(performance_data[value_col].to_numpy()))
        drawdown = (performance_data[value_col] / running_max - 1).to_numpy()
        max_drawdown = np.min(drawdown)
        
        # Calculate volatility
        daily_returns = performance_data["daily_return"].drop_nulls()
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        daily_risk_free = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_returns = daily_returns - daily_risk_free
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        # Calculate win rate
        positive_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": abs(daily_returns.filter(daily_returns > 0).sum() / daily_returns.filter(daily_returns < 0).sum()) if daily_returns.filter(daily_returns < 0).sum() != 0 else float('inf')
        }
    
    def calculate_trade_metrics(self, trades_df):
        """
        Calculate trade-specific metrics.
        
        Args:
            trades_df (pl.DataFrame): Trades data
            
        Returns:
            dict: Trade metrics
        """
        if len(trades_df) == 0:
            return {"total_trades": 0}
        
        # Pair buy and sell trades
        buys = trades_df.filter(pl.col("type") == "BUY")
        sells = trades_df.filter(pl.col("type") == "SELL")
        
        if len(buys) == 0 or len(sells) == 0:
            return {"total_trades": len(trades_df)}
        
        # Calculate profit/loss for completed trades
        completed_trades = min(len(buys), len(sells))
        profits = []
        
        for i in range(completed_trades):
            buy_cost = buys["cost"][i]
            sell_proceeds = sells["proceeds"][i]
            profit = sell_proceeds - buy_cost
            profits.append(profit)
        
        profits_series = pl.Series(profits)
        
        # Calculate metrics
        winning_trades = (profits_series > 0).sum()
        losing_trades = (profits_series < 0).sum()
        
        return {
            "total_trades": completed_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / completed_trades if completed_trades > 0 else 0,
            "average_profit": profits_series.mean(),
            "average_win": profits_series.filter(profits_series > 0).mean() if winning_trades > 0 else 0,
            "average_loss": profits_series.filter(profits_series < 0).mean() if losing_trades > 0 else 0,
            "profit_factor": abs(profits_series.filter(profits_series > 0).sum() / profits_series.filter(profits_series < 0).sum()) if profits_series.filter(profits_series < 0).sum() != 0 else float('inf'),
            "total_commission": trades_df["commission"].sum()
        }
