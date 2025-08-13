# --- backtest_engine.py ---
"""Module for running backtests."""

import polars as pl
from datetime import datetime
import pandas as pd
from decimal import Decimal
from config import BACKTEST_CONFIG

class BacktestEngine:
    """Executes backtests for trading strategies."""
    
    def __init__(self, initial_capital=None, commission=None, slippage=None):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital (float, optional): Initial capital
            commission (float, optional): Commission rate per trade
            slippage (float, optional): Slippage rate per trade
        """
        self.initial_capital = initial_capital or BACKTEST_CONFIG['initial_capital']
        self.commission = commission or BACKTEST_CONFIG['commission']
        self.slippage = slippage or BACKTEST_CONFIG['slippage']
        self.trades = []
    
    def to_float(self, value):
        """
        Convert a value to float, handling Decimal types.
        
        Args:
            value: Value to convert
            
        Returns:
            float: Converted value
        """
        if isinstance(value, Decimal):
            return float(value)
        return float(value)
    
    def run(self, data, strategy, price_col='close', date_col='date'):
        """
        Run backtest with the given data and strategy.
        
        Args:
            data (pl.DataFrame): Market data
            strategy (Strategy): Trading strategy
            price_col (str): Column name for price data
            date_col (str): Column name for date data
            
        Returns:
            tuple: (performance DataFrame, trades list)
        """
        # Generate signals
        data_with_signals = strategy.generate_signals(data)
        
        # Initialize portfolio tracking
        current_position = 0
        capital = self.initial_capital
        portfolio_values = []
        self.trades = []
        
        # Simulate trading
        for row in data_with_signals.iter_rows(named=True):
            price = self.to_float(row[price_col])
            date = row[date_col]
            position_change = row.get("position_change", 0)
            
            # Buy signal
            if position_change > 0:
                # Calculate shares to buy (use all available capital)
                shares_bought = int(capital / price)
                
                if shares_bought > 0:
                    # Calculate actual execution price with slippage
                    execution_price = price * (1 + self.slippage)
                    
                    # Calculate cost including commission
                    cost = shares_bought * execution_price * (1 + self.commission)
                    
                    # Record trade
                    self.trades.append({
                        'date': date,
                        'type': 'BUY',
                        'shares': shares_bought,
                        'price': execution_price,
                        'cost': cost,
                        'commission': shares_bought * execution_price * self.commission
                    })
                    
                    # Update capital
                    capital -= cost
                    current_position = shares_bought
            
            # Sell signal
            elif position_change < 0 and current_position > 0:
                # Calculate execution price with slippage
                execution_price = price * (1 - self.slippage)
                
                # Calculate proceeds including commission
                proceeds = current_position * execution_price * (1 - self.commission)
                
                # Record trade
                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'shares': current_position,
                    'price': execution_price,
                    'proceeds': proceeds,
                    'commission': current_position * execution_price * self.commission
                })
                
                # Update capital
                capital += proceeds
                current_position = 0
            
            # Calculate portfolio value
            portfolio_value = capital + (current_position * price)
            portfolio_values.append({
                date_col: date, 
                'value': portfolio_value,
                'cash': capital,
                'holdings': current_position * price,
                'position': current_position
            })
        
        # Create performance data
        performance = pl.DataFrame(portfolio_values)
        
        # Final portfolio value calculation (close any open positions)
        if current_position > 0:
            final_price = self.to_float(data_with_signals[price_col].tail(1)[0])
            print(f"Note: Closing {current_position} shares at final price {final_price}")
        
        return performance, self.trades
    
    def get_trades_df(self):
        """
        Get trades as a DataFrame.
        
        Returns:
            pl.DataFrame: Trades information
        """
        return pl.DataFrame(self.trades)