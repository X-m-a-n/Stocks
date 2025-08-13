# --- config.py ---
"""Configuration settings for the backtesting framework."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'market_data'),
    'port': int(os.getenv('DB_PORT', '3306')),
}

# Backtesting configuration
BACKTEST_CONFIG = {
    'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000')),
    'commission': float(os.getenv('COMMISSION', '0.001')),
    'slippage': float(os.getenv('SLIPPAGE', '0.0005')),
}

# Column name mapping
COLUMN_MAPPING = {
    'date': os.getenv('DATE_COLUMN', 'date_time'),
    'open': os.getenv('OPEN_COLUMN', 'open_price'),
    'high': os.getenv('HIGH_COLUMN', 'high_price'),
    'low': os.getenv('LOW_COLUMN', 'low_price'),
    'close': os.getenv('CLOSE_COLUMN', 'close_price'),
    'volume': os.getenv('VOLUME_COLUMN', 'volume'),
}
