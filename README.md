# --- README.md ---
# Modular Backtesting Framework

A flexible and modular framework for backtesting trading strategies using custom market data from MySQL.

## Features

- Modular design with clear separation of concerns
- MySQL database integration for market data
- Multiple built-in trading strategies
- Detailed performance analysis and reporting
- Configurable parameters via environment variables or command line
- Comprehensive trade tracking and metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/backtesting-framework.git
cd backtesting-framework

# Install the package
pip install -e .
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your database credentials and other settings.

## Usage

### Command Line Interface

```bash
# Run a simple backtest
backtest --symbol AAPL --start-date 2020-01-01 --end-date 2023-01-01 --strategy ma_crossover

# Specify strategy parameters
backtest --symbol AAPL --start-date 2020-01-01 --end-date 2023-01-01 --strategy ma_crossover --params '{"short_window":50,"long_window":200}'

# Specify database table and output directory
backtest --symbol AAPL --start-date 2020-01-01 --end-date 2023-01-01 --strategy rsi --table my_price_table --output-dir ./my_results
```

### Python API

```python
from backtesting_framework.data_loader import DataLoader
from backtesting_framework.strategy import create_strategy
from backtesting_framework.backtest_engine import BacktestEngine
from backtesting_framework.performance import PerformanceAnalyzer
from backtesting_framework.utils import plot_performance

# Load data
data_loader = DataLoader()
market_data = data_loader.fetch_data(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2023-01-01",
    table="price_history"
)

# Create strategy
strategy = create_strategy("ma_crossover", short_window=50, long_window=200)

# Run backtest
engine = BacktestEngine(initial_capital=100000)
performance, trades = engine.run(market_data, strategy)

# Analyze performance
analyzer = PerformanceAnalyzer(initial_capital=100000)
metrics = analyzer.calculate_metrics(performance)
trade_metrics = analyzer.calculate_trade_metrics(engine.get_trades_df())

# Plot results
plot_performance(performance, "AAPL - MA Crossover Strategy")
```

## Available Strategies

- `ma_crossover`: Moving Average Crossover Strategy
  - `short_window`: Short-term moving average period (default: 50)
  - `long_window`: Long-term moving average period (default: 200)

- `rsi`: Relative Strength Index Strategy
  - `period`: RSI calculation period (default: 14)
  - `oversold`: Oversold threshold for buy signals (default: 30)
  - `overbought`: Overbought threshold for sell signals (default: 70)

- `bollinger_bands`: Bollinger Bands Strategy
  - `period`: Calculation period (default: 20)
  - `std_dev`: Standard deviation multiplier (default: 2)

## Database Schema

The framework expects a table with at least the following columns:
- `symbol`: Trading symbol/ticker
- `date_time`: Date and time of the data point
- `open_price`: Opening price
- `high_price`: High price
- `low_price`: Low price
- `close_price`: Closing price
- `volume`: Trading volume

You can customize the column names in the `.env` file.

## Example Schema SQL

```sql
CREATE TABLE price_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date_time DATETIME NOT NULL,
    open_price DECIMAL(15, 6) NOT NULL,
    high_price DECIMAL(15, 6) NOT NULL,
    low_price DECIMAL(15, 6) NOT NULL,
    close_price DECIMAL(15, 6) NOT NULL,
    volume BIGINT NOT NULL,
    INDEX idx_symbol_date (symbol, date_time)
);
```

## License

MIT

# --- backtesting_framework/__init__.py ---
"""Modular Backtesting Framework."""

__version__ = "0.1.0"

