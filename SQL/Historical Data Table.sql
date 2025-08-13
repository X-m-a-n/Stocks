CREATE DATABASE IF NOT EXISTS trade_data;
USE trade_data;

CREATE SCHEMA IF NOT EXISTS market_data;

-- Drop the table if it exists
DROP TABLE IF EXISTS market_data.historical_summary;

CREATE TABLE market_data.historical_summary (
    symbol VARCHAR(50) NOT NULL,
    trade_date DATE NOT NULL,
    last_price DECIMAL(10,2) NOT NULL,
    closing_price DECIMAL(10,2) NOT NULL, 
    price_change DECIMAL(10,2) NOT NULL,
    bid DECIMAL(10,2) NOT NULL,
    ask DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    trading_range_open DECIMAL(10,2) NOT NULL,
    trading_range_close DECIMAL(10,2) NOT NULL,
    year_range_open DECIMAL(10,2) NOT NULL,
    year_range_close DECIMAL(10,2) NOT NULL,
    PRIMARY KEY (symbol, trade_date)
);