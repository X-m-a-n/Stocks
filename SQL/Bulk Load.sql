LOAD DATA LOCAL INFILE 'C:/Users/michaelsjo/Desktop/Stocks/Data/combined_trade_data.csv' 
INTO TABLE market_data.historical_summary
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(
    symbol,
    @trade_date,
    @last_price,
    @closing_price,
    @price_change,
    @bid,
    @ask,
    @volume,
    @trading_range_open,
    @trading_range_close,
    @year_range_open,
    @year_range_close
)
SET
    trade_date = STR_TO_DATE(@trade_date, '%Y-%m-%d'),
    last_price = NULLIF(@last_price, ''),
    closing_price = NULLIF(@closing_price, ''),
    price_change = NULLIF(@price_change, ''),
    bid = NULLIF(@bid, ''),
    ask = NULLIF(@ask, ''),
    volume = NULLIF(@volume, ''),
    trading_range_open = NULLIF(@trading_range_open, ''),
    trading_range_close = NULLIF(@trading_range_close, ''),
    year_range_open = NULLIF(@year_range_open, ''),
    year_range_close = NULLIF(@year_range_close, '');