SELECT min(trade_date) FROM market_data.historical_summary;

SELECT year(trade_date), count(*) FROM market_data.historical_summary group by year(trade_date) order by year(trade_date);

SELECT count(*) FROM market_data.historical_summary where trade_date >= '2020-01-01';

select distinct trade_date from market_data.historical_summary;

select * from market_data.holidays;