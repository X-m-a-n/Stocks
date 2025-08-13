-- Step 1: Handle dual currency symbols first
UPDATE old_trade_data 
SET currency = (
    SELECT 
        CASE 
            WHEN daily_count > 1 THEN
                CASE 
                    WHEN old_trade_data.last_price = min_price THEN 'USD'
                    WHEN old_trade_data.last_price = max_price THEN 'JMD'
                    ELSE 'MID'
                END
            ELSE NULL
        END
    FROM (
        SELECT 
            clean_symbol,
            trade_date,
            COUNT(*) as daily_count,
            MIN(last_price) as min_price,
            MAX(last_price) as max_price
        FROM old_trade_data
        WHERE clean_symbol IN ('ASBH', 'MTL', 'MPCCEL', 'PROVEN', 'SIL', 'TJH')
            AND last_price > 0
        GROUP BY clean_symbol, trade_date
    ) daily_stats
    WHERE daily_stats.clean_symbol = old_trade_data.clean_symbol
        AND daily_stats.trade_date = old_trade_data.trade_date
)
WHERE clean_symbol IN ('ASBH', 'MTL', 'MPCCEL', 'PROVEN', 'SIL', 'TJH')
    AND last_price > 0;
    
    -- Step 2: Fill remaining NULL currencies from instruments table
UPDATE old_trade_data t1,
       jse_database.instruments i
SET t1.currency = COALESCE(i.currency, 'JMD')
WHERE t1.clean_symbol = i.symbol
    AND t1.currency IS NULL
    AND t1.last_price > 0;
    
    
    -- Step 3: Set any remaining NULL currencies to JMD (for symbols not in instruments table)
UPDATE old_trade_data 
SET currency = 'JMD'
WHERE currency IS NULL
    AND last_price > 0;
    
    