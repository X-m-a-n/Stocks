-- Old Trade Data Entitlement Migration Script
-- Backs up old_trade_data table and implements entitlement tracking

USE market_data;

-- Step 1: Create backup of old_trade_data table
DROP TABLE IF EXISTS old_trade_data_backup;

-- Create backup table matching your actual old_trade_data structure
-- Removed unique constraint to handle duplicate symbol-date combinations
CREATE TABLE old_trade_data_backup (
    backup_id INT AUTO_INCREMENT PRIMARY KEY,
    id INT,
    symbol VARCHAR(50) NOT NULL,
    trade_date DATE NOT NULL,
    last_price DECIMAL(15,6),
    closing_price DECIMAL(15,6),
    price_change DECIMAL(15,6),
    bid DECIMAL(15,6),
    ask DECIMAL(15,6),
    volume BIGINT,
    daily_range_low DECIMAL(15,6),
    daily_range_high DECIMAL(15,6),
    year_range_low DECIMAL(15,6),
    year_range_high DECIMAL(15,6),
    created_at TIMESTAMP,
    backup_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_symbol (symbol),
    INDEX idx_trade_date (trade_date),
    INDEX idx_id (id),
    INDEX idx_symbol_date (symbol, trade_date)
);

-- Insert all data from old_trade_data (including duplicates)
INSERT INTO old_trade_data_backup (
    id, symbol, trade_date, last_price, closing_price, price_change, 
    bid, ask, volume, daily_range_low, daily_range_high, 
    year_range_low, year_range_high, created_at
)
SELECT 
    id, symbol, trade_date, last_price, closing_price, price_change,
    bid, ask, volume, daily_range_low, daily_range_high,
    year_range_low, year_range_high, created_at
FROM old_trade_data;

-- Verify backup creation
SELECT 
    COUNT(*) as backup_record_count,
    MIN(trade_date) as earliest_date,
    MAX(trade_date) as latest_date
FROM old_trade_data_backup;

-- Step 2: Add entitlement_status and clean_symbol columns to old_trade_data
-- Add entitlement_status column if it doesn't exist
-- ALTER TABLE old_trade_data ADD COLUMN entitlement_status VARCHAR(5) NULL AFTER year_range_high;

-- Add clean_symbol column if it doesn't exist
-- ALTER TABLE old_trade_data ADD COLUMN clean_symbol VARCHAR(50) NULL AFTER entitlement_status;

-- Add indexes manually if needed (uncomment and run separately)
-- CREATE INDEX idx_entitlement_status_old ON old_trade_data (entitlement_status);
-- CREATE INDEX idx_clean_symbol_old ON old_trade_data (clean_symbol);

-- Step 3: Extract entitlement status and clean symbols from old_trade_data
-- Update records with entitlement patterns like SYMBOL(XD), SYMBOL(XB), etc.
UPDATE old_trade_data 
SET 
    entitlement_status = CASE 
        WHEN symbol REGEXP '\\(XD\\)$' THEN 'XD'
        WHEN symbol REGEXP '\\(XB\\)$' THEN 'XB' 
        WHEN symbol REGEXP '\\(XS\\)$' THEN 'XS'
        WHEN symbol REGEXP '\\(XR\\)$' THEN 'XR'
        WHEN symbol REGEXP '\\(XC\\)$' THEN 'XC'
        WHEN symbol REGEXP '\\(XW\\)$' THEN 'XW'
        ELSE entitlement_status  -- Keep existing value if no pattern matches
    END,
    clean_symbol = CASE
        WHEN symbol REGEXP '\\([A-Z]{2,3}\\)$' THEN 
            TRIM(REGEXP_REPLACE(symbol, '\\([A-Z]{2,3}\\)$', ''))
        ELSE symbol
    END;

-- Step 4: Populate entitlement_history from old_trade_data (append to existing data)
-- First, create a temporary table with the aggregated data and primary key
CREATE TEMPORARY TABLE temp_old_entitlements (
    temp_id INT AUTO_INCREMENT PRIMARY KEY,
    clean_symbol VARCHAR(50),
    entitlement_status VARCHAR(5),
    effective_date DATE,
    ex_date DATE,
    notes TEXT,
    
    INDEX idx_symbol_entitlement (clean_symbol, entitlement_status, effective_date)
);

-- Insert aggregated data into temporary table
INSERT INTO temp_old_entitlements (clean_symbol, entitlement_status, effective_date, ex_date, notes)
SELECT 
    clean_symbol,
    entitlement_status,
    MIN(trade_date) as effective_date,
    MIN(trade_date) as ex_date,
    'Migrated from old_trade_data table' as notes
FROM old_trade_data 
WHERE entitlement_status IS NOT NULL
    AND clean_symbol IS NOT NULL
GROUP BY clean_symbol, entitlement_status;

-- Insert only records that don't already exist in entitlement_history
INSERT IGNORE INTO entitlement_history (symbol, entitlement_code, effective_date, ex_date, notes)
SELECT 
    t.clean_symbol,
    t.entitlement_status,
    t.effective_date,
    t.ex_date,
    t.notes
FROM temp_old_entitlements t
WHERE NOT EXISTS (
    SELECT 1 FROM entitlement_history eh 
    WHERE eh.symbol = t.clean_symbol 
    AND eh.entitlement_code = t.entitlement_status
    AND eh.effective_date = t.effective_date
)
ORDER BY t.clean_symbol, t.effective_date;

-- Drop the temporary table
DROP TEMPORARY TABLE temp_old_entitlements;

-- Step 5: Create stored procedure for old trade data entitlement management
DELIMITER //

DROP PROCEDURE IF EXISTS UpdateOldTradeDataEntitlement//

CREATE PROCEDURE UpdateOldTradeDataEntitlement(
    IN p_original_symbol VARCHAR(50),
    IN p_clean_symbol VARCHAR(50),
    IN p_entitlement_code VARCHAR(5)
)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;
    
    START TRANSACTION;
    
    UPDATE old_trade_data 
    SET 
        clean_symbol = p_clean_symbol,
        entitlement_status = p_entitlement_code
    WHERE symbol = p_original_symbol;
    
    COMMIT;
    
    SELECT CONCAT('Updated ', ROW_COUNT(), ' records in old_trade_data') as result;
END//

DELIMITER ;

-- Step 6: Data validation and quality checks for old_trade_data
-- Check for any unmatched entitlement patterns
SELECT 
    'OLD TRADE DATA - Unmatched Entitlement Patterns' as check_type,
    symbol,
    COUNT(*) as occurrences
FROM old_trade_data 
WHERE symbol REGEXP '\\([A-Z]+\\)$' 
    AND entitlement_status IS NULL
GROUP BY symbol
HAVING COUNT(*) > 0

UNION ALL

-- Check entitlement distribution in old_trade_data
SELECT 
    'OLD TRADE DATA - Entitlement Distribution' as check_type,
    COALESCE(entitlement_status, 'NONE') as status,
    COUNT(*) as count
FROM old_trade_data
GROUP BY entitlement_status

UNION ALL

-- Check clean symbol validation
SELECT 
    'OLD TRADE DATA - Clean Symbol Stats' as check_type,
    'Total Clean Symbols' as status,
    COUNT(DISTINCT clean_symbol) as count
FROM old_trade_data
WHERE clean_symbol IS NOT NULL;

-- Step 7: Final verification queries for old_trade_data migration
SELECT 'OLD TRADE DATA MIGRATION SUMMARY' as summary_type, '' as detail, '' as count
UNION ALL
SELECT 'Backup Records', 'Total backed up', COUNT(*) FROM old_trade_data_backup
UNION ALL
SELECT 'Original Records', 'Total in main table', COUNT(*) FROM old_trade_data
UNION ALL
SELECT 'Records with Entitlements', 'Have entitlement status', COUNT(*) FROM old_trade_data WHERE entitlement_status IS NOT NULL
UNION ALL
SELECT 'Unique Clean Symbols', 'Distinct symbols', COUNT(DISTINCT clean_symbol) FROM old_trade_data WHERE clean_symbol IS NOT NULL
UNION ALL
SELECT 'Date Range', 'Earliest trade date', MIN(trade_date) FROM old_trade_data
UNION ALL
SELECT 'Date Range', 'Latest trade date', MAX(trade_date) FROM old_trade_data;

-- Sample queries to verify the old_trade_data migration
SELECT 
    'OLD TRADE DATA - Sample Entitlement Data' as section,
    clean_symbol,
    entitlement_status,
    COUNT(*) as record_count,
    MIN(trade_date) as first_date,
    MAX(trade_date) as last_date
FROM old_trade_data 
WHERE entitlement_status IS NOT NULL
GROUP BY clean_symbol, entitlement_status
ORDER BY clean_symbol, entitlement_status
LIMIT 10;

-- Combined view query to see entitlements across both tables
SELECT 
    'COMBINED ENTITLEMENT SUMMARY' as source_table,
    clean_symbol,
    entitlement_status,
    COUNT(*) as total_records,
    MIN(trade_date) as earliest_date,
    MAX(trade_date) as latest_date
FROM (
    SELECT clean_symbol, entitlement_status, trade_date FROM historical_summary WHERE entitlement_status IS NOT NULL
    UNION ALL
    SELECT clean_symbol, entitlement_status, trade_date FROM old_trade_data WHERE entitlement_status IS NOT NULL
) combined_data
GROUP BY clean_symbol, entitlement_status
ORDER BY clean_symbol, entitlement_status;

-- Usage Examples:
-- 
-- Update old trade data entitlement:
-- CALL UpdateOldTradeDataEntitlement('GK(XD)', 'GK', 'XD');
--
-- View all entitlements from both tables:
-- SELECT * FROM entitlement_history ORDER BY symbol, effective_date;
--
-- Find symbols with entitlements in old data:
-- SELECT DISTINCT clean_symbol, entitlement_status 
-- FROM old_trade_data 
-- WHERE entitlement_status IS NOT NULL;