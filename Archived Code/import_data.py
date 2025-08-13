# --- import_data.py ---
"""Script to import market data into MySQL."""

import argparse
import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text
from pathlib import Path
import os
from dotenv import load_dotenv

def connect_to_db():
    """Create database connection."""
    load_dotenv()
    
    # Get database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'market_data'),
        'port': int(os.getenv('DB_PORT', '3306')),
    }
    
    # Create connection string
    conn_str = (f"mysql+mysqlconnector://{db_config['user']}:"
                f"{db_config['password']}@{db_config['host']}:"
                f"{db_config['port']}/{db_config['database']}")
    
    # Create SQLAlchemy engine
    return create_engine(conn_str)

def create_schema(engine):
    """Create database schema if it doesn't exist."""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS price_history (
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
    """
    
    with engine.connect() as conn:
        conn.execute(text(schema_sql))
        conn.commit()
    
    print("Schema created or verified.")

def import_csv(file_path, symbol, engine):
    """Import CSV data into MySQL."""
    # Read CSV file
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check required columns
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
        return False
    
    # Rename columns to match database schema
    column_mapping = {
        'date': 'date_time',
        'open': 'open_price',
        'high': 'high_price',
        'low': 'low_price',
        'close': 'close_price',
        'volume': 'volume'
    }
    
    # Map only columns that exist in the dataframe
    valid_mapping = {old: new for old, new in column_mapping.items() if old in df.columns}
    df = df.rename(columns=valid_mapping)
    
    # Ensure date is in correct format
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Add symbol column
    df['symbol'] = symbol
    
    # Save to database
    print(f"Importing {len(df)} rows for {symbol}...")
    df.to_sql('price_history', engine, if_exists='append', index=False)
    
    print(f"Successfully imported {len(df)} rows for {symbol}")
    return True

def main():
    """Command line interface for importing data."""
    parser = argparse.ArgumentParser(description="Import market data into MySQL")
    
    parser.add_argument("file", type=str, help="CSV file path")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol")
    
    args = parser.parse_args()
    
    # Check if file exists
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {args.file} not found")
        return
    
    # Connect to database
    engine = connect_to_db()
    
    # Create schema
    create_schema(engine)
    
    # Import data
    import_csv(file_path, args.symbol, engine)

if __name__ == "__main__":
    main()
