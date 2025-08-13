# --- data_loader.py ---
"""Module for loading market data from MySQL database."""

import polars as pl
import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_CONFIG, COLUMN_MAPPING

class DataLoader:
    """Handles database connections and data retrieval for backtesting."""
    
    def __init__(self, connection_params=None):
        """
        Initialize with database connection parameters.
        
        Args:
            connection_params (dict): MySQL connection parameters
                                     (host, user, password, database, port)
        """
        self.connection_params = connection_params or DB_CONFIG
        self.conn_str = (f"mysql+mysqlconnector://{self.connection_params['user']}:"
                        f"{self.connection_params['password']}@{self.connection_params['host']}:"
                        f"{self.connection_params['port']}/{self.connection_params['database']}")
        self.engine = create_engine(self.conn_str)
    
    def fetch_data(self, query=None, symbol=None, start_date=None, end_date=None, table=None):
        """
        Fetch market data using either a custom query or by specifying symbol and date range.
        
        Args:
            query (str, optional): Custom SQL query
            symbol (str, optional): Trading symbol/ticker
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            table (str, optional): Table name containing market data
            
        Returns:
            pl.DataFrame: Market data as a Polars DataFrame
        """
        if query is None and (symbol is None or table is None):
            raise ValueError("Either a custom query or symbol, date range, and table must be provided")
        
        # Build query if not provided
        if query is None:
            query = f"""
            SELECT 
                {COLUMN_MAPPING['date']} AS date,
                {COLUMN_MAPPING['open']} AS open,
                {COLUMN_MAPPING['high']} AS high,
                {COLUMN_MAPPING['low']} AS low,
                {COLUMN_MAPPING['close']} AS close,
                {COLUMN_MAPPING['volume']} AS volume
            FROM 
                {table}
            WHERE 
                symbol = :symbol
            """
            
            if start_date:
                query += f" AND {COLUMN_MAPPING['date']} >= :start_date"
            
            if end_date:
                query += f" AND {COLUMN_MAPPING['date']} <= :end_date"
                
            query += f" ORDER BY {COLUMN_MAPPING['date']} ASC"
        
        # Execute query with parameters
        with self.engine.connect() as conn:
            if symbol or start_date or end_date:
                params = {'symbol': symbol, 'start_date': start_date, 'end_date': end_date}
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            
            # Convert to pandas first (easier with SQLAlchemy)
            df_pandas = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Convert to Polars DataFrame
        df = pl.from_pandas(df_pandas)
        
        # Ensure date column is properly formatted
        if 'date' in df.columns:
            df = df.with_columns(pl.col('date').cast(pl.Datetime))
        
        return df
    
    def save_results(self, results_df, table_name, if_exists='append'):
        """
        Save backtest results to the database.
        
        Args:
            results_df (pl.DataFrame): Results to save
            table_name (str): Table name to save data to
            if_exists (str): How to handle existing data ('fail', 'replace', 'append')
        """
        # Convert Polars to Pandas for SQLAlchemy compatibility
        pandas_df = results_df.to_pandas()
        
        # Save to database
        pandas_df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        print(f"Results saved to {table_name} table")
    
    def get_available_symbols(self, table_name):
        """
        Get list of available symbols in the database.
        
        Args:
            table_name (str): Table name containing market data
            
        Returns:
            list: Available symbols
        """
        query = f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            symbols = [row[0] for row in result]
        
        return symbols
    
    def get_data_date_range(self, symbol, table_name):
        """
        Get the available date range for a symbol.
        
        Args:
            symbol (str): Trading symbol/ticker
            table_name (str): Table name containing market data
            
        Returns:
            tuple: (min_date, max_date) as datetime objects
        """
        query = f"""
        SELECT 
            MIN({COLUMN_MAPPING['date']}), 
            MAX({COLUMN_MAPPING['date']}) 
        FROM 
            {table_name} 
        WHERE 
            symbol = :symbol
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'symbol': symbol})
            min_date, max_date = result.fetchone()
        
        return min_date, max_date
