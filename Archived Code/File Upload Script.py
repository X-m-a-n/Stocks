import polars as pl
import time
import mysql.connector
from mysql.connector import Error
import os
from typing import Optional

class DatabaseConnection:
    def __init__(self):
        self.config = {
            'user': 'avnadmin',
            'password': 'AVNS_byO0w2yQxY3CG_HS1CY',
            'host': 'mysql-29a0426d-origin.c.aivencloud.com',
            'port': 14872,
            'database': 'market_data',
            #'ssl': {
            #    'ssl_disabled': False,
            #    'ssl_verify_cert': True
            #}
        }
        self.connection = None
        
    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                print("Successfully connected to MySQL database")
                return True
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return False
            
    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")


class FileUploader:
    def __init__(self, db_connection):
        self.db = db_connection
        
    def upload_file_to_db(self, file_path, batch_size=5000):
        """
        Upload large CSV file to the old_trade_data table in batches
        
        Args:
            file_path: Path to the CSV file to upload
            batch_size: Number of records to insert in each batch
        """
        if not self.db.connection or not self.db.connection.is_connected():
            if not self.db.connect():
                print("Failed to connect to database. Exiting...")
                return False
                
        try:
            print(f"Reading file: {file_path}")
            # Use polars to efficiently read the CSV file
            df = pl.read_csv(file_path, try_parse_dates=True)
            
            print(f"File contains {df.height} records")
            
            # Display column names for debugging
            print(f"Columns in file: {df.columns}")
            
            # Process in batches to avoid memory issues
            total_records = df.height
            total_batches = (total_records + batch_size - 1) // batch_size
            
            cursor = self.db.connection.cursor()
            
            # Insert query for old_trade_data table
            insert_query = """
                INSERT INTO old_trade_data (
                    symbol, trade_date, last_price, closing_price, price_change,
                    bid, ask, volume, daily_range_low, daily_range_high,
                    year_range_low, year_range_high
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    last_price = VALUES(last_price),
                    closing_price = VALUES(closing_price),
                    price_change = VALUES(price_change),
                    bid = VALUES(bid),
                    ask = VALUES(ask),
                    volume = VALUES(volume),
                    daily_range_low = VALUES(daily_range_low),
                    daily_range_high = VALUES(daily_range_high),
                    year_range_low = VALUES(year_range_low),
                    year_range_high = VALUES(year_range_high)
            """
            
            records_inserted = 0
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_records)
                
                print(f"Processing batch {batch_num + 1}/{total_batches} (records {start_idx + 1}-{end_idx})")
                
                # Extract batch and convert to list of tuples
                batch_df = df.slice(start_idx, end_idx - start_idx)
                
                # Convert DataFrame batch to list of tuples for insertion
                data = [
                    (
                        row['symbol'],
                        row['trade_date'],
                        float(row['last_price']) if 'last_price' in row and row['last_price'] is not None else 0.0,
                        float(row['closing_price']) if 'closing_price' in row and row['closing_price'] is not None else 0.0,
                        float(row['price_change']) if 'price_change' in row and row['price_change'] is not None else 0.0,
                        float(row['bid']) if 'bid' in row and row['bid'] is not None else 0.0,
                        float(row['ask']) if 'ask' in row and row['ask'] is not None else 0.0,
                        int(row['volume']) if 'volume' in row and row['volume'] is not None else 0,
                        float(row['daily_range_low']) if 'daily_range_low' in row and row['daily_range_low'] is not None else 0.0,
                        float(row['daily_range_high']) if 'daily_range_high' in row and row['daily_range_high'] is not None else 0.0, 
                        float(row['year_range_low']) if 'year_range_low' in row and row['year_range_low'] is not None else 0.0,
                        float(row['year_range_high']) if 'year_range_high' in row and row['year_range_high'] is not None else 0.0
                    )
                    for row in batch_df.to_dicts()
                ]
                
                cursor.executemany(insert_query, data)
                self.db.connection.commit()
                
                records_inserted += len(data)
                print(f"Inserted {records_inserted}/{total_records} records")
                
                # Small delay between batches to reduce database load
                time.sleep(0.5)
                
            print(f"Successfully uploaded all {records_inserted} records to old_trade_data table")
            return True
            
        except Exception as e:
            print(f"Error uploading file to database: {e}")
            if self.db.connection:
                self.db.connection.rollback()
            return False
            
        finally:
            if cursor:
                cursor.close()


def main():
    db = DatabaseConnection()
    
    file_path = input("Enter the path to the CSV file: ")
    
    if os.path.exists(file_path):
        uploader = FileUploader(db)
        batch_size = 5000  # Default batch size
        try:
            custom_batch = input("Enter batch size (default 5000, press Enter to use default): ")
            if custom_batch:
                batch_size = int(custom_batch)
        except ValueError:
            print("Invalid batch size, using default 5000")
            
        uploader.upload_file_to_db(file_path, batch_size)
    else:
        print(f"File not found: {file_path}")
        
    print("Operation completed")


if __name__ == "__main__":
    main()