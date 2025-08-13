import requests
import polars as pl
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
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
            

class JSEScraper:
    def __init__(self):
        self.db = DatabaseConnection()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Jamaican holidays
        """self.holidays = {
            # 2024 holidays
            "2024-01-01", "2024-02-14", "2024-03-29", "2024-04-01",
            "2024-05-27", "2024-08-01", "2024-08-06", "2024-10-21",
            "2024-12-25", "2024-12-26",
            # 2025 holidays
            '2025-01-01', '2025-03-05', '2025-04-18', '2025-04-21',
            '2025-05-23', '2025-08-01', '2025-08-06', '2025-10-20'
            '2025-12-25', '2025-12-26'
        }"""

    def get_missing_dates(self, start_date: str, end_date: str) -> list:
        """Identify trading days that are missing from the database between start_date and end_date"""
        if not self.db.connection or not self.db.connection.is_connected():
            print("Database not connected")
            return []
        
        try:
            # Convert string dates to datetime objects
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get all potential trading days in the range
            current = start
            trading_days = []
            
            # Initialize holidays
            self.holidays = self.get_holidays()
            
            # Generate list of all trading days in range
            while current <= end:
                if self.is_trading_day(current):
                    trading_days.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)
            
            if not trading_days:
                return []
            
            # Query database for dates that already have data
            cursor = self.db.connection.cursor()
            
            # Format dates for SQL IN clause
            dates_str = ", ".join([f"'{date}'" for date in trading_days])
            
            # We'll check if each date has any records
            query = f"""
                SELECT DISTINCT trade_date 
                FROM historical_summary 
                WHERE trade_date IN ({dates_str})
            """
            
            cursor.execute(query)
            existing_dates = {row[0].strftime('%Y-%m-%d') for row in cursor.fetchall()}
            cursor.close()
            
            # Find missing dates
            missing_dates = [date for date in trading_days if date not in existing_dates]
            
            print(f"Found {len(missing_dates)} missing trading days out of {len(trading_days)} possible trading days")
            return missing_dates
        
        except Error as e:
            print(f"Error finding missing dates: {e}")
            return []

    def get_holidays(self):
        """Fetch holidays from the database."""
        try:
            # Query to select all holiday dates from the database
            query = "SELECT holiday_date FROM holidays ORDER BY holiday_date"
            
            # Execute query using the existing database connection
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                holidays = {row[0].strftime("%Y-%m-%d") for row in cursor.fetchall()}
            
            return holidays
        except Exception as e:
            print(f"Error fetching holidays: {e}")
            # Fall back to hardcoded holidays if database query fails
            return {
                # 2024 holidays
                "2024-01-01", "2024-02-14", "2024-03-29", "2024-04-01",
                "2024-05-27", "2024-08-01", "2024-08-06", "2024-10-21",
                "2024-12-25", "2024-12-26",
                # 2025 holidays
                "2025-01-01", "2025-03-05", "2025-04-18", "2025-04-21",
                "2025-05-23", "2025-08-01", "2025-08-06", "2025-10-20",
                "2025-12-25", "2025-12-26"
            }

    def is_trading_day(self, date: datetime) -> bool:
        """Check if given date is a trading day"""
        # Check if weekend
        if date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            return False
            
        # Check if holiday
        date_str = date.strftime('%Y-%m-%d')
        return date_str not in self.holidays

    def scrape_date(self, date: str, max_retries: int = 3) -> pl.DataFrame:
        """Scrape data for a specific date with retry logic"""
        url = f"https://www.jamstockex.com/trading/trade-quotes/?market=50&date={date}"
        
        for attempt in range(max_retries):
            try:
                print(f"Requesting data for: {date} (Attempt {attempt + 1}/{max_retries})")
                response = requests.get(url, headers=self.headers, timeout=30)
                
                # Check for potential blocking responses
                if response.status_code == 403:
                    print("Received 403 Forbidden - Possibly being rate limited")
                    sleep_time = 300 + random.uniform(0, 60)  # 5-6 minutes
                    print(f"Waiting {sleep_time:.0f} seconds before retrying...")
                    time.sleep(sleep_time)
                    continue
                    
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                data = []
                tables = soup.find_all('table')
                
                for table in tables:
                    headers = table.find_all('th')
                    if any('Symbol' in h.get_text(strip=True) for h in headers):
                        rows = table.find_all('tr')[1:]  # Skip header
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) >= 11:
                                try:
                                    # Split range into low and high
                                    daily_range = cells[8].get_text(strip=True)
                                    daily_low, daily_high = self.split_range(daily_range)
                                    
                                    # Split year range into low and high
                                    year_range = cells[9].get_text(strip=True)
                                    year_low, year_high = self.split_range(year_range)
                                    
                                    data.append({
                                        'symbol': cells[1].get_text(strip=True),
                                        'trade_date': date,
                                        'last_price': self.parse_float(cells[2].get_text(strip=True)),
                                        'closing_price': self.parse_float(cells[3].get_text(strip=True)),
                                        'price_change': self.parse_float(cells[4].get_text(strip=True)),
                                        'bid': self.parse_float(cells[5].get_text(strip=True)),
                                        'ask': self.parse_float(cells[6].get_text(strip=True)),
                                        'volume': self.parse_int(cells[7].get_text(strip=True)),
                                        'trading_range_open': daily_low,
                                        'trading_range_close': daily_high,
                                        'year_range_open': year_low,
                                        'year_range_close': year_high
                                    })
                                except Exception as e:
                                    print(f"Error parsing row: {str(e)}")
                                    continue
                
                return pl.DataFrame(data)
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = 60 + random.uniform(0, 30)  # 1-1.5 minutes
                    print(f"Waiting {sleep_time:.0f} seconds before retrying...")
                    time.sleep(sleep_time)
        
        print(f"Failed to scrape data for {date} after {max_retries} attempts")
        return pl.DataFrame()

    @staticmethod
    def parse_float(value: str) -> float:
        """Parse float values, handling special characters"""
        try:
            cleaned = value.strip().replace(',', '').replace('$', '')
            return float(cleaned) if cleaned else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def parse_int(value: str) -> int:
        """Parse integer values, handling special characters"""
        try:
            cleaned = value.strip().replace(',', '')
            return int(cleaned) if cleaned else 0
        except (ValueError, TypeError):
            return 0
            
    def split_range(self, range_str: str) -> tuple[float, float]:
        """Split a range string into low and high values"""
        try:
            if not range_str or range_str.strip() == '0.00 -  0.00':
                return 0.0, 0.0
                
            low, high = range_str.split('-')
            return (
                self.parse_float(low),
                self.parse_float(high)
            )
        except Exception as e:
            print(f"Error parsing range '{range_str}': {str(e)}")
            return 0.0, 0.0
        
    def save_to_db(self, df: pl.DataFrame, date: str):
        """Save scraped data to MySQL database"""
        if df.is_empty():
            return
            
        try:
            cursor = self.db.connection.cursor()
            
            # Convert DataFrame to list of tuples for insertion
            data = [
                (
                    row['symbol'],
                    row['trade_date'],
                    float(row['last_price']),
                    float(row['closing_price']),
                    float(row['price_change']),
                    float(row['bid']),
                    float(row['ask']),
                    int(row['volume']),
                    float(row['trading_range_open']),
                    float(row['trading_range_close']), 
                    float(row['year_range_open']),
                    float(row['year_range_close'])
                )
                for row in df.to_dicts()
            ]
                
            # Insert query
            insert_query = """
                INSERT INTO historical_summary (
                    symbol, trade_date, last_price, closing_price, price_change,
                    bid, ask, volume, trading_range_open, trading_range_close,
                    year_range_open, year_range_close
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    last_price = VALUES(last_price),
                    closing_price = VALUES(closing_price),
                    price_change = VALUES(price_change),
                    bid = VALUES(bid),
                    ask = VALUES(ask),
                    volume = VALUES(volume),
                    trading_range_open = VALUES(trading_range_open),
                    trading_range_close = VALUES(trading_range_close),
                    year_range_open = VALUES(year_range_open),
                    year_range_close = VALUES(year_range_close)
            """
            
            cursor.executemany(insert_query, data)
            self.db.connection.commit()
            print(f"Successfully saved {len(data)} records for {date}")
            
        except Error as e:
            print(f"Error saving data to database: {e}")
            self.db.connection.rollback()
        finally:
            cursor.close()

    def scrape_date_range(self, start_date: str, end_date: str):
        """Scrape data for missing dates in a date range and save to database"""
        if not self.db.connect():
            print("Failed to connect to database. Exiting...")
            return
            
        try:
            # Initialize holidays
            self.holidays = self.get_holidays()
            
            # Get dates missing from database
            missing_dates = self.get_missing_dates(start_date, end_date)
            
            if not missing_dates:
                print("No missing dates found. Exiting...")
                return
                
            print(f"Found {len(missing_dates)} missing dates to scrape")
            
            for date_str in missing_dates:
                print(f"Processing date: {date_str}")
                df = self.scrape_date(date_str)
                
                if not df.is_empty():
                    self.save_to_db(df, date_str)
                
                # Random delay between requests (2-5 seconds)
                delay = random.uniform(2, 4)
                print(f"Waiting {delay:.2f} seconds before next request...")
                time.sleep(delay)
                
                # Add longer delay every 10 dates
                if missing_dates.index(date_str) % 10 == 9:
                    long_delay = random.uniform(15, 30)
                    print(f"Taking a longer break of {long_delay:.2f} seconds...")
                    time.sleep(long_delay)
                
        finally:
            self.db.disconnect()

def main():
    scraper = JSEScraper()
    
    # Use a wider date range
    start_date = '2025-01-01'  # Or any date you want to check from
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Checking for missing data from {start_date} to {end_date}")
    scraper.scrape_date_range(start_date, end_date)
    print("Scraping completed")

if __name__ == "__main__":
    main()