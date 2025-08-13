import requests
import polars as pl
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
from typing import List, Set
import time
import random

class JSEScraper:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Jamaican holidays (update for each year)
        self.holidays = {
            # 2020 holidays
            "2020-01-01", "2020-02-26", "2020-04-10", "2020-04-13", 
            "2020-05-25", "2020-08-01", "2020-08-06", "2020-10-19",
            "2020-12-25", "2020-12-28",
            # 2021 holidays
            "2021-01-01", "2021-02-17", "2021-04-02", "2021-04-05",
            "2021-05-24", "2021-08-02", "2021-08-06", "2021-10-18",
            "2021-12-27", "2021-12-28",
            # 2022 holidays
            "2022-01-03", "2022-03-02", "2022-04-15", "2022-04-18",
            "2022-05-23", "2022-08-01", "2022-08-06", "2022-10-17",
            "2022-12-26", "2022-12-27",
            # 2023 holidays
            "2023-01-02", "2023-02-22", "2023-04-07", "2023-04-10",
            "2023-05-22", "2023-08-01", "2023-08-07", "2023-10-16",
            "2023-12-25", "2023-12-26",
            # 2024 holidays (add as needed)
            "2024-01-01", "2024-02-14", "2024-03-29", "2024-04-01",
            "2024-05-27", "2024-08-01", "2024-08-06", "2024-10-21",
            "2024-12-25", "2024-12-26",
            # 2025 hoidays
            '2025-01-01', '2025-03-05'
        }

    def is_trading_day(self, date: datetime) -> bool:
        """Check if given date is a trading day"""
        # Check if weekend
        if date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            return False
            
        # Check if holiday
        date_str = date.strftime('%Y-%m-%d')
        return date_str not in self.holidays

    def scrape_date(self, date: str) -> pl.DataFrame:
        """Scrape data for a specific date"""
        url = f"https://www.jamstockex.com/trading/trade-quotes/?market=50&date={date}"
        
        try:
            print(f"Requesting data for: {date}")
            response = requests.get(url, headers=self.headers)
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
                                data.append({
                                    'date': date,
                                    'symbol': cells[1].get_text(strip=True),
                                    'last_price': self.parse_float(cells[2].get_text(strip=True)),
                                    'closing_price': self.parse_float(cells[3].get_text(strip=True)),
                                    'price_change': self.parse_float(cells[4].get_text(strip=True)),
                                    'bid': self.parse_float(cells[5].get_text(strip=True)),
                                    'ask': self.parse_float(cells[6].get_text(strip=True)),
                                    'volume': self.parse_int(cells[7].get_text(strip=True)),
                                    'range': cells[8].get_text(strip=True),
                                    'year_range': cells[9].get_text(strip=True)
                                })
                            except Exception as e:
                                print(f"Error parsing row: {str(e)}")
                                continue
            
            return pl.DataFrame(data)
            
        except Exception as e:
            print(f"Error scraping {date}: {str(e)}")
            return pl.DataFrame()

    def scrape_date_range(self, start_date: str, end_date: str):
        """
        Scrape data for a date range, excluding weekends and holidays
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current = start
        
        while current <= end:
            if self.is_trading_day(current):
                date_str = current.strftime('%Y-%m-%d')
                
                # Check if file already exists
                filename = os.path.join(self.output_dir, f"jse_stocks_{date_str}.csv")
                if os.path.exists(filename):
                    print(f"File already exists for {date_str}, skipping...")
                    current += timedelta(days=1)
                    continue
                
                df = self.scrape_date(date_str)
                
                if not df.is_empty():
                    os.makedirs(self.output_dir, exist_ok=True)
                    df.write_csv(filename)
                    print(f"Saved data for {date_str}")
                
                # Random delay between requests (1-3 seconds)
                time.sleep(random.uniform(1, 3))
            
            current += timedelta(days=1)

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

def main():
    output_dir = r"C:\Users\michaelsjo\Desktop\Stocks\Data\eod_trade_summary"
    scraper = JSEScraper(output_dir)
    
    # Scrape from 2020 to present
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Starting scrape from {start_date} to {end_date}")
    scraper.scrape_date_range(start_date, end_date)
    print("Scraping completed")

if __name__ == "__main__":
    main()