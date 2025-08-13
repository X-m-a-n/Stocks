import polars as pl
import pandas as pd
import re
from datetime import date
from typing import Optional, List
from fuzzywuzzy import fuzz, process
# Custom Imports
from data_loader import DataLoader
from jse_companies import jse_company_names
from technical_indicators import add_all_indicators, filter_buy_signals, get_signals_summary
# Warnings
import warnings
warnings.filterwarnings('ignore')


class MarketDataPipeline:
    """Main class for handling market data operations."""
    
    def __init__(self, data_loader=None, jse_companies=None):
        """
        Initialize the pipeline.
        
        Args:
            data_loader: Database connection/loader instance
            jse_companies: Dictionary mapping symbols to company name variations
        """
        self.loader = data_loader
        self.jse_company_names = jse_companies or {}
    
    @staticmethod
    def clean_text(df: pl.DataFrame, text_column: str) -> pl.DataFrame:
        """
        Clean text data by normalizing case, removing URLs and special characters.
        
        Args:
            df: Polars DataFrame containing text data
            text_column: Name of column to clean
            
        Returns:
            DataFrame with additional 'cleaned_text' column
        """
        df = df.with_columns([
            pl.col(text_column)
                .str.to_lowercase()
                .str.replace(r"http\S+", "", literal=False)
                .str.replace(r"[^a-zA-Z\s]", " ", literal=False)
                .str.replace(r"\s+", " ", literal=False)
                .str.strip_chars()
                .alias('cleaned_text')
        ])
        return df
    
    def find_company_symbol(self, company_name: str, threshold: int = 80) -> Optional[str]:
        """
        Find stock symbol for company name using fuzzy matching.
        Handles preprocessing edge cases and empty strings.
        
        Args:
            company_name: Company name to match
            threshold: Minimum fuzzy match score (0-100)
            
        Returns:
            Stock symbol if found, None otherwise
        """
        if not company_name or pd.isna(company_name) or str(company_name).strip() == '':
            return None
        
        company_name_clean = str(company_name).lower().strip()
        
        # Try exact matching first (fastest)
        for symbol, name_variations in self.jse_company_names.items():
            for variation in name_variations:
                if variation.lower() == company_name_clean:
                    return symbol
        
        def clean_name(text: str, aggressive: bool = False) -> str:
            """Clean company name with optional aggressive cleaning."""
            if aggressive:
                # Remove common business suffixes
                text = re.sub(r'\b(ltd|limited|inc|incorporated|corp|corporation|co|company|plc|llc)\b', '', text.lower())
                text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
            else:
                # Light cleaning - just normalize
                text = text.lower().strip()
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            return text.strip()
        
        # Try light cleaning first
        company_light_clean = clean_name(company_name_clean, aggressive=False)
        
        # Create variation mappings with light cleaning
        variation_to_symbol = {}
        for symbol, variations in self.jse_company_names.items():
            for variation in variations:
                cleaned_var = clean_name(variation, aggressive=False)
                if cleaned_var:
                    variation_to_symbol[cleaned_var] = symbol
        
        # Try fuzzy match with light cleaning
        if company_light_clean:
            result = process.extractOne(
                company_light_clean,
                variation_to_symbol.keys(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold
            )
            
            if result:
                matched_variation, score = result
                return variation_to_symbol[matched_variation]
        
        # Fallback to aggressive cleaning
        company_aggressive_clean = clean_name(company_name_clean, aggressive=True)
        
        if company_aggressive_clean and len(company_aggressive_clean) > 1:
            # Create aggressively cleaned variation mapping
            aggressive_variation_to_symbol = {}
            for symbol, variations in self.jse_company_names.items():
                for variation in variations:
                    cleaned_var = clean_name(variation, aggressive=True)
                    if cleaned_var and len(cleaned_var) > 1:
                        aggressive_variation_to_symbol[cleaned_var] = symbol
            
            result = process.extractOne(
                company_aggressive_clean,
                aggressive_variation_to_symbol.keys(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=max(threshold - 10, 60)
            )
            
            if result:
                matched_variation, score = result
                return aggressive_variation_to_symbol[matched_variation]
        
        return None
    
    def get_company_variations(self, symbol: str) -> List[str]:
        """
        Get all name variations for a stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of name variations, empty list if symbol not found
        """
        return self.jse_company_names.get(symbol.upper(), [])
    
    def load_stock_data(self, start_date: str = '2017-01-01', end_date: str = '2025-03-31') -> pl.DataFrame:
        """
        Load stock market data from database.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with stock data
        """
        query = """
        SELECT DISTINCT
            trade_date as date,
            clean_symbol as symbol,
            daily_range_high,
            daily_range_low,
            closing_price,
            closing_price + price_change as open_price,
            volume
        FROM
            old_trade_data TD
            LEFT JOIN jse_database.instruments I on (TD.CLEAN_symbol = I.symbol AND TD.CURRENCY = I.CURRENCY)
        WHERE
            trade_date BETWEEN :start_date AND :end_date
            AND I.TYPE = 'ORDINARY'
            AND TD.CURRENCY = 'JMD'
        ORDER BY
            trade_date ASC,
            clean_symbol ASC
        """
        
        return self.loader.fetch_data(
            query=query,
            start_date=start_date,
            end_date=end_date
        )
    
    def load_sentiment_data(self, file_path: str) -> pl.DataFrame:
        """
        Load and process sentiment data from parquet file.
        
        Args:
            file_path: Path to sentiment data parquet file
            
        Returns:
            Processed sentiment DataFrame
        """
        cols = [
            'filename', 'title', 'authors', 'cleaned_text', 'named_entities', 
            'sentiment_polarity', 'all_dates_found', 'raw_dates_found',
            'sentiment_subjectivity', 'organizations', 'publication_date', 
            'title_polarity', 'title_subjectivity', 'title_sentiment_label', 
            'title_organizations', 'title_fin_polarity', 'title_fin_subjectivity', 
            'title_fin_sentiment_label', 'title_fin_sentiment', 'text_fin_polarity', 
            'text_fin_subjectivity', 'text_fin_sentiment_label', 'text_fin_sentiment'
        ]
        
        raw_sentiment_df = pl.read_parquet(source=file_path, columns=cols)
        
        print(f"Total rows with null dates before imputation: {raw_sentiment_df.filter(pl.col('publication_date').is_null()).height}")
        
        # Fill null publication dates
        raw_sentiment_df = raw_sentiment_df.with_columns([
            pl.when(pl.col("publication_date").is_null())
            .then(
                pl.coalesce([
                    # Try filename date first (YYYY_MM_DD)
                    pl.col("filename").str.extract(r"(\d{4}_\d{2}_\d{2})")
                    .str.replace_all("_", "-").str.to_date("%Y-%m-%d"),
                    # Fallback to directory date (YYYY_MM, set to 1st of month)
                    pl.col("filename").str.extract(r"(\d{4}_\d{2})_\w+")
                    .str.replace("_", "-").str.strptime(pl.Date, "%Y-%m")
                ])
            )
            .otherwise(pl.col("publication_date"))
            .alias("publication_date")
        ])
        
        print(f"Total rows with null dates after imputation: {raw_sentiment_df.filter(pl.col('publication_date').is_null()).height}")
        
        # Combine organizations columns
        raw_sentiment_df = raw_sentiment_df.with_columns(
            pl.col("organizations").list.concat(pl.col("title_organizations"))
        )
        
        # Filter valid date range
        initial_rows = raw_sentiment_df.height
        raw_sentiment_df = raw_sentiment_df.filter(
            (pl.col("publication_date") >= date(2017, 1, 1)) & 
            (pl.col("publication_date") <= date(2025, 3, 31))
        )
        
        print(f'{initial_rows - raw_sentiment_df.height} rows with invalid dates dropped. {raw_sentiment_df.height} rows remain')
        
        # Clean up columns
        raw_sentiment_df = raw_sentiment_df.drop([
            'title_organizations', 'filename', 'authors', 'cleaned_text', 
            'all_dates_found', 'raw_dates_found'
        ])
        
        return raw_sentiment_df
    
    def process_sentiment_data(self, raw_sentiment_df: pl.DataFrame) -> pl.DataFrame:
        """
        Process sentiment data and map to company symbols.
        
        Args:
            raw_sentiment_df: Raw sentiment DataFrame
            
        Returns:
            Processed sentiment data with company symbol mapping
        """
        sentiment_df = (
            raw_sentiment_df
            .select([
                'publication_date',
                'organizations',
                'title',
                'sentiment_polarity', 
                'sentiment_subjectivity',
                'title_polarity', 
                'title_subjectivity', 
                'title_fin_polarity', 
                'title_fin_subjectivity', 
                'text_fin_polarity', 
                'text_fin_subjectivity', 
            ])
            .explode('organizations')  # Create separate row for each organization
            .with_columns([
                pl.col('organizations').str.strip_chars().str.to_lowercase().alias('institution'),
                pl.col('publication_date').alias('date')
            ])
            .filter(
                pl.col('institution').is_not_null() & 
                (pl.col('institution') != "")
            )
            .with_columns([
                pl.col('institution').map_elements(
                    self.find_company_symbol, 
                    return_dtype=pl.Utf8
                ).alias('symbol')
            ])
            .filter(pl.col('symbol').is_not_null())
        )
        
        print(f"Mapped {sentiment_df.height:,} news articles to JSE companies")
        return sentiment_df
    
    def integrate_market_sentiment(self, stocks_df: pl.DataFrame, sentiment_df: pl.DataFrame) -> pl.DataFrame:
        """
        Integrate market data with sentiment analysis.
        
        Args:
            stocks_df: Stock market data
            sentiment_df: Sentiment data
            
        Returns:
            Integrated DataFrame with market and sentiment data
        """
        print("Step 1: Aggregating daily sentiment by company...")
        
        # Determine date column name
        date_col = 'date' if 'date' in sentiment_df.columns else 'publication_date'
        
        # Aggregate daily sentiment by company
        daily_sentiment = (
            sentiment_df
            .group_by([date_col, 'symbol'])
            .agg([
                pl.col('sentiment_polarity').mean().alias('sentiment'),
                pl.col('sentiment_subjectivity').mean().alias('subjectivity'),
                pl.col('text_fin_polarity').mean().alias('fin_sentiment'),
                pl.col('text_fin_subjectivity').mean().alias('fin_subjectivity'),
                pl.col('title_polarity').mean().alias('title_sentiment'),
                pl.col('title_subjectivity').mean().alias('title_subjectivity'),
                pl.col('title_fin_polarity').mean().alias('title_fin_sentiment'),
                pl.col('title_fin_subjectivity').mean().alias('title_fin_subjectivity'),
                pl.len().alias('news_count'),
                pl.col('title').first().alias('sample_headline')
            ])
            .rename({date_col: 'date'})
        )
        
        print(f"Created {daily_sentiment.height} daily sentiment records")
        
        print("Step 2: Joining with market data...")
        
        # Join and create integrated dataset
        integrated = (
            stocks_df
            .join(daily_sentiment, on=['date', 'symbol'], how='left')
            .with_columns([
                # Fill missing sentiment values
                pl.col('sentiment').fill_null(0.0),
                pl.col('subjectivity').fill_null(0.0),
                pl.col('fin_sentiment').fill_null(0.0), 
                pl.col('fin_subjectivity').fill_null(0.0), 
                pl.col('title_sentiment').fill_null(0.0), 
                pl.col('title_subjectivity').fill_null(0.0),
                pl.col('title_fin_sentiment').fill_null(0.0),
                pl.col('title_fin_subjectivity').fill_null(0.0),
                pl.col('news_count').fill_null(0.0),
                
                # Create derived features
                pl.when(pl.col('news_count') > 0)
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias('has_news'),
                
                pl.when(pl.col('sentiment') > 0.1)
                  .then(pl.lit('Positive'))
                  .when(pl.col('sentiment') < -0.1)
                  .then(pl.lit('Negative'))
                  .otherwise(pl.lit('Neutral'))
                  .alias('sentiment_category')
            ])
            .sort(['date', 'symbol'])
        )
        
        return integrated


def create_pipeline(data_loader=None, jse_companies=None, sentiment_file_path=None):
    """
    Factory function to create and run the complete data pipeline.
    
    Args:
        data_loader: Database connection instance
        jse_companies: Company name mapping dictionary
        sentiment_file_path: Path to sentiment data file
        
    Returns:
        Integrated market and sentiment data
    """
    pipeline = MarketDataPipeline(data_loader, jse_companies)
    
    # Load stock data
    stocks_df = pipeline.load_stock_data()
    print(f"Loaded {stocks_df.height:,} stock records")
    
    # Load and process sentiment data
    if sentiment_file_path:
        raw_sentiment = pipeline.load_sentiment_data(sentiment_file_path)
        sentiment_df = pipeline.process_sentiment_data(raw_sentiment)
        
        # Integrate data
        integrated_data = pipeline.integrate_market_sentiment(stocks_df, sentiment_df)
        print(f"Created integrated dataset with {integrated_data.height:,} records")
        
        return integrated_data
    
    return stocks_df