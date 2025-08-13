import polars as pl
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DateExtractor:
    """Extract dates from HTML newspaper articles."""
    
    def __init__(self):
        # Common date patterns for newspaper articles
        self.date_patterns = [
            # ISO format: 2024-12-31, 2024/12/31
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            # US format: 12/31/2024, 12-31-2024
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            # Long format: January 1, 2024 | Jan 1, 2024
            r'\b([A-Za-z]+\s+\d{1,2},?\s+\d{4})\b',
            # European format: 31/12/2024, 31.12.2024
            r'\b(\d{1,2}[./]\d{1,2}[./]\d{4})\b',
            # Month day format: Jan 1 | January 1
            r'\b([A-Za-z]+\s+\d{1,2})\b(?=\s)',
            # Published/Updated timestamps
            r'(?:published|updated|posted)(?:\s*:?\s*)([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            # Datetime attributes in HTML
            r'datetime=["\']([^"\']+)["\']',
            # Meta tags with dates
            r'<meta[^>]+(?:date|time)[^>]+content=["\']([^"\']+)["\']',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.date_patterns]
    
    def extract_text_from_html(self, html_content: str) -> str:
        """Extract text content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.warning(f"Error parsing HTML: {e}")
            return html_content
    
    def extract_dates_from_text(self, text: str) -> List[str]:
        """Extract all potential dates from text."""
        dates = []
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            dates.extend(matches)
        
        return list(set(dates))  # Remove duplicates
    
    def normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date string to ISO format."""
        if not date_str:
            return None
            
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%d/%m/%Y',
            '%d.%m.%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%B %d %Y',
            '%b %d %Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format matches, try to extract year at least
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return f"{year_match.group(0)}-01-01"  # Default to Jan 1 if only year
        
        return None

def process_html_file(file_path: Path, extractor: DateExtractor) -> Dict:
    """Process a single HTML file and extract date information."""
    try:
        # Read HTML file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Extract text from HTML
        text_content = extractor.extract_text_from_html(html_content)
        
        # Extract dates from text
        raw_dates = extractor.extract_dates_from_text(text_content)
        
        # Normalize dates
        normalized_dates = []
        for date_str in raw_dates:
            normalized = extractor.normalize_date(date_str)
            if normalized:
                normalized_dates.append(normalized)
        
        # Get the most likely publication date (earliest valid date)
        publication_date = min(normalized_dates) if normalized_dates else None
        
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'folder': file_path.parent.name,
            'publication_date': publication_date,
            'all_dates_found': normalized_dates,
            'raw_dates_found': raw_dates,
            'total_dates_found': len(raw_dates),
            'processed_at': datetime.now().isoformat(),
            'file_size_bytes': file_path.stat().st_size,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'folder': file_path.parent.name,
            'publication_date': None,
            'all_dates_found': [],
            'raw_dates_found': [],
            'total_dates_found': 0,
            'processed_at': datetime.now().isoformat(),
            'file_size_bytes': 0,
            'success': False,
            'error': str(e)
        }

def get_html_files_from_folders(folder_paths: List[str]) -> List[Path]:
    """Get all HTML files from the specified folders."""
    html_files = []
    
    for folder_path in folder_paths:
        folder = Path(folder_path)
        if not folder.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            continue
        
        if not folder.is_dir():
            logger.warning(f"Path is not a directory: {folder_path}")
            continue
        
        # Find all HTML files recursively
        pattern_files = list(folder.rglob('*.html')) + list(folder.rglob('*.htm'))
        html_files.extend(pattern_files)
        logger.info(f"Found {len(pattern_files)} HTML files in {folder_path}")
    
    return html_files

def process_newspaper_articles(folder_paths: List[str], 
                             output_file: str = "newspaper_dates.parquet",
                             num_processes: Optional[int] = None,
                             chunk_size: int = 100) -> pl.DataFrame:
    """
    Main function to process newspaper articles and extract dates.
    
    Args:
        folder_paths: List of folder paths containing HTML files
        output_file: Output file path for the results
        num_processes: Number of processes to use (default: CPU count)
        chunk_size: Number of files to process per chunk (affects memory usage)
    
    Returns:
        Polars DataFrame with extracted date information
    """
    # Optimize process count for maximum CPU usage
    if num_processes is None:
        num_processes = cpu_count()
    
    # On Windows, be very conservative with process count due to handle limits
    # The multiprocessing pool creates additional handles beyond the worker processes
    if os.name == 'nt':  # Windows
        num_processes = min(num_processes, 16)  # Very conservative limit for Windows
    else:
        num_processes = min(num_processes * 2, 64)  # Higher limit for Unix systems
    
    logger.info(f"Starting processing with {num_processes} processes")
    
    # Get all HTML files
    html_files = get_html_files_from_folders(folder_paths)
    
    if not html_files:
        logger.warning("No HTML files found in the specified folders")
        return pl.DataFrame()
    
    logger.info(f"Processing {len(html_files)} HTML files")
    
    # Create date extractor
    extractor = DateExtractor()
    
    # Process files using multiprocessing with progress bar
    results = []
    try:
        with Pool(processes=num_processes) as pool:
            process_func = partial(process_html_file, extractor=extractor)
            
            # Use imap for better progress tracking and memory efficiency
            with tqdm(total=len(html_files), desc="Processing files", unit="files") as pbar:
                # Process in chunks to show progress
                for result in pool.imap(process_func, html_files, chunksize=chunk_size):
                    results.append(result)
                    pbar.update(1)
                    
                    # Update description with current stats every 100 files
                    if len(results) % 100 == 0:
                        successful = sum(1 for r in results if r['success'])
                        with_dates = sum(1 for r in results if r['publication_date'] is not None)
                        pbar.set_postfix({
                            'Success': f"{successful}/{len(results)}",
                            'With_Dates': f"{with_dates}/{len(results)}"
                        })
            
            # Explicitly close and join the pool
            pool.close()
            pool.join()
            
    except Exception as e:
        logger.error(f"Error during multiprocessing: {e}")
        # If multiprocessing fails, fall back to sequential processing
        logger.info("Falling back to sequential processing...")
        results = []
        for i, file_path in enumerate(tqdm(html_files, desc="Sequential processing")):
            results.append(process_html_file(file_path, extractor))
    
    logger.info("Multiprocessing complete, creating DataFrame...")
    
    # Convert results to Polars DataFrame
    df = pl.DataFrame(results)
    
    # Convert date columns to proper date types
    if not df.is_empty():
        df = df.with_columns([
            pl.col("publication_date").str.to_date("%Y-%m-%d", strict=False),
            pl.col("processed_at").str.to_datetime("%Y-%m-%dT%H:%M:%S.%f", strict=False)
        ])
    
    # Save to file
    if output_file:
        logger.info(f"Saving {len(df)} records to {output_file}...")
        try:
            if output_file.endswith('.parquet'):
                df.write_parquet(output_file)
            elif output_file.endswith('.csv'):
                df.write_csv(output_file)
            else:
                df.write_parquet(f"{output_file}.parquet")
            
            logger.info(f"✓ Results successfully saved to {output_file}")
            
            # Verify file exists and show size
            output_path = Path(output_file)
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"✓ File size: {file_size:.2f} MB")
            else:
                logger.error(f"✗ File was not created: {output_file}")
                
        except Exception as e:
            logger.error(f"✗ Error saving file: {e}")
            # Try saving as CSV as fallback
            try:
                fallback_file = output_file.replace('.parquet', '_fallback.csv')
                df.write_csv(fallback_file)
                logger.info(f"✓ Fallback save successful: {fallback_file}")
            except Exception as e2:
                logger.error(f"✗ Fallback save also failed: {e2}")
    
    # Log summary statistics
    if not df.is_empty():
        successful_files = df.filter(pl.col("success") == True).height
        failed_files = df.filter(pl.col("success") == False).height
        files_with_dates = df.filter(pl.col("publication_date").is_not_null()).height
        
        logger.info(f"✓ Processing complete!")
        logger.info(f"  - Total files: {len(html_files):,}")
        logger.info(f"  - Successfully processed: {successful_files:,}")
        logger.info(f"  - Failed: {failed_files:,}")
        logger.info(f"  - Files with publication dates: {files_with_dates:,}")
        if len(html_files) > 0:
            logger.info(f"  - Success rate: {successful_files/len(html_files)*100:.1f}%")
        if successful_files > 0:
            logger.info(f"  - Date extraction rate: {files_with_dates/successful_files*100:.1f}%")
    
    logger.info("✓ Script execution complete!")
    return df

# Main execution
if __name__ == '__main__':
    try:
        # Define folder paths
        folders = [
            r"C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\ICInsider",
            r"C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaGleaner",
            r"C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JIS",
            r"C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaObserver\business"
        ]
        
        output_file = r'C:\Users\Joshh\Projects\Stocks\newspaper_dates.parquet'
        
        print(f"Starting processing...")
        print(f"Output file will be: {os.path.abspath(output_file)}")
        
        # Process the files
        df = process_newspaper_articles(
            folder_paths=folders,
            output_file=output_file,
            num_processes=12,  # Safe conservative limit for Windows
            chunk_size=50     # Larger chunks for better efficiency
        )
        
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'='*50}")
        print(f"Processed {len(df):,} files")
        
        # Check if file exists
        output_path = Path(output_file)
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ Output file created: {output_path.absolute()}")
            print(f"✓ File size: {file_size:.2f} MB")
        else:
            print(f"✗ Output file not found: {output_path.absolute()}")
        
        if not df.is_empty():
            print(f"\nSample results:")
            print(df.select([
                "file_name", "folder", "publication_date", 
                "total_dates_found", "success"
            ]).head(5))
            
            # Show detailed statistics
            total_files = len(df)
            successful = df.filter(pl.col("success") == True).height
            with_dates = df.filter(pl.col("publication_date").is_not_null()).height
            
            print(f"\nFinal Statistics:")
            print(f"Total files processed: {total_files:,}")
            print(f"Successfully processed: {successful:,} ({successful/total_files*100:.1f}%)")
            print(f"Files with dates found: {with_dates:,} ({with_dates/total_files*100:.1f}%)")
            
            if with_dates > 0:
                date_stats = df.filter(pl.col("publication_date").is_not_null()).select([
                    pl.col("publication_date").min().alias("earliest_date"),
                    pl.col("publication_date").max().alias("latest_date")
                ]).row(0)
                print(f"Date range: {date_stats[0]} to {date_stats[1]}")
        
        print(f"\n✓ Script completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n✗ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Script failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"✓ Cleanup complete, script exiting...")