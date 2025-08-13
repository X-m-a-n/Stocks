# Imports
import newspaper
from pathlib import Path
import polars as pl
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import os
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime

# Configure Logging
log_file = "C:/Users/Joshh/Projects/Stocks/Logs/HTML_extraction_logs_multi.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def get_files(folder_locations):
    '''Find all HTML files in the given folders'''
    all_files = []
    
    for folder in folder_locations:
        folder = Path(folder)
        folder_files = list(folder.glob('**/*.html'))
        # Filter out folders
        files_only = [f for f in folder_files if f.is_file()]
        all_files.extend(files_only)
    
    return all_files


def standardize_datetime(dt_value):
    """Convert datetime to string representation to avoid type issues"""
    if dt_value is None:
        return None
    try:
        # Try to convert to ISO format string if it's a datetime
        if isinstance(dt_value, (datetime, np.datetime64)):
            return dt_value.isoformat()
        # If it's already a string, return as is
        return str(dt_value)
    except:
        # If any error, return None
        return None


def extract_data(file_path) -> dict:
    """Process a single file"""
    try:
        # Read the file with buffering to improve I/O performance
        with open(file_path, 'r', encoding='utf-8', errors='ignore', buffering=1024*1024) as f:
            html_content = f.read()
        
        # Create an Article object
        news_article = newspaper.Article(url="")
        news_article.set_html(html_content)
        news_article.download_state = 2
        news_article.parse()
        
        # NLP is one of the most CPU intensive operations
        news_article.nlp()
        
        # Convert datetime to string format to prevent type issues
        publish_date_str = standardize_datetime(news_article.publish_date)
        
        # Convert set objects to lists for serialization
        article_dict = {
            'filename': str(file_path),
            'title': news_article.title or "",
            'text': news_article.text or "",
            'authors': list(news_article.authors) if isinstance(news_article.authors, set) else news_article.authors or [],
            'publish_date': publish_date_str,  # Use standardized string format
            'tags': list(news_article.tags) if isinstance(news_article.tags, set) else news_article.tags or [],
            'keywords': list(news_article.keywords) if isinstance(news_article.keywords, set) else news_article.keywords or [],
            'summary': news_article.summary or ""
        }
        return article_dict
    except Exception as e:
        # Log error but continue processing
        logger.debug(f"Error processing {file_path}: {e}")
        return {
            'filename': str(file_path),
            'title': "",
            'text': "",
            'authors': [],
            'publish_date': None,
            'tags': [],
            'keywords': [],
            'summary': f"Error: {str(e)}"
        }


def process_batch(file_batch):
    """Process a batch of files"""
    results = []
    for file in file_batch:
        try:
            results.append(extract_data(file))
        except Exception as e:
            # Catch any unexpected exceptions to prevent worker crashes
            logger.debug(f"Unexpected error processing {file}: {e}")
    return results


def save_intermediate_results(results_batch, batch_idx, out_dir):
    """Save intermediate results to avoid memory issues"""
    try:
        os.makedirs(out_dir, exist_ok=True)
        batch_df = pl.DataFrame(results_batch)
        batch_path = os.path.join(out_dir, f"batch_{batch_idx:05d}.parquet")
        batch_df.write_parquet(batch_path)
        return batch_path
    except Exception as e:
        logger.error(f"Error saving batch {batch_idx}: {e}")
        # Fallback to saving as JSON if Parquet fails
        try:
            import json
            batch_path = os.path.join(out_dir, f"batch_{batch_idx:05d}.json")
            with open(batch_path, 'w') as f:
                json.dump(results_batch, f)
            return batch_path
        except Exception as e2:
            logger.error(f"Error saving batch as JSON: {e2}")
            return None


def main():
    start_time = time.time()
    
    folders = [r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\ICInsider',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaGleaner',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JIS',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaObserver',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\nationwide_business_articles\raw_data\html']

    logger.info('Loading files')
    files = get_files(folders)
    total_files = len(files)
    logger.info(f'Found {total_files} HTML files to process')
    
    # Use smaller batches for better load balancing
    cpu_count = mp.cpu_count()
    batch_multiplier = 8
    ideal_batch_count = cpu_count * batch_multiplier
    
    batch_size = max(1, min(20, total_files // ideal_batch_count))
    file_batches = [files[i:i + batch_size] for i in range(0, total_files, batch_size)]
    batch_count = len(file_batches)
    
    logger.info(f"Processing with {cpu_count} CPUs, {batch_count} batches, {batch_size} files per batch")
    
    # Create temporary directory for intermediate results
    temp_dir = r"C:\Users\Joshh\Projects\Stocks\Data\temp_batches"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process in parallel
    batch_paths = []
    
    max_workers = min(batch_count, int(cpu_count * 1.5))
    
    # Use a two-phase approach: process batches and save intermediate results
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(file_batches)}
        
        # Process results as they complete to reduce memory usage
        with tqdm(total=batch_count, desc="Processing batches") as pbar:
            for future in futures:
                batch_idx = futures[future]
                try:
                    batch_results = future.result()
                    # Save each batch to disk to avoid memory issues
                    batch_path = save_intermediate_results(batch_results, batch_idx, temp_dir)
                    if batch_path:
                        batch_paths.append(batch_path)
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                finally:
                    pbar.update(1)
    
    # Now combine all the intermediate files
    logger.info(f"Combining {len(batch_paths)} result batches")
    try:
        # Read and combine all batch files
        dfs = []
        for path in tqdm(batch_paths, desc="Reading batch files"):
            try:
                df = pl.read_parquet(path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading batch file {path}: {e}")
        
        # Use polars' vstack to efficiently combine dataframes
        if dfs:
            final_df = pl.concat(dfs, how="diagonal_relaxed")
            
            out_path = r'C:\Users\Joshh\Projects\Stocks\Data\newspaper_extract_v2.parquet'
            logger.info(f'Writing {len(final_df)} records to {out_path}')
            final_df.write_parquet(out_path)
            
            # Cleanup temp files
            logger.info("Cleaning up temporary files")
            for path in batch_paths:
                try:
                    os.remove(path)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
            elapsed_time = time.time() - start_time
            files_per_second = total_files / elapsed_time
            logger.info(f'Process Complete: {total_files} files in {elapsed_time:.2f} seconds ({files_per_second:.2f} files/sec)')
        else:
            logger.error("No valid batch files to combine")
    except Exception as e:
        logger.error(f"Error in final combination: {e}")
        # Emergency fallback - save the paths so data isn't lost
        with open(r'C:\Users\Joshh\Projects\Stocks\Data\batch_paths.txt', 'w') as f:
            for path in batch_paths:
                f.write(f"{path}\n")
        logger.info("Saved batch paths to batch_paths.txt for manual recovery")


if __name__ == "__main__":
    # Set start method for Windows multiprocessing
    mp.set_start_method('spawn', force=True)
    main()