# Imports
import newspaper
from urllib.parse import urlparse
import os
from pathlib import Path
import polars as pl
import pandas as pd
from functools import partial
from tqdm import tqdm
import logging

# Configure Logging
log_file = "C:/Users/Joshh/Projects/Stocks/Logs/HTML_extraction_logs.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure all messages are processed by handlers

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


# Define a function to import the raw html files
def get_files(folderLoc):
    ''''This function accepts a list of folders and returns a list of html file paths'''

    allFiles = []

    folders = folderLoc

    for folder in folders:
        folder = Path(folder)
        folderFiles = list(folder.glob('**/*.html'))
        allFiles.append(folderFiles)
        # Filter out folders
        filesOnly = [f for f in folderFiles if f.is_file()]
        allFiles.extend(filesOnly)

    return allFiles

def extract_data(file) -> dict:
    """Process a single file"""
    try:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
         # Create an Article object
        newsArticle = newspaper.Article(url="")
        newsArticle.set_html(html_content)
        newsArticle.download_state = 2
        newsArticle.parse()
        newsArticle.nlp()
        
        article_dict = {
            'filename': str(file),
            'title': newsArticle.title or "",
            'text': newsArticle.text or "",
            'authors': list(newsArticle.authors) if isinstance(newsArticle.authors, set) else newsArticle.authors or [],
            'publish_date': newsArticle.publish_date,
            'tags': list(newsArticle.tags) if isinstance(newsArticle.tags, set) else newsArticle.tags or [],
            'keywords': list(newsArticle.keywords) if isinstance(newsArticle.keywords, set) else newsArticle.keywords or [],
            'summary': newsArticle.summary or ""
        }
        return article_dict
    except Exception as e:
        logger.info('File Error')
        logger.debug(f"Error processing {file}: {e}")
        return {
            'filename': str(file),
            'title': "",
            'text': "",
            'authors': [],
            'publish_date': None,
            'tags': [],
            'keywords': [],
            'summary': f"Error: {str(e)}"
        }


def process_files(files: list) -> list:
    #return list(tqdm(map(extract_data, files), desc="Processing files"))
    #return list(tqdm(map(extract_data, files[:100]), desc="Processing files", total_segments = 100, ))
    return list(tqdm(
        map(extract_data, files),
        desc="Processing files",
        total=100,
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        colour="green"
    ))
    

def main():

    folders = [r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\ICInsider',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaGleaner',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JIS',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaObserver',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\nationwide_business_articles\raw_data\html']

    logger.info('Loading files')
    files = get_files(folders)
    
    df = pl.DataFrame(process_files(files))
    outPath = r'C:\Users\Joshh\Projects\Stocks\Data\newspaper_extract.parquet'

    logger.info(f'Writing {len(df)} files to disc')
    df.write_parquet(outPath)

    logger.info('Process Complete')

if __name__ == "__main__":
    main()