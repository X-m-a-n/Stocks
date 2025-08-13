#!/usr/bin/env python3
"""
Complete Multiprocessing Sentiment Analysis Script
==================================================

A comprehensive text analysis pipeline using multiprocessing for:
- Named Entity Recognition (spaCy)
- Sentiment Analysis (TextBlob)
- Entity categorization and extraction

Date: 2025-06-21
"""

import polars as pl
import spacy
from textblob import TextBlob
from typing import List, Tuple, Optional
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import time
import sys
from pathlib import Path
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Complete sentiment analysis pipeline with multiprocessing support."""
    
    def __init__(self, model_name: str = "en_core_web_lg", use_gpu: bool = False):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: spaCy model name to use
            use_gpu: Whether to attempt GPU usage (main process only)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.nlp = None
        self._load_main_model()
    
    def _load_main_model(self):
        """Load spaCy model in main process."""
        try:
            if self.use_gpu:
                spacy.require_gpu()
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(f"Model {self.model_name} not found, trying en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.model_name = "en_core_web_sm"
                logger.info("Loaded fallback model: en_core_web_sm")
            except OSError:
                logger.error("No spaCy models found. Please install with: python -m spacy download en_core_web_sm")
                sys.exit(1)
    
    @staticmethod
    def init_worker(model_name: str):
        """Initialize spaCy model in each worker process."""
        global nlp
        try:
            # Don't use GPU in worker processes to avoid conflicts
            nlp = spacy.load(model_name)
        except OSError:
            nlp = spacy.load("en_core_web_sm")
    
    @staticmethod
    def extract_entities_worker(text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text using spaCy in worker process."""
        if not text or not isinstance(text, str) or text.strip() == "":
            return []
        try:
            doc = nlp(text[:1000000])  # Limit text length to prevent memory issues
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception as e:
            logger.warning(f"NER failed for text: {str(e)[:100]}")
            return []
    
    @staticmethod
    def get_sentiment_worker(text: str) -> Tuple[float, float]:
        """Get sentiment polarity and subjectivity using TextBlob in worker process."""
        if not text or not isinstance(text, str) or text.strip() == "":
            return (0.0, 0.0)
        try:
            blob = TextBlob(text[:1000000])  # Limit text length
            return (blob.sentiment.polarity, blob.sentiment.subjectivity)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for text: {str(e)[:100]}")
            return (0.0, 0.0)
    
    @staticmethod
    def process_ner_chunk(texts_chunk: List[str]) -> List[List[Tuple[str, str]]]:
        """Process a chunk of texts for NER."""
        return [SentimentAnalyzer.extract_entities_worker(text) for text in texts_chunk]
    
    @staticmethod
    def process_sentiment_chunk(texts_chunk: List[str]) -> List[Tuple[float, float]]:
        """Process a chunk of texts for sentiment analysis."""
        return [SentimentAnalyzer.get_sentiment_worker(text) for text in texts_chunk]
    
    def clean_text(self, df: pl.DataFrame, text_column: str) -> pl.DataFrame:
        """Clean text data using polars expressions."""
        logger.info(f"Cleaning text in column: {text_column}")
        
        cleaned_df = df.with_columns([
            pl.col(text_column)
                .str.to_lowercase()
                .str.replace(r"http\S+", "", literal=False)
                .str.replace(r"[^a-zA-Z\s]", " ", literal=False)
                .str.replace(r"\s+", " ", literal=False)
                .str.strip_chars()
                .alias('cleaned_text')
        ])
        
        logger.info("Text cleaning completed")
        return cleaned_df
    
    def create_chunks(self, lst: List, chunk_size: int) -> List[List]:
        """Create chunks from a list."""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
    def process_multiprocessing(
        self, 
        df: pl.DataFrame, 
        text_column: str, 
        n_processes: Optional[int] = None,
        chunk_size: int = 100,
        clean_text: bool = True
    ) -> pl.DataFrame:
        """
        Complete text analysis pipeline with multiprocessing.
        
        Args:
            df: Polars DataFrame
            text_column: Name of the text column to analyze
            n_processes: Number of processes to use (default: CPU count - 1)
            chunk_size: Size of chunks to process in parallel
            clean_text: Whether to clean text before processing
        
        Returns:
            DataFrame with sentiment and NER results
        """
        
        if n_processes is None:
            n_processes = max(1, mp.cpu_count() - 1)
        
        logger.info(f"Starting multiprocessing analysis with {n_processes} processes")
        logger.info(f"Processing {len(df)} texts in chunks of {chunk_size}")
        
        # Clean text if requested
        if clean_text and 'cleaned_text' not in df.columns:
            df = self.clean_text(df, text_column)
            text_column = 'cleaned_text'
        
        # Convert to list for processing
        texts = df[text_column].to_list()
        
        # Filter out null/empty texts
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
        
        logger.info(f"Found {len(valid_texts)} valid texts out of {len(texts)}")
        
        # Create chunks for parallel processing
        text_chunks = self.create_chunks(valid_texts, chunk_size)
        
        # Initialize results lists
        all_entities = [[] for _ in range(len(texts))]
        all_sentiments = [(0.0, 0.0) for _ in range(len(texts))]
        
        # Step 1: Named Entity Recognition with multiprocessing
        logger.info(f"Step 1: Processing Named Entity Recognition...")
        
        with mp.Pool(
            processes=n_processes, 
            initializer=self.init_worker,
            initargs=(self.model_name,)
        ) as pool:
            ner_results = list(tqdm(
                pool.imap(self.process_ner_chunk, text_chunks),
                total=len(text_chunks),
                desc="NER Chunks"
            ))
        
        # Map results back to original indices
        flat_ner_results = []
        for chunk_result in ner_results:
            flat_ner_results.extend(chunk_result)
        
        for i, result in enumerate(flat_ner_results):
            if i < len(valid_indices):
                all_entities[valid_indices[i]] = result
        
        logger.info(f"✓ NER completed for {len(flat_ner_results)} texts")
        
        # Step 2: Sentiment Analysis with multiprocessing
        logger.info(f"Step 2: Processing Sentiment Analysis...")
        
        with mp.Pool(
            processes=n_processes,
            initializer=self.init_worker,
            initargs=(self.model_name,)
        ) as pool:
            sentiment_results = list(tqdm(
                pool.imap(self.process_sentiment_chunk, text_chunks),
                total=len(text_chunks),
                desc="Sentiment Chunks"
            ))
        
        # Map results back to original indices
        flat_sentiment_results = []
        for chunk_result in sentiment_results:
            flat_sentiment_results.extend(chunk_result)
        
        for i, result in enumerate(flat_sentiment_results):
            if i < len(valid_indices):
                all_sentiments[valid_indices[i]] = result
        
        logger.info(f"✓ Sentiment analysis completed for {len(flat_sentiment_results)} texts")
        
        # Step 3: Build result dataframe
        logger.info("Step 3: Building result dataframe...")
        
        result_df = df.with_columns([
            pl.Series("named_entities", all_entities),
            pl.Series("sentiment_polarity", [s[0] for s in all_sentiments]),
            pl.Series("sentiment_subjectivity", [s[1] for s in all_sentiments])
        ])
        
        # Step 4: Extract entity categories
        logger.info("Step 4: Categorizing entities...")
        
        # Extract organizations, people, and products
        organizations = []
        people = []
        products = []
        
        for entities in all_entities:
            orgs = [ent[0] for ent in entities if ent[1] == "ORG"]
            persons = [ent[0] for ent in entities if ent[1] == "PERSON"]
            prods = [ent[0] for ent in entities if ent[1] == "PRODUCT"]
            
            organizations.append(orgs)
            people.append(persons)
            products.append(prods)
        
        final_df = result_df.with_columns([
            pl.Series("organizations", organizations),
            pl.Series("people", people),
            pl.Series("products", products)
        ])
        
        logger.info("✓ Entity categorization complete")
        logger.info(f"🎉 Analysis finished! Processed {len(df)} texts successfully.")
        
        return final_df
    
    def analyze_file(
        self, 
        input_file: str, 
        output_file: str, 
        text_column: str,
        n_processes: Optional[int] = None,
        chunk_size: int = 100
    ):
        """
        Analyze a file and save results.
        
        Args:
            input_file: Path to input file (parquet, csv, json)
            output_file: Path to output file
            text_column: Name of text column to analyze
            n_processes: Number of processes
            chunk_size: Chunk size for processing
        """
        
        # Read input file
        input_path = Path(input_file)
        logger.info(f"Reading input file: {input_file}")
        
        if input_path.suffix.lower() == '.parquet':
            df = pl.read_parquet(input_file)
        elif input_path.suffix.lower() == '.csv':
            df = pl.read_csv(input_file)
        elif input_path.suffix.lower() == '.json':
            df = pl.read_json(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"Loaded {len(df)} rows from {input_file}")
        
        # Check if text column exists
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in input file")
            logger.info(f"Available columns: {df.columns}")
            sys.exit(1)
        
        # Process the data
        start_time = time.time()
        result_df = self.process_multiprocessing(
            df, text_column, n_processes, chunk_size
        )
        processing_time = time.time() - start_time
        
        # Save results
        output_path = Path(output_file)
        logger.info(f"Saving results to: {output_file}")
        
        if output_path.suffix.lower() == '.parquet':
            result_df.write_parquet(output_file)
        elif output_path.suffix.lower() == '.csv':
            result_df.write_csv(output_file)
        elif output_path.suffix.lower() == '.json':
            result_df.write_json(output_file)
        else:
            # Default to parquet
            result_df.write_parquet(output_file)
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info(f"ANALYSIS COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Texts processed: {len(df)}")
        logger.info(f"Speed: {len(df)/processing_time:.1f} texts/second")
        logger.info(f"Output saved to: {output_file}")
        
        # Show sample results
        logger.info(f"\nSample results:")
        sample = result_df.select([
            text_column, 
            "sentiment_polarity", 
            "sentiment_subjectivity", 
            "organizations", 
            "people"
        ]).head(3)
        print(sample)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Multiprocessing Sentiment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sentiment_analysis.py input.parquet output.parquet --text-column "text"
    python sentiment_analysis.py data.csv results.csv --text-column "content" --processes 4
    python sentiment_analysis.py news.json analyzed.json --text-column "article_text" --chunk-size 50
        """
    )
    
    parser.add_argument(
        "input_file", 
        help="Input file path (parquet, csv, or json)"
    )
    parser.add_argument(
        "output_file", 
        help="Output file path (parquet, csv, or json)"
    )
    parser.add_argument(
        "--text-column", 
        required=True,
        help="Name of the text column to analyze"
    )
    parser.add_argument(
        "--processes", 
        type=int, 
        default=None,
        help="Number of processes to use (default: CPU count - 1)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=100,
        help="Chunk size for parallel processing (default: 100)"
    )
    parser.add_argument(
        "--model", 
        default="en_core_web_lg",
        help="spaCy model to use (default: en_core_web_lg)"
    )
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Attempt to use GPU for spaCy (main process only)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SentimentAnalyzer(model_name=args.model, use_gpu=args.gpu)
    
    # Process the file
    analyzer.analyze_file(
        input_file=args.input_file,
        output_file=args.output_file,
        text_column=args.text_column,
        n_processes=args.processes,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()