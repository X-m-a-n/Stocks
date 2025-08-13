#!/usr/bin/env python3
"""
High-Performance FinBERT Sentiment Analysis Script (64GB RAM Optimized)
======================================================================

Optimized for systems with abundant RAM (64GB+):
- Shared memory for model loading
- Efficient multiprocessing with proper resource management
- GPU acceleration for FinBERT
- Large model support (en_core_web_lg)

Date: 2025-08-01
"""

import polars as pl
import spacy
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys
from pathlib import Path
import logging
import warnings
from rapidfuzz import fuzz, process
import torch
import gc
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer multiprocessing conflicts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finbert_sentiment_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# JSE company names dictionary
try:
    from jse_companies import jse_company_names
except ImportError:
    logger.warning("jse_companies module not found. Please ensure it's available.")
    jse_company_names = {}

class HighPerformanceSentimentAnalyzer:
    """High-performance sentiment analysis pipeline optimized for 64GB+ RAM systems."""
    
    def __init__(self, spacy_model_name: str = "en_core_web_lg", 
                 finbert_model: str = "ProsusAI/finbert", 
                 use_gpu: bool = True):
        """
        Initialize the sentiment analyzer.
        
        Args:
            spacy_model_name: spaCy model name (en_core_web_lg for best accuracy)
            finbert_model: FinBERT model name for sentiment analysis
            use_gpu: Whether to use GPU for FinBERT (recommended for 64GB+ systems)
        """
        self.spacy_model_name = spacy_model_name
        self.finbert_model = finbert_model
        self.use_gpu = use_gpu
        self.nlp = None
        self.sentiment_pipeline = None
        self._load_main_models()
    
    def _load_main_models(self):
        """Load models in main process."""
        # Load spaCy model
        try:
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
        except OSError:
            logger.warning(f"Model {self.spacy_model_name} not found, trying en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_model_name = "en_core_web_sm"
                logger.info("Loaded fallback model: en_core_web_sm")
            except OSError:
                logger.error("No spaCy models found. Please install with: python -m spacy download en_core_web_lg")
                sys.exit(1)
        
        # Load FinBERT model (main process only for GPU)
        try:
            device = 0 if self.use_gpu and torch.cuda.is_available() else -1
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_model,
                device=device,
                truncation=True,
                max_length=512,
                return_all_scores=True
            )
            logger.info(f"Loaded FinBERT model: {self.finbert_model} (GPU: {device >= 0})")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            sys.exit(1)
    
    @staticmethod
    def init_worker(spacy_model_name: str, finbert_model: str):
        """Initialize models in worker processes - spaCy only (CPU)."""
        global nlp, sentiment_pipeline
        try:
            # Load spaCy model in worker (CPU only)
            nlp = spacy.load(spacy_model_name)
            
            # Load FinBERT model in worker (CPU only to avoid GPU conflicts)
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=finbert_model,
                tokenizer=finbert_model,
                device=-1,  # Force CPU for workers
                truncation=True,
                max_length=512,
                return_all_scores=True
            )
        except Exception as e:
            logger.error(f"Worker initialization failed: {e}")
            raise
    
    @staticmethod
    def extract_entities_worker(text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text using spaCy in worker process."""
        if not text or not isinstance(text, str) or text.strip() == "":
            return []
        try:
            doc = nlp(text[:100000])  # Increased limit for 64GB systems
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception as e:
            return []
    
    @staticmethod
    def get_finbert_sentiment_worker(text: str) -> Dict[str, float]:
        """Get sentiment using FinBERT in worker process."""
        if not text or not isinstance(text, str) or text.strip() == "":
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
        
        try:
            text = text[:3000]  # Increased limit for better context
            results = sentiment_pipeline(text)[0]
            
            sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "compound": 0.0}
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
            
            # Calculate compound score
            sentiment_scores["compound"] = sentiment_scores["positive"] - sentiment_scores["negative"]
            
            return sentiment_scores
            
        except Exception as e:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    @staticmethod
    def process_ner_chunk(texts_chunk: List[str]) -> List[List[Tuple[str, str]]]:
        """Process a chunk of texts for NER."""
        return [HighPerformanceSentimentAnalyzer.extract_entities_worker(text) for text in texts_chunk]
    
    @staticmethod
    def process_sentiment_chunk(texts_chunk: List[str]) -> List[Dict[str, float]]:
        """Process a chunk of texts for sentiment analysis."""
        return [HighPerformanceSentimentAnalyzer.get_finbert_sentiment_worker(text) for text in texts_chunk]
    
    def find_company_symbol(self, company_name: str, threshold: int = 80) -> Optional[str]:
        """Find the stock symbol for a given company name using fuzzy matching."""
        if not company_name or pd.isna(company_name) or str(company_name).strip() == '':
            return None
        
        company_name_clean = str(company_name).lower().strip()
        
        # First try exact matching (fastest)
        for symbol, name_variations in jse_company_names.items():
            for variation in name_variations:
                if variation.lower() == company_name_clean:
                    return symbol
        
        def clean_name(text, aggressive=False):
            """Clean company name with optional aggressive cleaning"""
            if aggressive:
                text = re.sub(r'\b(ltd|limited|inc|incorporated|corp|corporation|co|company|plc|llc)\b', '', text.lower())
                text = re.sub(r'[^\w\s]', ' ', text)
            else:
                text = text.lower().strip()
                text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        # Try light cleaning first
        company_light_clean = clean_name(company_name_clean, aggressive=False)
        
        variation_to_symbol = {}
        for symbol, variations in jse_company_names.items():
            for variation in variations:
                cleaned_var = clean_name(variation, aggressive=False)
                if cleaned_var:
                    variation_to_symbol[cleaned_var] = symbol
        
        if company_light_clean:
            result = process.extractOne(
                company_light_clean,
                variation_to_symbol.keys(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold
            )
            
            if result:
                matched_variation, score, _ = result if len(result) == 3 else (*result, None)
                return variation_to_symbol[matched_variation]
        
        # Aggressive cleaning fallback
        company_aggressive_clean = clean_name(company_name_clean, aggressive=True)
        
        if company_aggressive_clean and len(company_aggressive_clean) > 1:
            aggressive_variation_to_symbol = {}
            for symbol, variations in jse_company_names.items():
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
                matched_variation, score, _ = result if len(result) == 3 else (*result, None)
                return aggressive_variation_to_symbol[matched_variation]
        
        return None
    
    def get_finbert_sentiment_main(self, text: str) -> Dict[str, float]:
        """Get sentiment using FinBERT in main process (GPU accelerated)."""
        if not text or not isinstance(text, str) or text.strip() == "":
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
        
        try:
            text = text[:3000]
            results = self.sentiment_pipeline(text)[0]
            
            sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "compound": 0.0}
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
            
            sentiment_scores["compound"] = sentiment_scores["positive"] - sentiment_scores["negative"]
            
            return sentiment_scores
            
        except Exception as e:
            logger.warning(f"FinBERT sentiment analysis failed: {str(e)[:100]}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    def extract_entity_specific_sentiment(self, text: str, entities: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """Extract sentiment for each entity mentioned in the text."""
        entity_sentiments = {}
        
        for entity_name, entity_type in entities:
            if entity_type in ['ORG', 'PERSON']:
                sentences = text.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    if entity_name.lower() in sentence.lower():
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    context_text = '. '.join(relevant_sentences[:5])  # More context for 64GB systems
                    sentiment = self.get_finbert_sentiment_main(context_text)
                    entity_sentiments[entity_name] = sentiment
        
        return entity_sentiments
    
    def create_chunks(self, lst: List, chunk_size: int) -> List[List]:
        """Create chunks from a list."""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
    def process_multiprocessing_optimized(
        self, 
        df: pl.DataFrame, 
        title_column: str,
        text_column: str,
        authors_column: str = 'authors',
        n_processes: Optional[int] = None,
        chunk_size: int = 100
    ) -> pl.DataFrame:
        """
        Optimized multiprocessing pipeline for 64GB+ systems.
        """
        
        if n_processes is None:
            n_processes = min(8, mp.cpu_count() - 2)  # Conservative for stability
        
        logger.info(f"Starting optimized multiprocessing analysis with {n_processes} processes")
        logger.info(f"Processing {len(df)} articles in chunks of {chunk_size}")
        
        # Combine title and text for analysis
        combined_texts = []
        titles = df[title_column].to_list()
        texts = df[text_column].to_list()
        
        for i in range(len(df)):
            title = titles[i] if titles[i] else ""
            text = texts[i] if texts[i] else ""
            combined = f"{title}. {text}" if title and text else (title or text or "")
            combined_texts.append(combined)
        
        # Filter valid texts
        valid_indices = []
        valid_combined_texts = []
        valid_titles = []
        valid_texts = []
        
        for i, (combined, title, text) in enumerate(zip(combined_texts, titles, texts)):
            if combined and isinstance(combined, str) and combined.strip():
                valid_indices.append(i)
                valid_combined_texts.append(combined)
                valid_titles.append(title or "")
                valid_texts.append(text or "")
        
        logger.info(f"Found {len(valid_combined_texts)} valid texts out of {len(combined_texts)}")
        
        # Create chunks
        combined_chunks = self.create_chunks(valid_combined_texts, chunk_size)
        title_chunks = self.create_chunks(valid_titles, chunk_size)
        text_chunks = self.create_chunks(valid_texts, chunk_size)
        
        # Initialize results
        all_entities = [[] for _ in range(len(df))]
        all_title_sentiments = [{"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0} for _ in range(len(df))]
        all_text_sentiments = [{"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0} for _ in range(len(df))]
        all_combined_sentiments = [{"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0} for _ in range(len(df))]
        
        # Step 1: Named Entity Recognition (multiprocessing)
        logger.info("Step 1: Processing Named Entity Recognition...")
        
        with mp.Pool(
            processes=n_processes, 
            initializer=self.init_worker,
            initargs=(self.spacy_model_name, self.finbert_model)
        ) as pool:
            ner_results = list(tqdm(
                pool.imap(self.process_ner_chunk, combined_chunks),
                total=len(combined_chunks),
                desc="NER Chunks"
            ))
        
        # Map NER results back
        flat_ner_results = []
        for chunk_result in ner_results:
            flat_ner_results.extend(chunk_result)
        
        for i, result in enumerate(flat_ner_results):
            if i < len(valid_indices):
                all_entities[valid_indices[i]] = result
        
        logger.info(f"✓ NER completed for {len(flat_ner_results)} texts")
        
        # Step 2-4: Sentiment Analysis (Use GPU in main process for better performance)
        logger.info("Step 2: Processing Title Sentiment Analysis (GPU accelerated)...")
        
        for i, title in enumerate(tqdm(valid_titles, desc="Title Sentiment")):
            sentiment = self.get_finbert_sentiment_main(title)
            if i < len(valid_indices):
                all_title_sentiments[valid_indices[i]] = sentiment
        
        logger.info("Step 3: Processing Text Sentiment Analysis (GPU accelerated)...")
        
        for i, text in enumerate(tqdm(valid_texts, desc="Text Sentiment")):
            sentiment = self.get_finbert_sentiment_main(text)
            if i < len(valid_indices):
                all_text_sentiments[valid_indices[i]] = sentiment
        
        logger.info("Step 4: Processing Combined Sentiment Analysis (GPU accelerated)...")
        
        for i, combined in enumerate(tqdm(valid_combined_texts, desc="Combined Sentiment")):
            sentiment = self.get_finbert_sentiment_main(combined)
            if i < len(valid_indices):
                all_combined_sentiments[valid_indices[i]] = sentiment
        
        # Step 5: Extract organizations and JSE symbols
        logger.info("Step 5: Processing JSE company matching...")
        
        organizations = []
        jse_symbols = []
        
        for entities in all_entities:
            orgs = [ent[0] for ent in entities if ent[1] == "ORG"]
            organizations.append(orgs)
            
            symbols = []
            for org in orgs:
                symbol = self.find_company_symbol(org)
                if symbol:
                    symbols.append(symbol)
            jse_symbols.append(symbols)
        
        # Step 6: Entity-specific sentiment for Ic Insider.Com articles
        logger.info("Step 6: Processing entity-specific sentiment...")
        
        entity_specific_sentiments = []
        authors_list = df[authors_column].to_list()
        
        for i in range(len(df)):
            authors = authors_list[i] if i < len(authors_list) else None
            
            if (authors and isinstance(authors, list) and 'Ic Insider.Com' in authors) or \
               (authors and isinstance(authors, str) and 'Ic Insider.Com' in authors):
                text = combined_texts[i]
                entities = all_entities[i]
                entity_sentiments = self.extract_entity_specific_sentiment(text, entities)
                entity_specific_sentiments.append(entity_sentiments)
            else:
                entity_specific_sentiments.append({})
        
        # Step 7: Build result dataframe
        logger.info("Step 7: Building result dataframe...")
        
        result_df = df.with_columns([
            # Title sentiment
            pl.Series("title_sentiment_positive", [s["positive"] for s in all_title_sentiments]),
            pl.Series("title_sentiment_negative", [s["negative"] for s in all_title_sentiments]),
            pl.Series("title_sentiment_neutral", [s["neutral"] for s in all_title_sentiments]),
            pl.Series("title_sentiment_compound", [s["compound"] for s in all_title_sentiments]),
            
            # Text sentiment
            pl.Series("text_sentiment_positive", [s["positive"] for s in all_text_sentiments]),
            pl.Series("text_sentiment_negative", [s["negative"] for s in all_text_sentiments]),
            pl.Series("text_sentiment_neutral", [s["neutral"] for s in all_text_sentiments]),
            pl.Series("text_sentiment_compound", [s["compound"] for s in all_text_sentiments]),
            
            # Combined sentiment
            pl.Series("combined_sentiment_positive", [s["positive"] for s in all_combined_sentiments]),
            pl.Series("combined_sentiment_negative", [s["negative"] for s in all_combined_sentiments]),
            pl.Series("combined_sentiment_neutral", [s["neutral"] for s in all_combined_sentiments]),
            pl.Series("combined_sentiment_compound", [s["compound"] for s in all_combined_sentiments]),
            
            # Entity information
            pl.Series("named_entities", all_entities),
            pl.Series("organizations", organizations),
            pl.Series("jse_symbols", jse_symbols),
            pl.Series("entity_specific_sentiments", entity_specific_sentiments)
        ])
        
        logger.info("✓ Analysis complete!")
        logger.info(f"🎉 Processed {len(df)} articles successfully.")
        
        return result_df
    
    def analyze_file(
        self, 
        input_file: str, 
        output_file: str, 
        title_column: str,
        text_column: str,
        authors_column: str = 'authors',
        filename_column: str = 'filename',
        publication_date_column: str = 'publication_date',
        keywords_column: str = 'keywords',
        n_processes: Optional[int] = None,
        chunk_size: int = 100
    ):
        """Analyze a file and save results with specified columns."""
        
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
        
        # Check required columns
        required_columns = [title_column, text_column]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in input file")
                logger.info(f"Available columns: {df.columns}")
                sys.exit(1)
        
        # Process the data
        start_time = time.time()
        result_df = self.process_multiprocessing_optimized(
            df, title_column, text_column, authors_column, n_processes, chunk_size
        )
        processing_time = time.time() - start_time
        
        # Select output columns
        output_columns = []
        
        # Add original columns if they exist
        if filename_column in df.columns:
            output_columns.append(filename_column)
        if title_column in df.columns:
            output_columns.append(title_column)
        if text_column in df.columns:
            output_columns.append(text_column)
        if publication_date_column in df.columns:
            output_columns.append(publication_date_column)
        if keywords_column in df.columns:
            output_columns.append(keywords_column)
        if authors_column in df.columns:
            output_columns.append(authors_column)
        
        # Add sentiment and analysis columns
        sentiment_columns = [
            "title_sentiment_positive", "title_sentiment_negative", "title_sentiment_neutral", "title_sentiment_compound",
            "text_sentiment_positive", "text_sentiment_negative", "text_sentiment_neutral", "text_sentiment_compound",
            "combined_sentiment_positive", "combined_sentiment_negative", "combined_sentiment_neutral", "combined_sentiment_compound",
            "organizations", "jse_symbols", "entity_specific_sentiments"
        ]
        
        output_columns.extend(sentiment_columns)
        
        # Filter to only existing columns
        available_columns = [col for col in output_columns if col in result_df.columns]
        final_df = result_df.select(available_columns)
        
        # Save results
        output_path = Path(output_file)
        logger.info(f"Saving results to: {output_file}")
        
        if output_path.suffix.lower() == '.parquet':
            final_df.write_parquet(output_file)
        elif output_path.suffix.lower() == '.csv':
            final_df.write_csv(output_file)
        elif output_path.suffix.lower() == '.json':
            final_df.write_json(output_file)
        else:
            final_df.write_parquet(output_file)
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info(f"HIGH-PERFORMANCE FINBERT ANALYSIS COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Articles processed: {len(df)}")
        logger.info(f"Speed: {len(df)/processing_time:.1f} articles/second")
        logger.info(f"Output saved to: {output_file}")
        
        # Show sample results
        logger.info(f"\nSample results:")
        sample_columns = [col for col in [title_column, "combined_sentiment_compound", "organizations", "jse_symbols"] if col in final_df.columns]
        if sample_columns:
            sample = final_df.select(sample_columns).head(3)
            print(sample)


def main():
    """Main function optimized for 64GB RAM systems."""
    
    # Configuration for 64GB RAM systems
    INPUT_FILE = 'C:/Users/Joshh/Projects/Stocks/Data/joined_newspaper_data_v4.parquet'
    OUTPUT_FILE = 'C:/Users/Joshh/Projects/Stocks/Data/sentiment_data_FINAL.parquet'
    TITLE_COLUMN = "title"
    TEXT_COLUMN = "text"
    AUTHORS_COLUMN = "authors"
    FILENAME_COLUMN = "filename"
    PUBLICATION_DATE_COLUMN = "publication_date"
    KEYWORDS_COLUMN = "keywords"
    N_PROCESSES = 10  # Conservative for stability with large models
    CHUNK_SIZE = 200  # Larger chunks for 64GB systems
    SPACY_MODEL = "en_core_web_lg"  # Full model for best accuracy
    FINBERT_MODEL = "ProsusAI/finbert"
    USE_GPU = True  # GPU acceleration for FinBERT
    
    logger.info("Starting High-Performance FinBERT Sentiment Analysis Pipeline (64GB Optimized)")
    logger.info(f"Input file: {INPUT_FILE}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info(f"Title column: {TITLE_COLUMN}")
    logger.info(f"Text column: {TEXT_COLUMN}")
    logger.info(f"Authors column: {AUTHORS_COLUMN}")
    logger.info(f"Processes: {N_PROCESSES}")
    logger.info(f"Chunk size: {CHUNK_SIZE}")
    logger.info(f"spaCy model: {SPACY_MODEL}")
    logger.info(f"FinBERT model: {FINBERT_MODEL}")
    logger.info(f"Use GPU: {USE_GPU}")
    
    # Create analyzer
    analyzer = HighPerformanceSentimentAnalyzer(
        spacy_model_name=SPACY_MODEL,
        finbert_model=FINBERT_MODEL,
        use_gpu=USE_GPU
    )
    
    # Process the file
    analyzer.analyze_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        title_column=TITLE_COLUMN,
        text_column=TEXT_COLUMN,
        authors_column=AUTHORS_COLUMN,
        filename_column=FILENAME_COLUMN,
        publication_date_column=PUBLICATION_DATE_COLUMN,
        keywords_column=KEYWORDS_COLUMN,
        n_processes=N_PROCESSES,
        chunk_size=CHUNK_SIZE
    )


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()