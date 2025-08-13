import spacy
import polars as pl
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Load spaCy model and add TextBlob component for sentiment
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('spacytextblob')
spacy.require_gpu()

# Load FinBERT model for financial sentiment analysis
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Create FinBERT pipeline
finbert_pipeline = pipeline(
    "sentiment-analysis",
    model=finbert_model,
    tokenizer=finbert_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True,
    truncation=True,
    max_length=512
)

def clean_text(df: pl.DataFrame, text_column) -> pl.DataFrame:
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

def extract_organizations(doc):
    """Extract organization entities from spaCy doc."""
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
    return list(set(orgs))  # Remove duplicates

def get_finbert_sentiment(text):
    """
    Get financial sentiment from FinBERT model.
    Returns the label with highest confidence score.
    """
    if not text or len(text.strip()) == 0:
        return 'neutral'
    
    try:
        # Tokenize and truncate properly using the tokenizer
        tokens = finbert_tokenizer.encode(
            text, 
            truncation=True, 
            max_length=512,
            add_special_tokens=True
        )
        
        # Decode back to text to ensure proper truncation
        truncated_text = finbert_tokenizer.decode(tokens, skip_special_tokens=True)
        
        results = finbert_pipeline(truncated_text)
        # Get the label with highest score
        best_result = max(results[0], key=lambda x: x['score'])
        return best_result['label'].lower()
    except Exception as e:
        print(f"Error processing text with FinBERT: {e}")
        return 'neutral'

def analyze_text_with_ner(text):
    """
    Analyze sentiment and extract organizations from a single text string.
    """
    doc = nlp(text)
    organizations = extract_organizations(doc)
    
    return {
        'text': text,
        'title_polarity': doc._.blob.polarity,
        'title_subjectivity': doc._.blob.subjectivity,
        'title_sentiment_label': 'positive' if doc._.blob.polarity > 0 else 'negative' if doc._.blob.polarity < 0 else 'neutral',
        'title_organizations': organizations,
        'title_org_count': len(organizations)
    }

def add_sentiment_ner_to_dataframe(df, text_column):
    """
    Add sentiment analysis and organization extraction columns to existing polars DataFrame.
    Now includes FinBERT financial sentiment analysis.
    
    Args:
        df: polars DataFrame containing text data
        text_column: name of the column containing text to analyze
    
    Returns:
        polars DataFrame with added sentiment and NER columns
    """
    # Extract texts and process with spaCy
    texts = df[text_column].to_list()
    docs = list(nlp.pipe(texts))
    
    # Create sentiment and NER data
    analysis_data = []
    for i, doc in enumerate(docs):
        organizations = extract_organizations(doc)
        
        # Get FinBERT sentiment for the text
        finbert_sentiment = get_finbert_sentiment(texts[i])
        
        analysis_data.append({
            'text_fin_polarity': doc._.blob.polarity,
            'text_fin_subjectivity': doc._.blob.subjectivity,
            'text_fin_sentiment_label': 'positive' if doc._.blob.polarity > 0 else 'negative' if doc._.blob.polarity < 0 else 'neutral',
            'text_fin_sentiment': finbert_sentiment,
            #'title_organizations': organizations,
            #'title_org_count': len(organizations)
        })
    
    # Create DataFrame from analysis results
    analysis_df = pl.DataFrame(analysis_data)
    
    # Concatenate with original DataFrame
    return pl.concat([df, analysis_df], how="horizontal")

def add_text_finbert_sentiment(df, text_column='text'):
    """
    Add FinBERT sentiment analysis for the text column if it exists.
    
    Args:
        df: polars DataFrame
        text_column: name of the text column to analyze
    
    Returns:
        polars DataFrame with added text_fin_sentiment column
    """
    if text_column not in df.columns:
        print(f"Warning: Column '{text_column}' not found in DataFrame")
        return df
    
    # Get FinBERT sentiment for text column
    texts = df[text_column].to_list()
    text_sentiments = [get_finbert_sentiment(text) for text in texts]
    
    # Add the new column
    return df.with_columns(
        pl.Series('text_fin_sentiment', text_sentiments)
    )

# Example usage
if __name__ == "__main__":
    fileloc = 'C:/Users/Joshh/Projects/Stocks/Data/joined_newspaper_data_v3.parquet'
    df = pl.read_parquet(fileloc)
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the title column
    df_cleaned = clean_text(df, 'text')
    
    # Add sentiment analysis and NER to DataFrame (includes text_fin_sentiment)
    df_with_analysis = add_sentiment_ner_to_dataframe(df_cleaned, 'text')
    
    # Add FinBERT sentiment for text column if it exists
    # df_with_analysis = add_text_finbert_sentiment(df_with_analysis, 'text')
    
    print("DataFrame with sentiment and NER analysis:")
    print(df_with_analysis)
    print()
    
    # Show the new financial sentiment columns
    if 'text_fin_sentiment' in df_with_analysis.columns:
        print("Text FinBERT sentiment distribution:")
        print(df_with_analysis['text_fin_sentiment'].value_counts())
        print()

    outfileLoc = 'C:/Users/Joshh/Projects/Stocks/Data/joined_newspaper_data_v4.parquet'
    df_with_analysis.write_parquet(outfileLoc)