import spacy
import polars as pl
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model and add TextBlob component for sentiment
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('spacytextblob')
spacy.require_gpu()

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
    for doc in docs:
        organizations = extract_organizations(doc)
        analysis_data.append({
            'title_polarity': doc._.blob.polarity,
            'title_subjectivity': doc._.blob.subjectivity,
            'title_sentiment_label': 'positive' if doc._.blob.polarity > 0 else 'negative' if doc._.blob.polarity < 0 else 'neutral',
            'title_organizations': organizations,
            'title_org_count': len(organizations)
        })
    
    # Create DataFrame from analysis results
    analysis_df = pl.DataFrame(analysis_data)
    
    # Concatenate with original DataFrame
    return pl.concat([df, analysis_df], how="horizontal")

# Example usage
if __name__ == "__main__":
    fileloc = 'C:/Users/Joshh/Projects/Stocks/Data/joined_newspaper_data.parquet'
    df = pl.read_parquet(fileloc)
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Add sentiment analysis and NER to DataFrame
    df_with_analysis = add_sentiment_ner_to_dataframe(clean_text(df, 'title'), 'title')
    
    print("DataFrame with sentiment and NER analysis:")
    print(df_with_analysis)
    print()

    outfileLoc = fileloc = 'C:/Users/Joshh/Projects/Stocks/Data/joined_newspaper_data_v2.parquet'
    df_with_analysis.write_parquet(outfileLoc)
    
