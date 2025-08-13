# Imports
import nltk
from textblob import TextBlob
from bs4 import BeautifulSoup
import newspaper
from urllib.parse import urlparse
import os
import polars as pl
import pandas as pd
import spacy
from bertopic import BERTopic

# Function to perform NER
def extract_entities(text: str) -> list:
    """
    This functions accepts a text as a string and returns a list of entities recognized using Spacy NER
    """
    # Force Spacy to utilize CUDA accelaration -- This improves performance on the large model
    # spacy.prefer_gpu()
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_lg")

    doc = nlp(text)
    
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to analyze sentiment
def analyze_sentiment(baseArticle: newspaper.Article) -> dict:
    baseArticle.parse()

    baseArticle.nlp()

    # Use TextBlob for sentiment analysis
    blob = TextBlob(baseArticle.text)

    # Extract named entities and sub categories
    organizations, people, products = [], [], []

    namedEntities = extract_entities(baseArticle.text)
    organizations = [entity[0] for entity in namedEntities if entity[1] == "ORG"]
    people = [entity[0] for entity in namedEntities if entity[1] == "PERSON"]
    products = [entity[0] for entity in namedEntities if entity[1] == "PRODUCT"]

    # print(organizations)
    # print(people)
    # print(products)

    article_dict = {
        'title': [baseArticle.title],
        'text': [baseArticle.text],
        'authors': [list(baseArticle.authors) if isinstance(baseArticle.authors, set) else baseArticle.authors],
        'publish_date': [baseArticle.publish_date],
        'tags': [list(baseArticle.tags) if isinstance(baseArticle.tags, set) else baseArticle.tags],
        'keywords': [list(baseArticle.keywords) if isinstance(baseArticle.keywords, set) else baseArticle.keywords],
        'summary': [baseArticle.summary],
        'polarity': [blob.sentiment[0]],
        'subjectivity': [blob.sentiment[1]],
        'named_entities': [namedEntities],
        'named_entities_orgs': [list(organizations) if isinstance(organizations, set) else organizations],
        'named_entities_people': [list(people) if isinstance(people, set) else people],
        'named_entities_products': [list(products) if isinstance(products, set) else products]
    }

    print(article_dict)

    return pl.DataFrame(article_dict)



# Function to get articles and perform sentiment analysis
def get_sentiment(files: list) -> pl.DataFrame:
    # Initialize the newspaper module
    testUrl = 'https://www.jamaicaobserver.com/2025/04/27/fa-cup-glory-wont-salvage-man-citys-troubled-season-guardiola'
    article = newspaper.Article(testUrl)

    articlesDf = pl.DataFrame()

    for file in files[:10]:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()

        # Set the HTML content directly
        article.html = html_content
        article.download_state = 2
        
        #print(analyze_sentiment(article))
        articlesDf = pl.concat([articlesDf, analyze_sentiment(article)], how = "vertical")

    return articlesDf


def main():

    folders = [r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\ICInsider',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaGleaner'
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JIS',
               r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaObserver']

    files = get_files(folders)
    
    print(get_sentiment(files))

    

if __name__ == "__main__":
    main()