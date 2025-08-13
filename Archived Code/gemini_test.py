import polars as pl
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta

# Sentiment and Text Analysis Libraries
from textblob import TextBlob
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired # For BERTopic custom representation
# import spacy # Uncomment if using spaCy for NER
# from fuzzywuzzy import fuzz # Uncomment if using fuzzy matching for company names
# from transformers import pipeline # Uncomment if using FinBERT or other Hugging Face models

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score

# Traditional ML Models
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Neural Network (TensorFlow/Keras)
import tensorflow as tf
from tensorflow import keras

print("Starting comprehensive sentiment analysis and stock prediction script...")

# --- Configuration ---
# Define paths to your data files
NEWS_DATA_PATH = "YOUR_NEWS_DATA_PATH.csv" # e.g., CSV of news articles
JSE_COMPANIES_PATH = "YOUR_JSE_COMPANIES_PATH.csv" # e.g., CSV of JSE company names and symbols
STOCK_DATA_PATH = "YOUR_STOCK_DATA_PATH.csv" # e.g., CSV of historical stock prices

# --- 1. Data Loading and Initial Cleaning ---
print("\n--- 1. Data Loading and Initial Cleaning ---")

def load_and_clean_data():
    """
    Loads raw news, company, and stock data.
    Placeholder for actual data loading and initial cleaning steps.
    """
    try:
        # YOUR_DATA_LOADING_CODE_HERE
        # Example (replace with your actual loading logic):
        news_df = pl.read_csv(NEWS_DATA_PATH).with_columns(
            pl.col('published_date').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias('date') # Adjust format as needed
        )
        jse_companies_df = pl.read_csv(JSE_COMPANIES_PATH)
        stock_df = pl.read_csv(STOCK_DATA_PATH).with_columns(
            pl.col('Date').str.strptime(pl.Datetime, "%Y-%m-%d").alias('date'), # Adjust format
            pl.col('Symbol').cast(pl.Utf8)
        )

        # Basic cleaning (adjust as per your original notebook)
        news_df = news_df.drop_nulls(['title', 'content'])
        stock_df = stock_df.drop_nulls(['Symbol', 'Date', 'Close'])

        print("Data loaded: News, JSE Companies, Stock Data.")
        return news_df, jse_companies_df, stock_df
    except Exception as e:
        print(f"Error loading data: {e}. Please ensure paths are correct and data exists.")
        exit()

news_df, jse_companies_df, stock_df = load_and_clean_data()

# Example JSE company names for matching (replace with actual data)
jse_company_names = jse_companies_df['Company_Name'].to_list() # Assuming a 'Company_Name' column
jse_company_symbols = {row['Company_Name']: row['Symbol'] for row in jse_companies_df.iter_rows(named=True)} # Assuming 'Symbol' column


# --- 2. Sentiment Analysis ---
print("\n--- 2. Sentiment Analysis ---")

def get_textblob_sentiment(text):
    """Calculates TextBlob sentiment polarity."""
    if text:
        return TextBlob(text).sentiment.polarity
    return 0.0

def get_textblob_subjectivity(text):
    """Calculates TextBlob sentiment subjectivity."""
    if text:
        return TextBlob(text).sentiment.subjectivity
    return 0.0

# Apply TextBlob
print("Applying TextBlob sentiment analysis...")
news_df = news_df.with_columns([
    pl.col('content').apply(get_textblob_sentiment).alias('sentiment'),
    pl.col('content').apply(get_textblob_subjectivity).alias('subjectivity')
])

# BERTopic (simplified, as full training requires more context/computation)
print("Applying BERTopic for topic modeling (simplified)...")
# For a full implementation, you'd fit BERTopic on a corpus of texts.
# representation_model = KeyBERTInspired()
# topic_model = BERTopic(representation_model=representation_model, verbose=True)
# topics, probs = topic_model.fit_transform(news_df['content'].to_list())
# news_df = news_df.with_columns(pl.Series(name="topic", values=topics))
# print("BERTopic applied. Topics assigned (simplified).")
print("BERTopic requires extensive training; skipping full application for this template.")


# --- 3. Company Name Matching and Article Aggregation ---
print("\n--- 3. Company Name Matching and Article Aggregation ---")

# Improved find_company_symbol function (conceptual - implement fuzzy matching/NER fully)
def find_company_symbol(text, jse_names, jse_symbols_map):
    """
    Improved function to find company symbols in text using fuzzy matching or NER.
    This is a conceptual implementation; expand with spacy/fuzzywuzzy.
    """
    found_symbols = []
    # Basic keyword matching (expand with fuzzy logic/NER)
    for name in jse_names:
        if re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE):
            found_symbols.append(jse_symbols_map.get(name))
            # If using fuzzywuzzy, you'd compare text against jse_names
            # score = fuzz.partial_ratio(name.lower(), text.lower())
            # if score > THRESHOLD:
            #     found_symbols.append(jse_symbols_map.get(name))

    # If using spaCy for NER:
    # doc = nlp(text)
    # for ent in doc.ents:
    #     if ent.label_ == "ORG": # Or a custom financial entity label
    #         # Compare ent.text with jse_names using fuzzy matching
    #         pass # YOUR_NER_MATCHING_LOGIC_HERE
    return list(set(sym for sym in found_symbols if sym is not None))

print("Matching news articles to JSE companies...")
# Note: This part can be computationally intensive for large datasets.
# Consider optimizing or running in chunks.
news_df = news_df.with_columns(
    pl.col('content').apply(lambda x: find_company_symbol(x, jse_company_names, jse_company_symbols)).alias('matched_symbols')
)

# Expand news_df to have one row per news-company match
news_df_exploded = news_df.explode('matched_symbols').rename({'matched_symbols': 'symbol'})
news_df_exploded = news_df_exploded.filter(pl.col('symbol').is_not_null())

# Aggregate sentiment by day and company
print("Aggregating sentiment by day and company...")
daily_sentiment = news_df_exploded.group_by(['date', 'symbol']).agg([
    pl.col('sentiment').mean().alias('avg_sentiment'),
    pl.col('sentiment').std().alias('std_sentiment'),
    pl.col('subjectivity').mean().alias('avg_subjectivity'),
    pl.count().alias('news_count')
]).with_columns(
    pl.col('date').dt.date().alias('trade_date') # Extract date part for joining
).drop('date')

# --- 4. Data Integration (News Sentiment + Stock Prices) ---
print("\n--- 4. Data Integration (News Sentiment + Stock Prices) ---")

# Ensure 'date' columns are consistent for joining
stock_df = stock_df.with_columns(pl.col('date').dt.date().alias('trade_date'))

# Join sentiment data with stock data
integrated_data = stock_df.join(
    daily_sentiment,
    on=['trade_date', 'symbol'],
    how='left'
)

# Handle missing sentiment data (e.g., fill with 0 for days with no news)
integrated_data = integrated_data.with_columns([
    pl.col('avg_sentiment').fill_null(0.0),
    pl.col('std_sentiment').fill_null(0.0),
    pl.col('avg_subjectivity').fill_null(0.0),
    pl.col('news_count').fill_null(0).cast(pl.Int64),
    (pl.col('news_count') > 0).alias('has_news') # Indicate if news was present
])

# Create sentiment category for classification
integrated_data = integrated_data.with_columns(
    pl.when(pl.col('avg_sentiment') > 0.05) # Threshold can be tuned
    .then(pl.lit('Positive'))
    .when(pl.col('avg_sentiment') < -0.05) # Threshold can be tuned
    .then(pl.lit('Negative'))
    .otherwise(pl.lit('Neutral'))
    .alias('sentiment_category')
)

print(f"Integrated data sample:\n{integrated_data.head()}")
print(f"Integrated data shape: {integrated_data.shape}")


# --- 5. Feature Engineering (Lagged Features for Prediction) ---
print("\n--- 5. Feature Engineering ---")

def create_lagged_features(df, group_col='symbol', date_col='trade_date', lags=[1, 3, 5]):
    """Creates lagged features for sentiment and stock prices."""
    df_sorted = df.sort([group_col, date_col])
    for lag in lags:
        df_sorted = df_sorted.with_columns([
            pl.col('avg_sentiment').shift(lag).over(group_col).alias(f'lag_sentiment_{lag}'),
            pl.col('news_count').shift(lag).over(group_col).alias(f'lag_news_count_{lag}'),
            pl.col('Close').shift(lag).over(group_col).alias(f'lag_close_{lag}'),
            pl.col('Volume').shift(lag).over(group_col).alias(f'lag_volume_{lag}')
        ])
    # Define target variable: next day's return
    df_sorted = df_sorted.with_columns(
        ((pl.col('Close').shift(-1).over(group_col) - pl.col('Close')) / pl.col('Close')).alias('next_day_return')
    )
    # Drop rows with NaN values resulting from lagging
    return df_sorted.drop_nulls()

integrated_data_features = create_lagged_features(integrated_data.clone(), lags=[1, 2, 3, 5])
print(f"Data with lagged features sample:\n{integrated_data_features.head()}")
print(f"Data with lagged features shape: {integrated_data_features.shape}")

# Filter out inf/-inf in 'next_day_return' if any
integrated_data_features = integrated_data_features.filter(
    pl.col('next_day_return').is_finite()
)

# --- 6. Machine Learning Model Implementation ---
print("\n--- 6. Machine Learning Model Implementation ---")

# Convert to Pandas for sklearn/keras compatibility (Polars is great, but these libs expect Pandas)
ml_data = integrated_data_features.to_pandas()

# Drop rows where the target is NaN (e.g., last day's return)
ml_data.dropna(subset=['next_day_return'], inplace=True)

# Define features and target for Regression (Predicting 'next_day_return')
regression_features = [col for col in ml_data.columns if 'lag_' in col or col in ['avg_sentiment', 'avg_subjectivity', 'news_count']]
regression_target = 'next_day_return'

X_reg = ml_data[regression_features]
y_reg = ml_data[regression_target]

# Drop rows with NaN in features if any (after feature engineering)
X_reg.dropna(inplace=True)
y_reg = y_reg.loc[X_reg.index] # Align target with cleaned features

# Split data using TimeSeriesSplit for time-series integrity
tscv = TimeSeriesSplit(n_splits=5)

print("\n--- Training Regression Models ---")
# RandomForest Regressor
print("Training RandomForestRegressor...")
for train_index, test_index in tscv.split(X_reg):
    X_train_reg, X_test_reg = X_reg.iloc[train_index], X_reg.iloc[test_index]
    y_train_reg, y_test_reg = y_reg.iloc[train_index], y_reg.iloc[test_index]

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_reg, y_train_reg)
    y_pred_rf = rf_model.predict(X_test_reg)
    rmse_rf = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))
    r2_rf = r2_score(y_test_reg, y_pred_rf)
    print(f"  RF - RMSE: {rmse_rf:.4f}, R2: {r2_rf:.4f}")

# XGBoost Regressor
print("Training XGBoost Regressor...")
for train_index, test_index in tscv.split(X_reg):
    X_train_reg, X_test_reg = X_reg.iloc[train_index], X_reg.iloc[test_index]
    y_train_reg, y_test_reg = y_reg.iloc[train_index], y_reg.iloc[test_index]

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_reg, y_train_reg)
    y_pred_xgb = xgb_model.predict(X_test_reg)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_reg, y_pred_xgb))
    r2_xgb = r2_score(y_test_reg, y_pred_xgb)
    print(f"  XGB - RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}")

# LightGBM Regressor
print("Training LightGBM Regressor...")
for train_index, test_index in tscv.split(X_reg):
    X_train_reg, X_test_reg = X_reg.iloc[train_index], X_reg.iloc[test_index]
    y_train_reg, y_test_reg = y_reg.iloc[train_index], y_reg.iloc[test_index]

    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    lgb_model.fit(X_train_reg, y_train_reg)
    y_pred_lgb = lgb_model.predict(X_test_reg)
    rmse_lgb = np.sqrt(mean_squared_error(y_test_reg, y_pred_lgb))
    r2_lgb = r2_score(y_test_reg, y_pred_lgb)
    print(f"  LGBM - RMSE: {rmse_lgb:.4f}, R2: {r2_lgb:.4f}")


print("\n--- Training Neural Network for Sentiment Classification ---")

# Filter data where news was present to train on meaningful sentiment
classification_data = ml_data[ml_data['has_news'] == True].copy()

# Define features (X) and target (y) for Classification
# Using 'avg_sentiment', 'avg_subjectivity', 'news_count' as primary features
# You can add more features like lagged sentiment or lagged stock prices
# if they are deemed relevant for sentiment category prediction.
numerical_features_nn = ['avg_sentiment', 'avg_subjectivity', 'news_count']
X_nn = classification_data[numerical_features_nn]
y_nn = classification_data['sentiment_category']

# Drop rows with NaN in features
X_nn.dropna(inplace=True)
y_nn = y_nn.loc[X_nn.index]

# Handle categorical target variable: One-Hot Encoding
encoder_nn = OneHotEncoder(sparse_output=False)
y_encoded_nn = encoder_nn.fit_transform(y_nn.values.reshape(-1, 1))
sentiment_classes_nn = encoder_nn.categories_[0]

# Split data into training and testing sets for NN (Stratified for balanced classes)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_nn, y_encoded_nn, test_size=0.2, random_state=42, stratify=y_encoded_nn
)

# Scale numerical features for NN
scaler_nn = StandardScaler()
X_train_scaled_nn = scaler_nn.fit_transform(X_train_nn)
X_test_scaled_nn = scaler_nn.transform(X_test_nn)

# Build the Neural Network Model
model_nn = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled_nn.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(y_encoded_nn.shape[1], activation='softmax') # Output layer with softmax
])

# Compile the model
model_nn.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Train the model
print("Training Neural Network...")
history_nn = model_nn.fit(X_train_scaled_nn, y_train_nn, epochs=20, batch_size=32,
                           validation_split=0.1, verbose=0)

# Evaluate the model
loss_nn, accuracy_nn = model_nn.evaluate(X_test_scaled_nn, y_test_nn, verbose=0)
print(f"Neural Network Model - Test Accuracy: {accuracy_nn:.4f}")

# Make predictions
y_pred_probs_nn = model_nn.predict(X_test_scaled_nn, verbose=0)
y_pred_nn = np.argmax(y_pred_probs_nn, axis=1)
y_test_labels_nn = np.argmax(y_test_nn, axis=1)

# Generate classification report
print("Neural Network Model - Classification Report:")
print(classification_report(y_test_labels_nn, y_pred_nn, target_names=sentiment_classes_nn))


# --- 7. Enhanced Visualization (Conceptual) ---
print("\n--- 7. Enhanced Visualization (Conceptual) ---")
# This section would contain plotting code using matplotlib/seaborn
# import matplotlib.pyplot as plt
# import seaborn as sns

# Example: Sentiment trends over time
# plt.figure(figsize=(12, 6))
# integrated_data.group_by('trade_date').agg(pl.col('avg_sentiment').mean().alias('daily_avg_sentiment')) \
#     .sort('trade_date').to_pandas().set_index('trade_date')['daily_avg_sentiment'].plot()
# plt.title('Daily Average Sentiment Over Time')
# plt.xlabel('Date')
# plt.ylabel('Average Sentiment Polarity')
# plt.grid(True)
# plt.show()

# Example: Correlation plot
# sns.heatmap(ml_data[regression_features + [regression_target]].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix of Features and Target')
# plt.show()

print("\nScript execution complete. Please review the output and fill in the placeholder sections.")