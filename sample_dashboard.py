import streamlit as st
import duckdb
import plotly.express as px
import pandas as pd

# Upload CSV
csv = st.file_uploader("Upload your CSV file", type="csv")
if csv:
    # Read uploaded file with pandas first
    df = pd.read_csv(csv)
    
    # Filter UI
    symbol = st.selectbox("Symbol", sorted(df['symbol'].unique()))
    
    # Use DuckDB to filter the pandas DataFrame
    filtered = duckdb.query("SELECT * FROM df WHERE symbol = ?", params=[symbol]).df()
    
    # Dashboard
    st.write(f"Showing data for {symbol}")
    fig = px.bar(filtered, x="trade_date", y="closing_price")
    st.plotly_chart(fig)