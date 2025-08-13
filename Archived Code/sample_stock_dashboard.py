import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("📈 Stock Prediction Dashboard")

# --- Sidebar Inputs ---
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# --- Fetch Data ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

try:
    data = fetch_stock_data(symbol, period, interval)
    if data.empty:
        st.warning("No data found for the selected stock symbol.")
    else:
        # --- Plot Price Chart ---
        st.subheader(f"📊 Historical Prices: {symbol.upper()}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close Price"))
        fig.update_layout(title=f"{symbol.upper()} Stock Price", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        # --- Dummy Predictions ---
        st.subheader("🤖 Model Predictions (Example)")

        last_close = data["Close"].iloc[-1]
        model_preds = {
            "Linear Regression": last_close * np.random.uniform(0.98, 1.02),
            "XGBoost": last_close * np.random.uniform(0.97, 1.03),
            "LSTM": last_close * np.random.uniform(0.96, 1.04),
        }

        pred_df = pd.DataFrame({
            "Model": list(model_preds.keys()),
            "Predicted Price": [round(v, 2) for v in model_preds.values()]
        })

        st.dataframe(pred_df, use_container_width=True)

        pred_bar = go.Figure([go.Bar(x=pred_df["Model"], y=pred_df["Predicted Price"], marker_color='indianred')])
        pred_bar.update_layout(title="Predicted Prices by Model", yaxis_title="Price (USD)")
        st.plotly_chart(pred_bar, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching data: {e}")
