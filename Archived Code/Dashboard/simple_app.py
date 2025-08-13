import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="JSE Stock Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

# Sample stock symbols (reduced for simplicity)
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "META", "NVDA", "BRK-B", "UNH", "JNJ"
]

# --- Helper Functions ---
def test_api_connection():
    """Test if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_api_request(endpoint, params=None):
    """Make API request with error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running on port 8000.")
        st.info("Run: `python simple_api.py` in another terminal")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🔥 API Error: {str(e)}")
        return None

def plot_price_prediction(historical_data, predictions, symbol, model_name):
    """Create interactive price prediction chart"""
    fig = go.Figure()
    
    # Historical prices
    if historical_data:
        dates = [item['date'] for item in historical_data]
        prices = [item['close'] for item in historical_data]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name="Historical Price",
            line=dict(color="blue", width=2)
        ))
    
    # Predictions
    if predictions:
        pred_dates = [item['date'] for item in predictions]
        pred_prices = [item['predicted_price'] for item in predictions]
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode="lines+markers",
            name=f"{model_name} Prediction",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=6)
        ))
        
        # Add confidence intervals if available
        if predictions and 'confidence_interval_low' in predictions[0]:
            low_prices = [item['confidence_interval_low'] for item in predictions]
            high_prices = [item['confidence_interval_high'] for item in predictions]
            
            fig.add_trace(go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=high_prices + low_prices[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
    
    fig.update_layout(
        title=f"{symbol.upper()} - {model_name} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )
    
    return fig

# --- Main App ---
def main():
    st.title("📈 Simple Stock Prediction Dashboard")
    st.markdown("*Stock price predictions using ARIMA, LSTM, and Sentiment Analysis*")
    
    # Check API connection
    with st.spinner("Checking API connection..."):
        api_connected = test_api_connection()
    
    if not api_connected:
        st.error("🚨 **API Server Not Running**")
        st.markdown("""
        **To start the API server:**
        1. Open a new terminal
        2. Run: `python simple_api.py`
        3. Wait for "API is running" message
        4. Refresh this page
        """)
        
        st.markdown("**Or test with sample data:**")
        if st.button("📊 Generate Sample Data"):
            show_sample_dashboard()
        return
    
    st.success("✅ API Connected Successfully")
    
    # --- Sidebar Controls ---
    st.sidebar.header("📊 Dashboard Controls")
    
    # Stock selection
    selected_symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        STOCK_SYMBOLS,
        index=0,
        help="Choose from available stocks"
    )
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Prediction Model",
        ["arima", "lstm", "sentiment"],
        format_func=lambda x: x.upper(),
        help="Select the prediction model to use"
    )
    
    # Time horizon
    prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to predict into the future"
    )
    
    # --- Main Content ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🎯 {selected_model.upper()} Predictions for {selected_symbol}")
        
        if st.button("🚀 Generate Prediction", type="primary"):
            with st.spinner(f"Generating {selected_model.upper()} predictions..."):
                # Fetch prediction data
                prediction_data = make_api_request(
                    f"/predict/{selected_symbol}",
                    params={"model": selected_model, "days": prediction_days}
                )
                
                if prediction_data:
                    # Store in session state for persistence
                    st.session_state.prediction_data = prediction_data
                    st.session_state.selected_symbol = selected_symbol
                    st.session_state.selected_model = selected_model
        
        # Display stored prediction data
        if 'prediction_data' in st.session_state:
            prediction_data = st.session_state.prediction_data
            
            # Plot predictions
            fig = plot_price_prediction(
                prediction_data.get('historical', []),
                prediction_data.get('predictions', []),
                prediction_data['symbol'],
                prediction_data['model'].upper()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            if prediction_data.get('predictions'):
                latest_pred = prediction_data['predictions'][-1]
                current_price = prediction_data.get('current_price', 0)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}"
                    )
                with col_b:
                    change = latest_pred['predicted_price'] - current_price
                    change_pct = (change / current_price * 100) if current_price > 0 else 0
                    st.metric(
                        f"Predicted Price ({prediction_days}d)",
                        f"${latest_pred['predicted_price']:.2f}",
                        f"{change_pct:+.2f}%"
                    )
                with col_c:
                    confidence = prediction_data.get('confidence', 0.85)
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.1%}"
                    )
    
    with col2:
        st.subheader("📊 Model Performance")
        
        # Fetch and display metrics
        if st.button("📈 Load Metrics"):
            with st.spinner("Loading model metrics..."):
                metrics_data = make_api_request(f"/metrics/{selected_symbol}")
                
                if metrics_data:
                    st.session_state.metrics_data = metrics_data
        
        if 'metrics_data' in st.session_state:
            metrics_data = st.session_state.metrics_data
            
            for model, metrics in metrics_data.items():
                with st.expander(f"{model.upper()} Metrics", expanded=(model == selected_model)):
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("MAE", f"{metrics['mae']:.4f}")
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    with col_m2:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                        st.metric("R²", f"{metrics['r2_score']:.4f}")
        
        st.subheader("💭 Sentiment Analysis")
        
        # Fetch and display sentiment
        if st.button("🔍 Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                sentiment_data = make_api_request(f"/sentiment/{selected_symbol}")
                
                if sentiment_data:
                    st.session_state.sentiment_data = sentiment_data
        
        if 'sentiment_data' in st.session_state:
            sentiment_data = st.session_state.sentiment_data
            
            sentiment_score = sentiment_data.get('polarity', 0)
            sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "gray"
            
            st.markdown(f"""
            **Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment_data.get('sentiment_label', 'Neutral')}</span>
            
            **Polarity:** {sentiment_score:.3f}
            
            **Subjectivity:** {sentiment_data.get('subjectivity', 0):.3f}
            """, unsafe_allow_html=True)
            
            st.markdown("**Recent Headlines:**")
            for headline in sentiment_data.get('recent_headlines', []):
                st.write(f"• {headline}")

def show_sample_dashboard():
    """Show sample dashboard with dummy data when API is not available"""
    st.info("📊 Showing sample data (API not connected)")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    prices = 150 + np.cumsum(np.random.normal(0, 2, 30))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        name="Sample Stock Price",
        line=dict(color="blue", width=2)
    ))
    
    # Add sample predictions
    pred_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=7, freq='D')
    pred_prices = prices[-1] + np.cumsum(np.random.normal(0, 1, 7))
    
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=pred_prices,
        mode="lines+markers",
        name="Sample Prediction",
        line=dict(color="red", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title="Sample Stock Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", "$152.34")
    with col2:
        st.metric("7-Day Prediction", "$156.78", "+2.9%")
    with col3:
        st.metric("Model Confidence", "87.5%")

# --- Footer ---
def show_footer():
    """Show informational footer"""
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.info("**ARIMA Model**\nTime series forecasting using autoregressive patterns")
    
    with col_info2:
        st.info("**LSTM Model**\nDeep learning with Long Short-Term Memory networks")
    
    with col_info3:
        st.info("**Sentiment Model**\nNeural network enhanced with market sentiment")

if __name__ == "__main__":
    try:
        main()
        show_footer()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.markdown("**Troubleshooting steps:**")
        st.markdown("1. Make sure all required packages are installed")
        st.markdown("2. Check that the API server is running")
        st.markdown("3. Verify your internet connection")
        st.markdown("4. Try refreshing the page")