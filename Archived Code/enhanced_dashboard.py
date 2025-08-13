import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
import numpy as np
import time
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="JSE Stock Prediction Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive { color: #00b04f; }
    .negative { color: #ff4b4b; }
    .neutral { color: #808080; }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stAlert {
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data(ttl=60)  # Cache for 1 minute
def test_api_connection():
    """Test if API is available with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        return False, None
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_stocks():
    """Get available stocks with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}/stocks", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('stocks', [])
    except Exception as e:
        st.error(f"Failed to fetch stocks: {e}")
        return []

def make_api_request(endpoint, params=None, timeout=30):
    """Make API request with comprehensive error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        error_msg = "❌ Cannot connect to API. Make sure the API server is running on port 8000."
        return None, error_msg
    except requests.exceptions.Timeout:
        error_msg = "⏱️ API request timed out. Please try again."
        return None, error_msg
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_msg = f"🔍 Resource not found: {e.response.text}"
        elif e.response.status_code == 500:
            error_msg = f"🔥 Server error: {e.response.text}"
        else:
            error_msg = f"📡 HTTP error: {e.response.status_code} - {e.response.text}"
        return None, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"🔥 Request error: {str(e)}"
        return None, error_msg

def plot_price_prediction(historical_data, predictions, symbol, model_name, current_price=None):
    """Create enhanced interactive price prediction chart"""
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
            line=dict(color="#1f77b4", width=2),
            hovertemplate="<b>Historical</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ))
        
        # Add current price line if available
        if current_price and dates:
            fig.add_hline(
                y=current_price,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"Current: ${current_price:.2f}",
                annotation_position="right"
            )
    
    # Predictions
    if predictions:
        pred_dates = [item['date'] for item in predictions]
        pred_prices = [item['predicted_price'] for item in predictions]
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode="lines+markers",
            name=f"{model_name} Prediction",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
            marker=dict(size=6, color="#ff7f0e"),
            hovertemplate="<b>Predicted</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ))
        
        # Add confidence intervals if available
        if predictions and 'confidence_interval_low' in predictions[0]:
            low_prices = [item['confidence_interval_low'] for item in predictions]
            high_prices = [item['confidence_interval_high'] for item in predictions]
            
            fig.add_trace(go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=high_prices + low_prices[::-1],
                fill='toself',
                fillcolor='rgba(255,127,14,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            ))
    
    # Customize layout
    fig.update_layout(
        title=dict(
            text=f"{symbol.upper()} - {model_name} Price Prediction",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Price (USD)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        template="plotly_white",
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_metrics_chart(metrics_data):
    """Create metrics comparison chart"""
    if not metrics_data:
        return None
    
    models = list(metrics_data.keys())
    metrics = ['mae', 'mape', 'rmse', 'r2_score']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [metrics_data.get(model, {}).get(metric, 0) for model in models]
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=models,
            y=values,
            text=[f"{v:.3f}" if v else "N/A" for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Metric Values",
        barmode='group',
        template="plotly_white",
        height=400
    )
    
    return fig

# --- Main App ---
def main():
    # Header
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    
    with col_h1:
        st.title("📈 JSE Stock Prediction Dashboard")
        st.markdown("*Advanced ML-powered stock predictions using ARIMA, LSTM, and Sentiment Analysis*")
    
    with col_h2:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    
    with col_h3:
        # Real-time status indicator
        api_connected, health_data = test_api_connection()
        if api_connected:
            st.success("🟢 API Connected")
            if health_data:
                st.caption(f"Stocks: {health_data.get('available_stocks', 'N/A')}")
        else:
            st.error("🔴 API Disconnected")
    
    # Check API connection
    if not api_connected:
        st.error("🚨 **API Server Not Running**")
        
        with st.expander("🔧 Troubleshooting Guide", expanded=True):
            st.markdown("""
            **To start the API server:**
            1. Open a terminal in your project directory
            2. Ensure your actual data files exist:
               - `./data/stock_data.parquet` (your stock price data)
               - `./Data/joined_newspaper_data_v5.parquet` (your sentiment data)
               - `./Models/ARIMA/*.pkl` and `./Models/LSTM/*.pkl` (your trained models)
            3. Install dependencies: `pip install fastapi uvicorn polars`
            4. Run: `python integrated_api.py`
            5. Wait for "API is running" message
            6. Refresh this page
            
            **Common Issues:**
            - Missing data files: Check file paths in your DataManager configuration
            - Port 8000 already in use: Change port in both API and dashboard
            - Import errors: Ensure all Python packages are installed
            - Model loading errors: Verify your model files are in the correct format
            """)
        
        st.info("ℹ️ This system requires your actual data and trained models to function.")
        return
    
    # Get available stocks
    available_stocks = get_available_stocks()
    if not available_stocks:
        st.warning("⚠️ No stocks available. Check your data configuration.")
        return
    
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Stock selection
        selected_symbol = st.selectbox(
            "Select Stock Symbol",
            available_stocks,
            index=0,
            help="Choose from available stocks with trained models"
        )
        
        # Model selection
        selected_model = st.selectbox(
            "Prediction Model",
            ["arima", "lstm", "sentiment"],
            format_func=lambda x: {
                "arima": "📈 ARIMA (Time Series)",
                "lstm": "🧠 LSTM (Deep Learning)", 
                "sentiment": "💭 Sentiment Enhanced"
            }[x],
            help="Select the prediction model to use"
        )
        
        # Time horizon
        prediction_days = st.slider(
            "Prediction Days",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to predict into the future"
        )
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Generating {selected_model.upper()} predictions..."):
                prediction_data, error = make_api_request(
                    f"/predict/{selected_symbol}",
                    params={"model": selected_model, "days": prediction_days}
                )
                
                if prediction_data:
                    st.session_state.prediction_data = prediction_data
                    st.session_state.prediction_timestamp = datetime.now()
                    st.success("✅ Prediction generated!")
                else:
                    st.error(f"❌ {error}")
        
        if st.button("📊 Load Metrics", use_container_width=True):
            with st.spinner("Loading model performance metrics..."):
                metrics_data, error = make_api_request(f"/metrics/{selected_symbol}")
                
                if metrics_data:
                    st.session_state.metrics_data = metrics_data
                    st.success("✅ Metrics loaded!")
                else:
                    st.error(f"❌ {error}")
        
        if st.button("💭 Analyze Sentiment", use_container_width=True):
            with st.spinner("Analyzing market sentiment..."):
                sentiment_data, error = make_api_request(f"/sentiment/{selected_symbol}")
                
                if sentiment_data:
                    st.session_state.sentiment_data = sentiment_data
                    st.success("✅ Sentiment analyzed!")
                else:
                    st.error(f"❌ {error}")
    
    # --- Main Content ---
    
    # Display prediction results if available
    if 'prediction_data' in st.session_state:
        prediction_data = st.session_state.prediction_data
        
        # Prediction header
        col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
        
        with col_p1:
            st.subheader(f"🎯 {prediction_data['model'].upper()} Predictions for {prediction_data['symbol']}")
        
        with col_p2:
            if 'prediction_timestamp' in st.session_state:
                timestamp = st.session_state.prediction_timestamp
                st.caption(f"Generated: {timestamp.strftime('%H:%M:%S')}")
        
        with col_p3:
            confidence = prediction_data.get('confidence', 0.85)
            confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
            st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{confidence:.1%}</span>", 
                       unsafe_allow_html=True)
        
        # Price prediction chart
        fig = plot_price_prediction(
            prediction_data.get('historical', []),
            prediction_data.get('predictions', []),
            prediction_data['symbol'],
            prediction_data['model'].upper(),
            prediction_data.get('current_price')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics row
        if prediction_data.get('predictions'):
            latest_pred = prediction_data['predictions'][-1]
            current_price = prediction_data.get('current_price', 0)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    help="Most recent closing price"
                )
            
            with col_m2:
                pred_price = latest_pred['predicted_price']
                change = pred_price - current_price
                change_pct = (change / current_price * 100) if current_price > 0 else 0
                st.metric(
                    f"Predicted Price ({prediction_days}d)",
                    f"${pred_price:.2f}",
                    f"{change_pct:+.2f}%",
                    help=f"Predicted price in {prediction_days} days"
                )
            
            with col_m3:
                if 'confidence_interval_low' in latest_pred and 'confidence_interval_high' in latest_pred:
                    range_low = latest_pred['confidence_interval_low']
                    range_high = latest_pred['confidence_interval_high']
                    range_width = range_high - range_low
                    st.metric(
                        "Price Range (95% CI)",
                        f"${range_width:.2f}",
                        f"${range_low:.2f} - ${range_high:.2f}",
                        help="95% confidence interval for prediction"
                    )
                else:
                    st.metric("Price Range", "N/A", help="Confidence interval not available")
            
            with col_m4:
                # Calculate prediction trend
                if len(prediction_data['predictions']) > 1:
                    first_pred = prediction_data['predictions'][0]['predicted_price']
                    last_pred = prediction_data['predictions'][-1]['predicted_price']
                    trend = "📈 Bullish" if last_pred > first_pred else "📉 Bearish"
                    trend_pct = ((last_pred - first_pred) / first_pred * 100)
                    st.metric(
                        "Trend Direction",
                        trend,
                        f"{trend_pct:+.1f}%",
                        help="Overall prediction trend"
                    )
                else:
                    st.metric("Trend Direction", "N/A")
        
        # Prediction details table
        with st.expander("📋 Detailed Predictions", expanded=False):
            if prediction_data.get('predictions'):
                pred_df = pd.DataFrame(prediction_data['predictions'])
                pred_df['date'] = pd.to_datetime(pred_df['date'])
                pred_df = pred_df.set_index('date')
                
                # Format columns
                for col in ['predicted_price', 'confidence_interval_low', 'confidence_interval_high']:
                    if col in pred_df.columns:
                        pred_df[col] = pred_df[col].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(pred_df, use_container_width=True)
            else:
                st.info("No prediction data available")
    
    else:
        # No prediction data - show instructions
        st.info("👆 **Get Started:** Select a stock and model from the sidebar, then click 'Generate Prediction'")
    
    # Two-column layout for metrics and sentiment
    col_left, col_right = st.columns([1, 1])
    
    # --- Model Performance Metrics ---
    with col_left:
        st.subheader("📊 Model Performance")
        
        if 'metrics_data' in st.session_state:
            metrics_data = st.session_state.metrics_data
            
            # Create metrics comparison chart
            metrics_chart = create_metrics_chart(metrics_data)
            if metrics_chart:
                st.plotly_chart(metrics_chart, use_container_width=True)
            
            # Detailed metrics in expandable sections
            for model, metrics in metrics_data.items():
                with st.expander(f"{model.upper()} Detailed Metrics", expanded=(model == selected_model)):
                    if metrics:
                        col_met1, col_met2 = st.columns(2)
                        
                        with col_met1:
                            mae = metrics.get('mae')
                            rmse = metrics.get('rmse')
                            st.metric("MAE (Mean Absolute Error)", f"{mae:.4f}" if mae else "N/A")
                            st.metric("RMSE (Root Mean Square Error)", f"{rmse:.4f}" if rmse else "N/A")
                        
                        with col_met2:
                            mape = metrics.get('mape')
                            r2 = metrics.get('r2_score')
                            st.metric("MAPE (%)", f"{mape:.2f}%" if mape else "N/A")
                            st.metric("R² Score", f"{r2:.4f}" if r2 else "N/A")
                        
                        # Performance interpretation
                        if r2:
                            if r2 > 0.9:
                                st.success("🟢 Excellent model performance")
                            elif r2 > 0.7:
                                st.info("🟡 Good model performance")
                            else:
                                st.warning("🟠 Fair model performance")
                    else:
                        st.warning("No metrics available for this model")
        else:
            st.info("Click 'Load Metrics' in the sidebar to view model performance data")
    
    # --- Sentiment Analysis ---
    with col_right:
        st.subheader("💭 Market Sentiment")
        
        if 'sentiment_data' in st.session_state:
            sentiment_data = st.session_state.sentiment_data
            
            # Sentiment overview
            sentiment_score = sentiment_data.get('polarity', 0)
            sentiment_label = sentiment_data.get('sentiment_label', 'Neutral')
            subjectivity = sentiment_data.get('subjectivity', 0)
            
            # Color coding for sentiment
            if sentiment_score > 0.1:
                sentiment_color = "#00b04f"
                sentiment_emoji = "😊"
            elif sentiment_score < -0.1:
                sentiment_color = "#ff4b4b"
                sentiment_emoji = "😟"
            else:
                sentiment_color = "#808080"
                sentiment_emoji = "😐"
            
            # Sentiment metrics
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.metric(
                    "Overall Sentiment",
                    f"{sentiment_emoji} {sentiment_label}",
                    f"Polarity: {sentiment_score:.3f}",
                    help="Sentiment polarity ranges from -1 (very negative) to +1 (very positive)"
                )
            
            with col_s2:
                st.metric(
                    "Subjectivity",
                    f"{subjectivity:.3f}",
                    "Objectivity" if subjectivity < 0.5 else "Subjective",
                    help="Subjectivity ranges from 0 (objective) to 1 (subjective)"
                )
            
            # Sentiment gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sentiment_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Score"},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': sentiment_color},
                    'steps': [
                        {'range': [-1, -0.1], 'color': "lightcoral"},
                        {'range': [-0.1, 0.1], 'color': "lightgray"},
                        {'range': [0.1, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Recent headlines
            st.markdown("**📰 Recent Headlines:**")
            headlines = sentiment_data.get('recent_headlines', [])
            if headlines:
                for i, headline in enumerate(headlines, 1):
                    st.markdown(f"{i}. {headline}")
            else:
                st.info("No recent headlines available")
            
            # Last updated
            updated_at = sentiment_data.get('updated_at', '')
            if updated_at:
                try:
                    update_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    st.caption(f"Last updated: {update_time.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    st.caption(f"Last updated: {updated_at}")
        
        else:
            st.info("Click 'Analyze Sentiment' in the sidebar to view market sentiment analysis")
    
    # --- System Information Footer ---
    with st.expander("ℹ️ System Information", expanded=False):
        system_info, error = make_api_request("/system/info")
        
        if system_info:
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown("**📊 Data Statistics**")
                st.write(f"Available Stocks: {system_info.get('available_stocks', 'N/A')}")
                st.write(f"API Version: {system_info.get('api_version', 'N/A')}")
            
            with col_info2:
                st.markdown("**🤖 Model Information**")
                model_stats = system_info.get('model_statistics', {})
                st.write(f"ARIMA Models: {model_stats.get('arima', 'N/A')}")
                st.write(f"LSTM Models: {model_stats.get('lstm', 'N/A')}")
            
            with col_info3:
                st.markdown("**⚙️ Configuration**")
                st.write(f"Max Prediction Days: {system_info.get('max_prediction_days', 'N/A')}")
                supported_models = system_info.get('supported_models', [])
                st.write(f"Supported Models: {', '.join(supported_models).upper()}")
        else:
            st.warning("Could not fetch system information")

def show_demo_dashboard():
    """Show demo dashboard with sample data when API is not available"""
    st.info("📊 **Demo Mode** - Showing sample data (API not connected)")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    base_price = 150
    prices = []
    
    for i in range(30):
        base_price *= np.random.uniform(0.995, 1.005)
        prices.append(base_price)
    
    # Create sample chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        name="Sample Stock Price",
        line=dict(color="#1f77b4", width=2)
    ))
    
    # Sample predictions
    pred_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=7, freq='D')
    pred_prices = []
    current_price = prices[-1]
    
    for i in range(7):
        current_price *= np.random.uniform(0.99, 1.01)
        pred_prices.append(current_price)
    
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=pred_prices,
        mode="lines+markers",
        name="Sample Prediction",
        line=dict(color="#ff7f0e", width=3, dash="dash"),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Sample Stock Prediction (Demo)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", "$152.34")
    
    with col2:
        st.metric("7-Day Prediction", "$156.78", "+2.9%")
    
    with col3:
        st.metric("Model Confidence", "87.5%")
    
    with col4:
        st.metric("Sentiment", "😊 Positive", "Polarity: +0.25")

# --- App Footer ---
def show_app_footer():
    """Show informational footer with model descriptions"""
    st.markdown("---")
    st.markdown("### 🔬 **Model Information**")
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.info("""
        **📈 ARIMA Model**
        
        AutoRegressive Integrated Moving Average model for time series forecasting. 
        Best for: Stocks with clear seasonal patterns and trends.
        
        ✅ Fast predictions  
        ✅ Interpretable results  
        ❌ Limited to historical patterns
        """)
    
    with col_model2:
        st.info("""
        **🧠 LSTM Model**
        
        Long Short-Term Memory neural network for complex pattern recognition.
        Best for: Stocks with non-linear relationships and complex behaviors.
        
        ✅ Captures complex patterns  
        ✅ Handles multiple variables  
        ❌ Requires more training data
        """)
    

if __name__ == "__main__":
    try:
        main()
        show_app_footer()
    except Exception as e:
        st.error(f"❌ **Application Error:** {str(e)}")
                
        if st.button("🔄 Restart Dashboard"):
            st.rerun()