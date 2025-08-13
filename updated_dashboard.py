import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

# Page Configuration
st.set_page_config(
    page_title="JSE Stock Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# UPDATED: Configuration for deployment with your actual URLs
API_BASE_URL = "https://stocks-t52l.onrender.com"

# Override with environment variable if set (for local development)
if os.environ.get("API_BASE_URL"):
    API_BASE_URL = os.environ.get("API_BASE_URL")

# Streamlit Cloud secrets override (if you set it in secrets)
try:
    if hasattr(st, 'secrets') and 'API_BASE_URL' in st.secrets:
        API_BASE_URL = st.secrets['API_BASE_URL']
except:
    pass  # Ignore if secrets not available

# Custom CSS
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
    .prediction-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# UTILITY FUNCTIONS
# ================================

SIMULATION_DATE = "2024-10-01"

def get_current_date():
    """Get current date or simulation date"""
    if SIMULATION_DATE:
        return datetime.strptime(SIMULATION_DATE, '%Y-%m-%d').date()
    else:
        return datetime.now().date()

@st.cache_data(ttl=60)
def test_api_connection():
    """Test API connection with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=300)
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
    """Make API request with error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Make sure the API server is running."
    except requests.exceptions.Timeout:
        return None, "⏱️ API request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return None, f"📡 HTTP error: {e.response.status_code}"
    except Exception as e:
        return None, f"🔥 Request error: {str(e)}"

def get_next_business_days(start_date, num_days):
    """Generate next N business days"""
    business_days = []
    current = start_date
    
    while len(business_days) < num_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            business_days.append(current)
    
    return business_days

def create_prediction_chart(historical_data, lstm_data=None, arima_data=None, symbol="", current_price=None):
    """Create unified prediction chart that can show both models"""
    fig = go.Figure()
    
    # Historical prices (always show)
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
        
        # Current price line
        if current_price and dates:
            fig.add_hline(
                y=current_price,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"Current: ${current_price:.2f}",
                annotation_position="right"
            )
    
    # LSTM predictions (orange)
    if lstm_data and lstm_data.get('predictions'):
        lstm_preds = lstm_data['predictions']
        pred_dates = [item['date'] for item in lstm_preds]
        pred_prices = [item['predicted_price'] for item in lstm_preds]
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode="lines+markers",
            name="LSTM Prediction",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
            marker=dict(size=8, color="#ff7f0e", symbol="diamond"),
            hovertemplate="<b>LSTM Predicted</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ))
        
        # LSTM confidence intervals
        if lstm_preds and 'confidence_low' in lstm_preds[0]:
            low_prices = [item['confidence_low'] for item in lstm_preds]
            high_prices = [item['confidence_high'] for item in lstm_preds]
            
            fig.add_trace(go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=high_prices + low_prices[::-1],
                fill='toself',
                fillcolor='rgba(255,127,14,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='LSTM 95% Confidence',
                showlegend=True,
                hoverinfo='skip'
            ))
    
    # ARIMA predictions (green)
    if arima_data and arima_data.get('predictions'):
        arima_preds = arima_data['predictions']
        pred_dates = [item['date'] for item in arima_preds]
        pred_prices = [item['predicted_price'] for item in arima_preds]
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode="lines+markers",
            name="ARIMA Prediction",
            line=dict(color="#2ca02c", width=3, dash="dash"),
            marker=dict(size=8, color="#2ca02c", symbol="circle"),
            hovertemplate="<b>ARIMA Predicted</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ))
        
        # ARIMA confidence intervals
        if arima_preds and 'confidence_low' in arima_preds[0]:
            low_prices = [item['confidence_low'] for item in arima_preds]
            high_prices = [item['confidence_high'] for item in arima_preds]
            
            fig.add_trace(go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=high_prices + low_prices[::-1],
                fill='toself',
                fillcolor='rgba(44,160,44,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='ARIMA 95% Confidence',
                showlegend=True,
                hoverinfo='skip'
            ))
    
    # Chart title based on what's being shown
    if lstm_data and arima_data:
        title_text = f"{symbol.upper()} - Model Comparison (LSTM vs ARIMA)"
    elif lstm_data:
        title_text = f"{symbol.upper()} - LSTM Price Prediction"
    elif arima_data:
        title_text = f"{symbol.upper()} - ARIMA Price Prediction"
    else:
        title_text = f"{symbol.upper()} - Historical Price Data"
    
    # Layout
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            font=dict(size=20)
        ),
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Price (JMD)", showgrid=True),
        hovermode='x unified',
        template="plotly_white",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def display_wordcloud_image(wordcloud_b64):
    """Display base64 encoded wordcloud image"""
    if wordcloud_b64:
        try:
            image_data = base64.b64decode(wordcloud_b64)
            image = Image.open(BytesIO(image_data))
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying word cloud: {e}")

# ================================
# MAIN APPLICATION
# ================================

def main():
    # UPDATED: Show API connection info
    st.info(f"🔗 **API Connection:** `{API_BASE_URL}`")
    
    # Header
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    
    with col_h1:
        st.title("📈 JSE Stock Prediction Dashboard")
        st.markdown("*Advanced PyTorch LSTM & ARIMA predictions with sentiment analysis*")
    
    with col_h2:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    
    with col_h3:
        # API status
        api_connected, health_data = test_api_connection()
        if api_connected:
            st.success("🟢 API Connected")
            if health_data:
                st.caption(f"Stocks: {health_data.get('available_stocks', 'N/A')}")
        else:
            st.error("🔴 API Disconnected")
    
    # Check API connection
    if not api_connected:
        st.error("🚨 **API Server Not Available**")
        
        with st.expander("🔧 Connection Information", expanded=True):
            st.markdown(f"""
            **Current API URL:** `{API_BASE_URL}`
            
            **Troubleshooting:**
            1. Verify the API server is deployed and running
            2. Check if the API URL is correct
            3. Ensure your network can reach the API endpoint
            4. Try refreshing this page
            
            **For Local Development:**
            - Make sure your API server is running locally
            - Check that it's accessible at `http://localhost:8000`
            """)

        return
    
    # Get available stocks
    available_stocks = get_available_stocks()
    if not available_stocks:
        st.warning("⚠️ No stocks available. Check your model files.")
        return
    
    # ================================
    # SIDEBAR CONTROLS - UNCHANGED
    # ================================
    
    with st.sidebar:
        st.header("📊 Prediction Controls")
        
        # Stock selection
        selected_symbol = st.selectbox(
            "Select Stock Symbol",
            available_stocks,
            index=0,
            help="Choose from stocks with trained models"
        )
        
        # Model selection
        selected_model = st.selectbox(
            "Model Type",
            ["lstm", "arima"],
            format_func=lambda x: {
                "lstm": "🧠 LSTM (Deep Learning)",
                "arima": "📈 ARIMA (Time Series)"
            }[x],
            help="Select the prediction model to use"
        )
        
        st.markdown("---")
        
        # Date selection method
        date_method = st.radio(
            "Prediction Method",
            ["Auto Generate", "Custom Dates"],
            help="Choose how to select prediction dates"
        )
        
        if date_method == "Auto Generate":
            # Auto-generate business days
            prediction_days = st.slider(
                "Number of Days",
                min_value=1,
                max_value=30,
                value=7,
                help="Generate predictions for next N business days"
            )
            prediction_dates = None
        else:
            # Custom date selection
            st.subheader("Custom Dates")
            
            # Date inputs
            start_date = get_current_date()
            max_date = datetime(2024, 12, 23).date()
            
            start_date = st.date_input(
                "Start Date",
                value=start_date + timedelta(days=1),
                min_value=start_date + timedelta(days=1),
                max_value=max_date
            )
            
            end_max_date = min(max_date, start_date + timedelta(days=30))

            end_date = st.date_input(
                "End Date",
                value=start_date + timedelta(days=6),
                min_value=start_date,
                max_value=end_max_date
            )
            
            # Generate date range
            if start_date <= end_date:
                date_range = []
                current = start_date
                while current <= end_date:
                    if current.weekday() < 5:  # Business days only
                        date_range.append(current.strftime('%Y-%m-%d'))
                    current += timedelta(days=1)
                
                if date_range:
                    prediction_dates = ",".join(date_range)
                    st.success(f"Selected {len(date_range)} business days")
                else:
                    prediction_dates = None
                    st.warning("No business days in selected range")
            else:
                prediction_dates = None
        
        st.markdown("---")
        
        # Action buttons
        if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Generating {selected_model.upper()} predictions..."):
                params = {"model": selected_model}
                if date_method == "Auto Generate":
                    params["days"] = prediction_days
                else:
                    if prediction_dates:
                        params["dates"] = prediction_dates
                    else:
                        st.error("Please select valid dates")
                        st.stop()
                
                prediction_data, error = make_api_request(
                    f"/predict/{selected_symbol}",
                    params=params
                )
                
                if prediction_data:
                    # Store single model prediction
                    if selected_model == "lstm":
                        st.session_state.lstm_data = prediction_data
                        if 'arima_data' in st.session_state:
                            del st.session_state.arima_data  # Clear other model
                    else:
                        st.session_state.arima_data = prediction_data
                        if 'lstm_data' in st.session_state:
                            del st.session_state.lstm_data  # Clear other model
                    
                    st.session_state.prediction_timestamp = datetime.now()
                    st.success("✅ Prediction generated!")
                else:
                    st.error(f"❌ {error}")

        # Compare Models button
        if st.button("🔀 Compare Models", use_container_width=True):
            with st.spinner("Generating both LSTM and ARIMA predictions..."):
                params_base = {}
                if date_method == "Auto Generate":
                    params_base["days"] = prediction_days
                else:
                    if prediction_dates:
                        params_base["dates"] = prediction_dates
                    else:
                        st.error("Please select valid dates")
                        st.stop()
                
                # Get LSTM predictions
                lstm_params = {**params_base, "model": "lstm"}
                lstm_data, lstm_error = make_api_request(
                    f"/predict/{selected_symbol}",
                    params=lstm_params
                )
                
                # Get ARIMA predictions  
                arima_params = {**params_base, "model": "arima"}
                arima_data, arima_error = make_api_request(
                    f"/predict/{selected_symbol}",
                    params=arima_params
                )
                
                if lstm_data and arima_data:
                    # Store both models for comparison
                    st.session_state.lstm_data = lstm_data
                    st.session_state.arima_data = arima_data
                    st.session_state.comparison_mode = True
                    st.session_state.prediction_timestamp = datetime.now()
                    st.success("✅ Both models generated!")
                else:
                    st.error(f"❌ Failed: LSTM={lstm_error}, ARIMA={arima_error}")

        if st.button("💭 Analyze Sentiment", use_container_width=True):
            with st.spinner("Analyzing sentiment..."):
                sentiment_data, error = make_api_request(f"/sentiment/{selected_symbol}")
                
                if sentiment_data:
                    st.session_state.sentiment_data = sentiment_data
                    st.success("✅ Sentiment analyzed!")
                else:
                    st.error(f"❌ {error}")
        
        if st.button("📊 Load Metrics", use_container_width=True):
            with st.spinner("Loading model metrics..."):
                metrics_data, error = make_api_request(f"/metrics/{selected_symbol}")
                
                if metrics_data:
                    st.session_state.metrics_data = metrics_data
                    st.success("✅ Metrics loaded!")
                else:
                    st.error(f"❌ {error}")
    
    # ================================
    # MAIN CONTENT - UNIFIED CHART
    # ================================
    
    # Check what prediction data we have
    lstm_data = st.session_state.get('lstm_data')
    arima_data = st.session_state.get('arima_data')
    comparison_mode = st.session_state.get('comparison_mode', False)
    
    if lstm_data or arima_data:
        # Determine chart header
        if comparison_mode and lstm_data and arima_data:
            st.subheader(f"🔀 Model Comparison for {selected_symbol}")
        elif lstm_data:
            st.subheader(f"🧠 LSTM Predictions for {selected_symbol}")
        elif arima_data:
            st.subheader(f"📈 ARIMA Predictions for {selected_symbol}")
        
        # Get historical data (same for both models)
        historical_data = (lstm_data or arima_data).get('historical', [])
        current_price = (lstm_data or arima_data).get('current_price')
        
        # Create unified chart
        fig = create_prediction_chart(
            historical_data=historical_data,
            lstm_data=lstm_data if comparison_mode or not arima_data else None,
            arima_data=arima_data if comparison_mode or not lstm_data else None,
            symbol=selected_symbol,
            current_price=current_price
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance summary
        if comparison_mode and lstm_data and arima_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
            
            with col2:
                lstm_final = lstm_data['predictions'][-1]['predicted_price'] if lstm_data.get('predictions') else 0
                lstm_change = ((lstm_final - current_price) / current_price * 100) if current_price else 0
                st.metric("LSTM Final", f"${lstm_final:.2f}", f"{lstm_change:+.1f}%")
            
            with col3:
                arima_final = arima_data['predictions'][-1]['predicted_price'] if arima_data.get('predictions') else 0
                arima_change = ((arima_final - current_price) / current_price * 100) if current_price else 0
                st.metric("ARIMA Final", f"${arima_final:.2f}", f"{arima_change:+.1f}%")
            
            with col4:
                if lstm_data.get('predictions') and arima_data.get('predictions'):
                    avg_diff = np.mean([
                        abs(l['predicted_price'] - a['predicted_price']) 
                        for l, a in zip(lstm_data['predictions'], arima_data['predictions'])
                    ])
                    st.metric("Avg Difference", f"${avg_diff:.2f}")
        
        # Single model metrics
        elif lstm_data and not comparison_mode:
            predictions = lstm_data.get('predictions', [])
            if predictions:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
                
                with col2:
                    final_pred = predictions[-1]['predicted_price']
                    change = ((final_pred - current_price) / current_price * 100) if current_price else 0
                    st.metric("Final Prediction", f"${final_pred:.2f}", f"{change:+.1f}%")
                
                with col3:
                    confidence = lstm_data.get('confidence', 0)
                    st.metric("Model Confidence", f"{confidence:.1%}")
                
                with col4:
                    model_metrics = lstm_data.get('model_metrics', {})
                    mae = model_metrics.get('mae', 0)
                    st.metric("Model MAE", f"{mae:.3f}%")
        
        elif arima_data and not comparison_mode:
            predictions = arima_data.get('predictions', [])
            if predictions:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
                
                with col2:
                    final_pred = predictions[-1]['predicted_price']
                    change = ((final_pred - current_price) / current_price * 100) if current_price else 0
                    st.metric("Final Prediction", f"${final_pred:.2f}", f"{change:+.1f}%")
                
                with col3:
                    confidence = arima_data.get('confidence', 0)
                    st.metric("Model Confidence", f"{confidence:.1%}")
                
                with col4:
                    model_metrics = arima_data.get('model_metrics', {})
                    mae = model_metrics.get('mae', 0)
                    st.metric("Model MAE", f"{mae:.3f}%")
        
        # Model Comparison Analysis (Dropdown)
        if comparison_mode and lstm_data and arima_data:
            with st.expander("🔀 Model Comparison Analysis", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🧠 LSTM Performance**")
                    lstm_confidence = lstm_data.get('confidence', 0)
                    st.metric("Model Accuracy", f"{lstm_confidence:.1%}")
                    
                    lstm_metrics = lstm_data.get('model_metrics', {})
                    st.metric("MAE", f"{lstm_metrics.get('mae', 0):.4f}")
                    st.metric("RMSE", f"{lstm_metrics.get('rmse', 0):.4f}")
                
                with col2:
                    st.markdown("**📈 ARIMA Performance**")
                    arima_confidence = arima_data.get('confidence', 0)
                    st.metric("Model Accuracy", f"{arima_confidence:.1%}")
                    
                    arima_metrics = arima_data.get('model_metrics', {})
                    st.metric("MAE", f"{arima_metrics.get('mae', 0):.4f}")
                    st.metric("RMSE", f"{arima_metrics.get('rmse', 0):.4f}")
                
                # Agreement analysis
                st.markdown("**📊 Model Agreement Analysis**")
                if lstm_data.get('predictions') and arima_data.get('predictions'):
                    lstm_preds = [p['predicted_price'] for p in lstm_data['predictions']]
                    arima_preds = [p['predicted_price'] for p in arima_data['predictions']]
                    
                    differences = [abs(l - a) for l, a in zip(lstm_preds, arima_preds)]
                    avg_diff = np.mean(differences)
                    max_diff = max(differences)
                    
                    col_a1, col_a2, col_a3 = st.columns(3)
                    with col_a1:
                        st.metric("Average Difference", f"${avg_diff:.2f}")
                    with col_a2:
                        st.metric("Maximum Difference", f"${max_diff:.2f}")
                    with col_a3:
                        agreement_pct = (1 - avg_diff / current_price) * 100 if current_price else 0
                        st.metric("Agreement %", f"{agreement_pct:.1f}%")
                    
                    if avg_diff < 5:
                        st.success("🤝 Models agree closely")
                    elif avg_diff < 15:
                        st.warning("⚠️ Models show moderate disagreement") 
                    else:
                        st.error("🚨 Models show significant disagreement")
        
        # Detailed Predictions (Dropdown)
        active_data = None
        active_model = ""
        
        if comparison_mode and lstm_data and arima_data:
            # Show both models' predictions in comparison mode
            with st.expander("📋 Detailed Predictions (Both Models)", expanded=False):
                if lstm_data.get('predictions') and arima_data.get('predictions'):
                    # Create combined dataframe
                    comparison_df = pd.DataFrame({
                        'Date': [p['date'] for p in lstm_data['predictions']],
                        'LSTM_Prediction': [f"${p['predicted_price']:.2f}" for p in lstm_data['predictions']],
                        'ARIMA_Prediction': [f"${p['predicted_price']:.2f}" for p in arima_data['predictions']],
                        'Difference': [f"${abs(l['predicted_price'] - a['predicted_price']):.2f}" 
                                     for l, a in zip(lstm_data['predictions'], arima_data['predictions'])]
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Export option
                    csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download Comparison CSV", 
                        csv, 
                        f"{selected_symbol}_comparison.csv",
                        mime="text/csv"
                    )
        
        else:
            # Show single model predictions
            if lstm_data and not comparison_mode:
                active_data = lstm_data
                active_model = "LSTM"
            elif arima_data and not comparison_mode:
                active_data = arima_data
                active_model = "ARIMA"
            
            if active_data:
                with st.expander(f"📋 Detailed {active_model} Predictions", expanded=False):
                    if active_data.get('predictions'):
                        pred_df = pd.DataFrame(active_data['predictions'])
                        pred_df['date'] = pd.to_datetime(pred_df['date'])
                        
                        # Format price columns
                        for col in ['predicted_price', 'confidence_low', 'confidence_high']:
                            if col in pred_df.columns:
                                pred_df[col] = pred_df[col].apply(lambda x: f"${x:.2f}")
                        
                        st.dataframe(pred_df, use_container_width=True)
    
    else:
        st.info("👆 **Get Started:** Select a stock and prediction method from the sidebar, then click 'Generate Prediction' or 'Compare Models'")
    
    # ================================
    # TWO-COLUMN LAYOUT (Metrics & Sentiment)
    # ================================
    
    col_left, col_right = st.columns([1, 1])
    
    # Model Performance Metrics
    with col_left:
        st.subheader("📊 Model Performance")
        
        if 'metrics_data' in st.session_state:
            metrics_data = st.session_state.metrics_data
            available_models = metrics_data.get('available_models', [])
            
            if available_models:
                # Show metrics for each available model
                for model_type in available_models:
                    model_metrics = metrics_data.get(model_type, {})
                    
                    if model_metrics:
                        with st.expander(f"{model_type.upper()} Model Metrics", expanded=(model_type == selected_model)):
                            col_met1, col_met2 = st.columns(2)
                            
                            with col_met1:
                                mae = model_metrics.get('mae', 0)
                                rmse = model_metrics.get('rmse', 0)
                                st.metric("MAE", f"{mae:.4f}")
                                st.metric("RMSE", f"{rmse:.4f}")
                            
                            with col_met2:
                                mape = model_metrics.get('mape', 0)
                                accuracy = model_metrics.get('accuracy', 0)
                                st.metric("MAPE", f"{mape:.2f}%")
                                st.metric("Accuracy", f"{accuracy:.1%}")
                            
                            # Performance assessment
                            if accuracy > 0.7:
                                st.success("🟢 Excellent model performance")
                            elif accuracy > 0.6:
                                st.info("🟡 Good model performance")
                            elif accuracy > 0.5:
                                st.warning("🟠 Fair model performance")
                            else:
                                st.error("🔴 Poor model performance")
            else:
                st.warning("No model metrics available")
        else:
            st.info("Click 'Load Metrics' to view model performance")
    
    # Sentiment Analysis
    with col_right:
        st.subheader("💭 Market Sentiment")
        
        if 'sentiment_data' in st.session_state:
            sentiment_data = st.session_state.sentiment_data
            
            # Sentiment overview
            company_sentiment = sentiment_data.get('company_sentiment', 0)
            market_sentiment = sentiment_data.get('market_sentiment', 0)
            sentiment_label = sentiment_data.get('sentiment_label', 'Neutral')
            
            # Sentiment metrics
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                if company_sentiment > 0.1:
                    emoji = "😊"
                    color = "#00b04f"
                elif company_sentiment < -0.1:
                    emoji = "😟"
                    color = "#ff4b4b"
                else:
                    emoji = "😐"
                    color = "#808080"
                
                st.markdown(f"**Company Sentiment**")
                st.markdown(f"<h2 style='color:{color}'>{emoji} {sentiment_label}</h2>", unsafe_allow_html=True)
                st.caption(f"Score: {company_sentiment:.3f}")
            
            with col_s2:
                st.metric(
                    "Market Sentiment",
                    f"{market_sentiment:.3f}",
                    help="Overall market sentiment score"
                )
            
            # Recent headlines
            st.markdown("**📰 Recent Headlines:**")
            headlines = sentiment_data.get('recent_headlines', [])
            if headlines:
                for i, headline in enumerate(headlines, 1):
                    date = headline.get('date', 'Unknown')
                    title = headline.get('title', 'No title')
                    sentiment = headline.get('sentiment', 0)
                    
                    sentiment_color = "#00b04f" if sentiment > 0.1 else "#ff4b4b" if sentiment < -0.1 else "#808080"
                    st.markdown(f"**{i}.** {title}")
                    st.caption(f"Date: {date} | Sentiment: {sentiment:.3f}")
            else:
                st.info("No recent headlines available")
            
            # Word cloud
            st.markdown("**🔤 Keywords from Latest Article:**")
            wordcloud_b64 = sentiment_data.get('wordcloud_image')
            if wordcloud_b64:
                display_wordcloud_image(wordcloud_b64)
            else:
                st.info("No word cloud available")
        
        else:
            st.info("Click 'Analyze Sentiment' to view sentiment analysis")
    
    # System information footer
    with st.expander("ℹ️ System Information", expanded=False):
        system_info, error = make_api_request("/system/info")
        
        if system_info:
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("**📊 System Stats**")
                st.write(f"Available Stocks: {system_info.get('available_stocks', 'N/A')}")
                st.write(f"API Version: {system_info.get('api_version', 'N/A')}")
                model_types = system_info.get('model_types', [])
                if model_types:
                    st.write(f"Model Types: {', '.join(model_types)}")
            
            with col_info2:
                st.markdown("**🎯 Configuration**")
                st.write(f"Max Prediction Days: {system_info.get('max_prediction_days', 'N/A')}")
                st.write(f"Features: {system_info.get('features', 'N/A')}")
                sample_stocks = system_info.get('sample_stocks', [])
                if sample_stocks:
                    st.write(f"Sample Stocks: {', '.join(sample_stocks[:5])}")

def show_footer():
    """Show informational footer"""
    st.markdown("---")
    st.markdown("### 🔬 **Model Information**")
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.info("""
        **🧠 PyTorch LSTM with Attention**
        
        Advanced neural network combining:
        - Bidirectional LSTM layers for temporal patterns
        - Multi-head attention mechanism
        - Sentiment-enhanced features from news
        - Technical indicators (RSI, MACD, etc.)
        
        ✅ Captures complex non-linear relationships  
        ✅ Handles multiple time-dependent variables  
        ✅ Incorporates market sentiment signals  
        ⚠ Requires sufficient historical data
        """)
    
    with col_model2:
        st.info("""
        **📈 ARIMA Time Series Model**
        
        AutoRegressive Integrated Moving Average:
        - Statistical time series forecasting
        - Captures trends and seasonal patterns
        - Fast and interpretable predictions
        - Works well with stationary data
        
        ✅ Fast predictions and training  
        ✅ Interpretable statistical results  
        ✅ Good for short-term forecasts  
        ⚠ Limited to historical price patterns
        """)

if __name__ == "__main__":
    try:
        main()
        show_footer()
    except Exception as e:
        st.error(f"❌ **Application Error:** {str(e)}")
        if st.button("🔄 Restart Dashboard"):
            st.rerun()