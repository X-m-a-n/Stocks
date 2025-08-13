# 📈 Advanced Stock Prediction Dashboard

A comprehensive stock price prediction system featuring ARIMA, LSTM, and sentiment-enhanced neural network models with a modern Streamlit frontend and FastAPI backend.

## 🚀 Features

- **Multiple Prediction Models**: ARIMA, LSTM, and Sentiment-enhanced Neural Networks
- **Interactive Dashboard**: Modern Streamlit interface with real-time updates
- **RESTful API**: FastAPI backend with automatic documentation
- **99 Stock Support**: Pre-configured for top market stocks
- **Sentiment Analysis**: News sentiment integration for enhanced predictions
- **Model Metrics**: MAE, MAPE, RMSE, and R² scores for model evaluation
- **Caching System**: Intelligent caching for improved performance
- **Batch Predictions**: Support for multiple stock predictions

## 🏗️ Project Structure

```
stock-prediction-dashboard/
├── app.py                 # Streamlit frontend
├── api.py                 # FastAPI backend
├── data.py                # Data management and model loading
├── requirements.txt       # Python dependencies
├── models/                # Model storage
│   ├── arima/            # ARIMA model files (.pkl)
│   ├── lstm/             # LSTM model files (.pkl)
│   └── sentiment/        # Sentiment model files (.pkl)
├── data/                 # Data storage
│   └── cache/            # Cached data files
└── README.md             # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd stock-prediction-dashboard
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create directory structure**
```bash
mkdir -p models/{arima,lstm,sentiment}
mkdir -p data/cache
```

## 🚀 Running the Application

### Option 1: Full Stack (Recommended)

**Terminal 1 - Start FastAPI Backend:**
```bash
python api.py
```
The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

**Terminal 2 - Start Streamlit Frontend:**
```bash
streamlit run app.py
```
The dashboard will be available at `http://localhost:8501`

### Option 2: Development Mode

**Backend with auto-reload:**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Frontend with auto-reload:**
```bash
streamlit run app.py --server.runOnSave true
```

## 📊 API Endpoints

### Core Prediction Endpoints

- `GET /predict/{symbol}` - Generate predictions for a stock
  - Parameters: `model` (arima/lstm/sentiment), `days` (1-30)
- `GET /sentiment/{symbol}` - Get sentiment analysis
- `GET /metrics/{symbol}` - Get model evaluation metrics
- `POST /predict/batch` - Batch prediction for multiple stocks

### Management Endpoints

- `GET /health` - Health check
- `GET /stocks` - List available stocks
- `POST /models/{symbol}/retrain` - Retrain model
- `DELETE /cache/clear` - Clear cache

### Example API Usage

```python
import requests

# Get prediction
response = requests.get(
    "http://localhost:8000/predict/AAPL",
    params={"model": "lstm", "days": 7}
)
prediction = response.json()

# Get sentiment
response = requests.get("http://localhost:8000/sentiment/AAPL")
sentiment = response.json()
```

## 🎯 Dashboard Usage

### Stock Selection
1. Choose from 99 pre-configured stocks in the sidebar
2. Select prediction model (ARIMA, LSTM, or Sentiment)
3. Set prediction horizon (1-30 days)

### Generating Predictions
1. Click "🚀 Generate Prediction" button
2. View interactive Plotly charts with historical and predicted prices
3. Monitor model metrics and confidence scores

### Sentiment Analysis
- View real-time sentiment scores
- See recent news headlines affecting the stock
- Understand sentiment impact on predictions

## 🧠 Model Details

### ARIMA Model
- **Purpose**: Time series forecasting using autoregressive patterns
- **Best for**: Stable, trending stocks with clear patterns
- **Output**: Conservative predictions with statistical confidence intervals

### LSTM Model
- **Purpose**: Deep learning approach for complex pattern recognition
- **Best for**: Volatile stocks with non-linear patterns
- **Output**: Adaptive predictions with wider confidence intervals

### Sentiment Model
- **Purpose**: Neural network enhanced with market sentiment
- **Best for**: Stocks heavily influenced by news and market sentiment
- **Output**: Sentiment-adjusted predictions with dynamic confidence

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom paths
export MODEL_PATH="./custom_models"
export DATA_PATH="./custom_data"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

### Customizing Stock List
Edit the `STOCK_SYMBOLS` list in `app.py` to add/remove stocks:

```python
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL",  # Add your stocks here
    # ... rest of symbols
]
```

## 📈 Adding New Models

### 1. Create Model File
```python
# models/arima/NEWSYMBOL.pkl
new_model = {
    'model_type': 'arima',
    'symbol': 'NEWSYMBOL',
    'coefficients': your_coefficients,
    'metrics': {
        'mae': your_mae,
        'mape': your_mape,
        'rmse': your_rmse,
        'r2_score': your_r2
    }
}

import pickle
with open('models/arima/NEWSYMBOL.pkl', 'wb') as f:
    pickle.dump(new_model, f)
```

### 2. Update Available Stocks
The system automatically scans for available models, or you can add to the stock list manually.

## 🔍 Monitoring & Debugging

### Health Check
```bash
curl http://localhost:8000/health
```

### Logs
- API logs: Console output from `python api.py`
- Streamlit logs: Console output from `streamlit run app.py`

### Common Issues

**Port conflicts:**
```bash
# Change ports in configuration
streamlit run app.py --server.port 8502
uvicorn api:app --port 8001
```

**Model loading errors:**
- Check model file permissions
- Verify pickle file integrity
- Review logs for specific error messages

## 🚀 Deployment

### Docker Deployment (Recommended)

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000 8501

# Start both services
CMD ["python", "api.py"] & ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### Production Considerations

1. **Use a process manager** (PM2, supervisord)
2. **Set up reverse proxy** (nginx)
3. **Enable HTTPS** for security
4. **Configure logging** (structured logs)
5. **Set up monitoring** (health checks, metrics)
6. **Use production database** (PostgreSQL)
7. **Implement rate limiting**

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

For issues and questions:
- Create GitHub issue
- Check API documentation at `/docs`
- Review logs for debugging information