import os
import sys
import time
import subprocess
import threading
from pathlib import Path
import requests

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🚀 JSE STOCK PREDICTION SYSTEM LAUNCHER")
    print("=" * 60)
    print("📈 Advanced LSTM & ARIMA Stock Predictions")
    print("💭 Market Sentiment Analysis")
    print("🎯 Interactive Dashboard")
    print("=" * 60)

def check_requirements():
    """Check if required files exist"""
    print("🔍 Checking system requirements...")
    
    required_files = [
        "updated_data_manager.py",
        "updated_api.py", 
        "updated_dashboard.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check for data files
    data_files = [
        "stocks_df_Sample.csv",
        "clean_stock_data.parquet",
        "Sentiment_data_sample.csv"
    ]
    
    data_found = False
    for file in data_files:
        if Path(file).exists():
            print(f"✅ Found data file: {file}")
            data_found = True
            break
    
    if not data_found:
        print("⚠️  No stock data files found. Place one of these files in the directory:")
        for file in data_files:
            print(f"   - {file}")
        print("📝 The system will attempt to use demo data if no real data is available.")
    
    # Check for model files
    models_dir = Path("models")
    if models_dir.exists():
        arima_models = list(models_dir.glob("ARIMA/*.pkl"))
        lstm_models = list(models_dir.glob("LSTM/*.pth")) + list(models_dir.glob("*.pth"))
        
        print(f"🤖 Found {len(arima_models)} ARIMA models")
        print(f"🧠 Found {len(lstm_models)} LSTM models")
        
        if len(arima_models) == 0 and len(lstm_models) == 0:
            print("⚠️  No trained models found. The system will use demo mode.")
    else:
        print("⚠️  No models directory found. Creating models directory...")
        models_dir.mkdir(exist_ok=True)
    
    print("✅ System check completed")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("📦 Checking Python dependencies...")
    
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "polars", "pandas", 
        "numpy", "torch", "plotly", "requests", "Pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing Python packages: {missing_packages}")
        print("💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies found")
    return True

def start_api_server():
    """Start the FastAPI server in a separate process"""
    print("🚀 Starting API server...")
    
    try:
        # Start API server
        process = subprocess.Popen([
            sys.executable, "updated_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server started successfully")
                print("📡 API available at: http://localhost:8000")
                print("📖 API documentation at: http://localhost:8000/docs")
                return process
            else:
                print("❌ API server started but health check failed")
                return None
        except requests.exceptions.RequestException:
            print("❌ API server failed to start or is not responding")
            return None
            
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("🎨 Starting dashboard...")
    
    try:
        # Start Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "updated_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check system requirements
    # if not check_requirements():
    #    input("Press Enter to exit...")
    #    return
    
    # Check dependencies
    # if not check_dependencies():
    #    input("Press Enter to exit...")
    #    return
    
    print("\n" + "=" * 60)
    print("🏁 STARTING SYSTEM COMPONENTS")
    print("=" * 60)
    
    # Start API server
    api_process = start_api_server()
    
    if api_process is None:
        print("❌ Failed to start API server. Check the console for errors.")
        input("Press Enter to exit...")
        return
    
    print("\n💡 API server is running in the background")
    print("🎨 Starting dashboard (this will open in your browser)...")
    print("\n📝 USAGE INSTRUCTIONS:")
    print("   1. The dashboard should open automatically in your browser")
    print("   2. If not, go to: http://localhost:8501")
    print("   3. Select a stock symbol and model type")
    print("   4. Choose prediction dates or use auto-generate")
    print("   5. Click 'Generate Prediction' to see results")
    print("\n⚠️  To stop the system: Close this terminal or press Ctrl+C")
    print("=" * 60)
    
    try:
        # Start dashboard (this will block until dashboard is closed)
        start_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
    finally:
        # Clean up API process
        if api_process:
            print("🔄 Stopping API server...")
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
                print("✅ API server stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing API server...")
                api_process.kill()
        
        print("👋 System shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Launcher interrupted by user")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        input("Press Enter to exit...")
