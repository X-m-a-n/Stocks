import os
import sys
import subprocess
import time
import signal
from pathlib import Path
import logging
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemSetup:
    """Handles system setup and service management"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.processes = []
    
    
    def start_api_server(self):
        """Start FastAPI server"""
        logger.info("🚀 Starting API server...")
        
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "integrated_api:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ])
            
            self.processes.append(process)
            logger.info("✅ API server started on http://localhost:8000")
            time.sleep(3)  # Wait for startup
            return process
            
        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            return None
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        logger.info("📈 Starting dashboard...")
        
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "enhanced_dashboard.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ])
            
            self.processes.append(process)
            logger.info("✅ Dashboard started on http://localhost:8501")
            return process
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return None
    
    def cleanup(self):
        """Stop all running processes"""
        if not self.processes:
            return
            
        logger.info("🧹 Stopping services...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.warning(f"Error stopping process: {e}")
        
        self.processes.clear()
        logger.info("✅ All services stopped")

def main():
    """Main setup and launch function"""
    setup = SystemSetup()
    
    def signal_handler(signum, frame):
        logger.info("\n🛑 Shutdown requested...")
        setup.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🏗️  JSE Stock Prediction System")
    print("="*40)
    
    try:
        print("\n🚀 Starting services...")
        
        # Start API server
        api_process = setup.start_api_server()
        if not api_process:
            return False
        
        # Start dashboard
        dashboard_process = setup.start_dashboard()
        if not dashboard_process:
            setup.cleanup()
            return False
        
        # Show success message
        print("\n" + "="*50)
        print("🎉 System Running Successfully!")
        print("="*50)
        print("📡 API Server:    http://localhost:8000")
        print("📊 API Docs:      http://localhost:8000/docs") 
        print("📈 Dashboard:     http://localhost:8501")
        print("🧪 Health Check:  http://localhost:8000/health")
        print("="*50)
        print("\n💡 Press Ctrl+C to stop all services")
        print("⏳ Wait 10-15 seconds for full startup...")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
                
                # Check if processes died
                if api_process.poll() is not None:
                    logger.error("❌ API server stopped unexpectedly")
                    break
                    
                if dashboard_process.poll() is not None:
                    logger.error("❌ Dashboard stopped unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False
    finally:
        setup.cleanup()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        success = main()
        sys.exit(0 if success else 1)
    
    else:
        print("❌ Too many arguments")
        sys.exit(1)