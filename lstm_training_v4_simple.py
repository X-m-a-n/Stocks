"""
Simple Parallel LSTM Training Script
Trains momentum-focused LSTM models for all stocks in parallel
Minimal debugging, clean output, saves results to Excel
"""

import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Simple Dataset class
class StockDataset(Dataset):
    def __init__(self, sequences, targets_reg, targets_class):
        self.sequences = torch.FloatTensor(sequences)
        self.targets_reg = torch.FloatTensor(targets_reg)
        self.targets_class = torch.LongTensor(targets_class)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets_reg[idx], self.targets_class[idx]

# Simple LSTM Model
class MomentumLSTM(nn.Module):
    def __init__(self, input_size):
        super(MomentumLSTM, self).__init__()
        
        # Bidirectional LSTM with attention
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=192,  # 96 * 2 (bidirectional)
            num_heads=6,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhancement network
        self.enhancement = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output heads
        self.regression_head = nn.Linear(96, 1)
        self.classification_head = nn.Linear(96, 4)  # down, neutral, up, big_up
    
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take last timestep
        features = attn_out[:, -1, :]
        
        # Enhancement
        enhanced = self.enhancement(features)
        
        # Predictions
        price_pred = self.regression_head(enhanced)
        direction_pred = self.classification_head(enhanced)
        
        return price_pred, direction_pred

def create_sequences(data, features, sequence_length=10):
    """Create training sequences from stock data"""
    sequences = []
    targets_reg = []
    targets_class = []
    
    for i in range(sequence_length, len(data)):
        # Create sequence
        seq = data.iloc[i-sequence_length:i][features].values
        
        # Skip if data is invalid
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue
        
        # Get targets
        reg_target = data.iloc[i]['next_day_change_pct']
        class_target = data.iloc[i]['direction_numeric']
        
        # Skip if targets are invalid
        if pd.isna(reg_target) or pd.isna(class_target):
            continue
        
        sequences.append(seq)
        targets_reg.append(reg_target)
        targets_class.append(class_target)
    
    return np.array(sequences), np.array(targets_reg), np.array(targets_class)

def prepare_stock_data(df, symbol, features):
    """Prepare data for one stock"""
    try:
        # Get stock data
        stock_data = df.filter(pl.col("symbol") == symbol).sort("date").to_pandas()
        
        # Define date splits
        train_end = pd.to_datetime('2024-09-01')
        test_start = pd.to_datetime('2024-09-30')
        
        # Convert dates
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        # Split data
        train_data = stock_data[stock_data['date'] < train_end].reset_index(drop=True)
        test_data = stock_data[stock_data['date'] >= test_start].reset_index(drop=True)
        
        # Check if enough data
        if len(train_data) < 100 or len(test_data) < 20:
            return None
        
        # Create sequences
        train_seq, train_reg, train_class = create_sequences(train_data, features)
        test_seq, test_reg, test_class = create_sequences(test_data, features)
        
        # Check if sequences created successfully
        if len(train_seq) == 0 or len(test_seq) == 0:
            return None
        
        return {
            'symbol': symbol,
            'train': (train_seq, train_reg, train_class),
            'test': (test_seq, test_reg, test_class),
            'train_days': len(train_data),
            'test_days': len(test_data)
        }
    
    except Exception:
        return None

def train_single_stock(args):
    """Train LSTM model for one stock"""
    symbol, data_dict = args
    
    try:
        # Extract data
        train_seq, train_reg, train_class = data_dict['train']
        test_seq, test_reg, test_class = data_dict['test']
        
        # Create datasets
        train_dataset = StockDataset(train_seq, train_reg, train_class)
        test_dataset = StockDataset(test_seq, test_reg, test_class)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize model
        model = MomentumLSTM(input_size=train_seq.shape[2])
        
        # Loss functions and optimizer
        reg_criterion = nn.SmoothL1Loss()
        class_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Training loop
        start_time = time.time()
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(80):  # Maximum 80 epochs
            model.train()
            epoch_loss = 0
            batches = 0
            
            for batch_seq, batch_reg, batch_class in train_loader:
                # Skip invalid batches
                if torch.isnan(batch_seq).any():
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                price_pred, direction_pred = model(batch_seq)
                
                # Calculate losses
                reg_loss = reg_criterion(price_pred.squeeze(), batch_reg)
                class_loss = class_criterion(direction_pred, batch_class)
                total_loss = 0.6 * reg_loss + 0.4 * class_loss
                
                # Skip if loss is invalid
                if torch.isnan(total_loss):
                    continue
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batches += 1
            
            if batches == 0:
                break
            
            avg_loss = epoch_loss / batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
                # Save best model
                torch.save(model.state_dict(), f'models/{symbol}_model.pth')
            else:
                patience += 1
                if patience >= 10:  # Stop if no improvement for 10 epochs
                    break
        
        # Load best model
        if os.path.exists(f'models/{symbol}_model.pth'):
            model.load_state_dict(torch.load(f'models/{symbol}_model.pth'))
        
        # Test evaluation
        model.eval()
        all_price_pred = []
        all_direction_pred = []
        all_price_true = []
        all_direction_true = []
        
        with torch.no_grad():
            for batch_seq, batch_reg, batch_class in test_loader:
                if torch.isnan(batch_seq).any():
                    continue
                
                price_pred, direction_pred = model(batch_seq)
                
                all_price_pred.extend(price_pred.squeeze().cpu().numpy())
                all_direction_pred.extend(torch.softmax(direction_pred, dim=1).argmax(dim=1).cpu().numpy())
                all_price_true.extend(batch_reg.cpu().numpy())
                all_direction_true.extend(batch_class.cpu().numpy())
        
        if len(all_price_pred) == 0:
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': 'No valid predictions',
                'training_time': time.time() - start_time
            }
        
        # Calculate metrics
        mae = mean_absolute_error(all_price_true, all_price_pred)
        rmse = np.sqrt(mean_squared_error(all_price_true, all_price_pred))
        direction_accuracy = accuracy_score(all_direction_true, all_direction_pred)
        
        # Calculate MAPE safely
        non_zero_mask = np.array(all_price_true) != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((np.array(all_price_true)[non_zero_mask] - 
                                 np.array(all_price_pred)[non_zero_mask]) / 
                                np.array(all_price_true)[non_zero_mask])) * 100
        else:
            mape = 0
        
        return {
            'symbol': symbol,
            'status': 'completed',
            'training_days': data_dict['train_days'],
            'test_days': data_dict['test_days'],
            'epochs_trained': epoch + 1,
            'training_time': time.time() - start_time,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'predictions_count': len(all_price_pred),
            'model_path': f'models/{symbol}_model.pth'
        }
    
    except Exception as e:
        return {
            'symbol': symbol,
            'status': 'failed',
            'error': str(e),
            'training_time': time.time() - start_time if 'start_time' in locals() else 0
        }

def main():
    """Main function - Simple and clean"""
    
    print("🚀 LSTM Parallel Training Started")
    print("="*50)
    
    # Load data
    data_path = r"C:\Users\Joshh\Projects\Stocks\Data\clean_stock_data.parquet"
    if not os.path.exists(data_path):
        print("❌ Data file not found:", data_path)
        return
    
    df = pl.read_parquet(data_path)
    print(f"✅ Loaded data: {df.shape}")
    
    # Load features
    if os.path.exists('preprocessing_metadata.json'):
        import json
        with open('preprocessing_metadata.json', 'r') as f:
            metadata = json.load(f)
            features = metadata.get('features_kept', [])[:20]  # Use top 20 features
    else:
        # Use basic features if metadata not found
        features = ['returns', 'intraday_return', 'daily_range', 'rsi_14', 
                   'macd', 'volume_ratio', 'volatility_10d', 'sma_20', 'sma_50']
        features = [f for f in features if f in df.columns]
    
    print(f"📊 Using {len(features)} features")
    
    # Get all stocks
    all_symbols = df['symbol'].unique().to_list()
    print(f"🏢 Found {len(all_symbols)} stocks")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Prepare data for all stocks
    print("📊 Preparing data...")
    prepare_args = [(df, symbol, features) for symbol in all_symbols]
    
    # Use half of CPU cores for data preparation
    with Pool(processes=mp.cpu_count() // 2) as pool:
        prepared_data = pool.starmap(prepare_stock_data, prepare_args)
    
    # Filter valid stocks
    valid_data = [(symbol, data) for symbol, data in zip(all_symbols, prepared_data) if data is not None]
    print(f"✅ Ready to train {len(valid_data)} stocks")
    
    if len(valid_data) == 0:
        print("❌ No valid stocks found")
        return
    
    # Train models in parallel
    print("🚀 Training models...")
    start_time = time.time()
    
    # Use 75% of CPU cores for training
    n_processes = max(1, int(mp.cpu_count() * 0.75))
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(train_single_stock, valid_data)
    
    total_time = time.time() - start_time
    
    # Process results
    successful = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] == 'failed']
    
    print("="*50)
    print("📈 TRAINING COMPLETED")
    print("="*50)
    print(f"✅ Successful: {len(successful)} models")
    print(f"❌ Failed: {len(failed)} models")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    
    if successful:
        # Calculate averages
        avg_accuracy = np.mean([r['direction_accuracy'] for r in successful])
        avg_mae = np.mean([r['mae'] for r in successful])
        avg_time = np.mean([r['training_time'] for r in successful])
        
        print(f"📊 Average accuracy: {avg_accuracy:.3f}")
        print(f"📊 Average MAE: {avg_mae:.3f}%")
        print(f"📊 Average time per model: {avg_time:.1f}s")
        
        # Show top performers
        top_5 = sorted(successful, key=lambda x: x['direction_accuracy'], reverse=True)[:5]
        print(f"\n🏆 Top 5 performers:")
        for i, model in enumerate(top_5, 1):
            print(f"   {i}. {model['symbol']}: {model['direction_accuracy']:.3f} accuracy")
    
    # Save results to Excel
    print(f"\n💾 Saving results to Excel...")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create Excel file with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"lstm_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # All results
        results_df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Successful models only
        if successful:
            success_df = pd.DataFrame(successful)
            success_df.to_excel(writer, sheet_name='Successful Models', index=False)
        
        # Failed models only
        if failed:
            failed_df = pd.DataFrame(failed)
            failed_df.to_excel(writer, sheet_name='Failed Models', index=False)
        
        # Summary statistics
        if successful:
            summary_data = {
                'Metric': ['Total Models', 'Successful', 'Failed', 'Success Rate', 
                          'Avg Accuracy', 'Avg MAE', 'Avg RMSE', 'Avg Training Time'],
                'Value': [len(results), len(successful), len(failed), 
                         f"{len(successful)/len(results)*100:.1f}%",
                         f"{avg_accuracy:.3f}", f"{avg_mae:.3f}%", 
                         f"{np.mean([r['rmse'] for r in successful]):.3f}%",
                         f"{avg_time:.1f}s"]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Results saved to: {excel_file}")
    print(f"🎯 Training completed successfully!")

if __name__ == "__main__":
    # Enable multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()