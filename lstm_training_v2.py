import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import multiprocessing as mp
from multiprocessing import Pool
import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StockDataset(Dataset):
    """Simple Dataset for stock prediction - no normalization"""
    
    def __init__(self, sequences, targets_regression, targets_classification):
        self.sequences = torch.FloatTensor(sequences)
        self.targets_regression = torch.FloatTensor(targets_regression)
        self.targets_classification = torch.LongTensor(targets_classification)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (self.sequences[idx], 
                self.targets_regression[idx], 
                self.targets_classification[idx])

class SimpleLSTM(nn.Module):
    """Simple LSTM model without complex normalization"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Simple LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Simple output layers
        self.regression_head = nn.Linear(hidden_size, 1)
        self.classification_head = nn.Linear(hidden_size, 4)  # 4 classes
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Generate predictions
        regression_output = self.regression_head(last_output)
        classification_output = self.classification_head(last_output)
        
        return regression_output, classification_output

def create_sequences_simple(data, features, target_reg, target_class, sequence_length=10):
    """Create sequences without normalization"""
    sequences = []
    targets_reg = []
    targets_class = []
    
    for i in range(sequence_length, len(data)):
        # Create sequence of features
        seq = data.iloc[i-sequence_length:i][features].values
        
        # Check for NaN or inf values
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue
        
        reg_target = data.iloc[i][target_reg]
        class_target = data.iloc[i][target_class]
        
        # Check targets are valid
        if pd.isna(reg_target) or pd.isna(class_target):
            continue
        if np.isnan(reg_target) or np.isinf(reg_target):
            continue
        
        sequences.append(seq)
        targets_reg.append(reg_target)
        targets_class.append(class_target)
    
    return np.array(sequences), np.array(targets_reg), np.array(targets_class)

def prepare_data_simple(df, symbol, features, sequence_length=10):
    """Simple data preparation without scaling"""
    
    # Filter data for symbol
    symbol_data = df.filter(pl.col("symbol") == symbol).sort("date")
    symbol_data_pd = symbol_data.to_pandas()
    
    # Define split dates
    train_end = pd.to_datetime('2024-09-01')
    val_end = pd.to_datetime('2024-09-30')
    
    # Convert date column if needed
    if not pd.api.types.is_datetime64_any_dtype(symbol_data_pd['date']):
        symbol_data_pd['date'] = pd.to_datetime(symbol_data_pd['date'])
    
    # Create splits
    train_data = symbol_data_pd[symbol_data_pd['date'] < train_end].copy()
    val_data = symbol_data_pd[(symbol_data_pd['date'] >= train_end) & 
                             (symbol_data_pd['date'] < val_end)].copy()
    test_data = symbol_data_pd[symbol_data_pd['date'] >= val_end].copy()
    
    # Check minimum data requirements
    if len(train_data) < sequence_length + 20:
        return None
    
    # Reset indices
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    # Check features exist
    available_features = [f for f in features if f in train_data.columns]
    if len(available_features) < 3:
        return None
    
    # Create sequences
    try:
        train_seq, train_reg, train_class = create_sequences_simple(
            train_data, available_features, 'next_day_change_pct', 'direction_numeric', sequence_length
        )
        
        if len(train_seq) == 0:
            return None
        
    except Exception as e:
        print(f"   Error creating training sequences for {symbol}: {e}")
        return None
    
    val_seq, val_reg, val_class = None, None, None
    if len(val_data) > sequence_length:
        try:
            val_seq, val_reg, val_class = create_sequences_simple(
                val_data, available_features, 'next_day_change_pct', 'direction_numeric', sequence_length
            )
        except Exception as e:
            print(f"   Warning: Could not create validation sequences for {symbol}: {e}")
    
    test_seq, test_reg, test_class = None, None, None
    if len(test_data) > sequence_length:
        try:
            test_seq, test_reg, test_class = create_sequences_simple(
                test_data, available_features, 'next_day_change_pct', 'direction_numeric', sequence_length
            )
        except Exception as e:
            print(f"   Warning: Could not create test sequences for {symbol}: {e}")
    
    return {
        'symbol': symbol,
        'train': (train_seq, train_reg, train_class),
        'val': (val_seq, val_reg, val_class) if val_seq is not None else None,
        'test': (test_seq, test_reg, test_class) if test_seq is not None else None,
        'train_days': len(train_data),
        'test_days': len(test_data),
        'features_used': available_features
    }

def train_simple_model(args):
    """Train simple LSTM model without normalization"""
    
    symbol, data_dict, config = args
    
    try:
        print(f"🚀 Training simple model for {symbol}...")
        start_time = time.time()
        
        # Extract training data
        train_seq, train_reg, train_class = data_dict['train']
        val_data = data_dict['val']
        test_data = data_dict['test']
        
        # Create training dataset
        train_dataset = StockDataset(train_seq, train_reg, train_class)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Validation loader
        val_loader = None
        if val_data is not None:
            val_seq, val_reg, val_class = val_data
            if len(val_seq) > 0:
                val_dataset = StockDataset(val_seq, val_reg, val_class)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize simple model
        input_size = train_seq.shape[2]
        model = SimpleLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        
        # Simple loss functions
        regression_criterion = nn.MSELoss()
        classification_criterion = nn.CrossEntropyLoss()
        
        # Simple optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batch_seq, batch_reg, batch_class in train_loader:
                # Check for NaN in batch
                if torch.isnan(batch_seq).any() or torch.isnan(batch_reg).any():
                    print(f"   Warning: NaN detected in batch for {symbol}, skipping...")
                    continue
                
                optimizer.zero_grad()
                
                reg_pred, class_pred = model(batch_seq)
                
                # Calculate losses
                reg_loss = regression_criterion(reg_pred.squeeze(), batch_reg)
                class_loss = classification_criterion(class_pred, batch_class)
                
                # Check for NaN in loss
                if torch.isnan(reg_loss) or torch.isnan(class_loss):
                    print(f"   Warning: NaN loss for {symbol} at epoch {epoch}")
                    continue
                
                total_loss = 0.7 * reg_loss + 0.3 * class_loss
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
            
            if batch_count == 0:
                print(f"   No valid batches for {symbol} at epoch {epoch}")
                break
            
            # Validation
            if val_loader is not None and epoch % 10 == 0:
                model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for val_seq, val_reg, val_class in val_loader:
                        if torch.isnan(val_seq).any() or torch.isnan(val_reg).any():
                            continue
                        
                        reg_pred, class_pred = model(val_seq)
                        
                        v_reg_loss = regression_criterion(reg_pred.squeeze(), val_reg)
                        v_class_loss = classification_criterion(class_pred, val_class)
                        
                        if not (torch.isnan(v_reg_loss) or torch.isnan(v_class_loss)):
                            val_loss += (0.7 * v_reg_loss + 0.3 * v_class_loss).item()
                            val_batches += 1
                
                if val_batches > 0:
                    val_loss /= val_batches
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), f'models/{symbol}_simple_model.pth')
                    else:
                        patience_counter += 1
                        if patience_counter >= config['patience']:
                            print(f"   Early stopping for {symbol} at epoch {epoch}")
                            break
        
        # Load best model
        if os.path.exists(f'models/{symbol}_simple_model.pth'):
            model.load_state_dict(torch.load(f'models/{symbol}_simple_model.pth'))
        
        # Evaluation
        model.eval()
        results = {
            'symbol': symbol,
            'status': 'completed',
            'training_days': data_dict['train_days'],
            'test_days': data_dict['test_days'],
            'epochs_trained': epoch + 1,
            'training_time': time.time() - start_time,
            'model_path': f'models/{symbol}_simple_model.pth',
            'features_count': len(data_dict['features_used'])
        }
        
        # Test evaluation
        if test_data is not None:
            test_seq, test_reg, test_class = test_data
            if len(test_seq) > 0:
                test_dataset = StockDataset(test_seq, test_reg, test_class)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
                
                all_reg_preds = []
                all_class_preds = []
                all_reg_true = []
                all_class_true = []
                
                with torch.no_grad():
                    for test_seq_batch, test_reg_batch, test_class_batch in test_loader:
                        if torch.isnan(test_seq_batch).any():
                            continue
                        
                        reg_pred, class_pred = model(test_seq_batch)
                        
                        all_reg_preds.extend(reg_pred.squeeze().cpu().numpy())
                        all_class_preds.extend(torch.softmax(class_pred, dim=1).argmax(dim=1).cpu().numpy())
                        all_reg_true.extend(test_reg_batch.cpu().numpy())
                        all_class_true.extend(test_class_batch.cpu().numpy())
                
                if len(all_reg_preds) > 0:
                    # Calculate metrics
                    mae = mean_absolute_error(all_reg_true, all_reg_preds)
                    rmse = np.sqrt(mean_squared_error(all_reg_true, all_reg_preds))
                    
                    # MAPE with safe division
                    non_zero_mask = np.array(all_reg_true) != 0
                    if np.sum(non_zero_mask) > 0:
                        mape = np.mean(np.abs((np.array(all_reg_true)[non_zero_mask] - 
                                             np.array(all_reg_preds)[non_zero_mask]) / 
                                            np.array(all_reg_true)[non_zero_mask])) * 100
                    else:
                        mape = 0
                    
                    direction_accuracy = accuracy_score(all_class_true, all_class_preds)
                    
                    results.update({
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'direction_accuracy': direction_accuracy,
                        'final_loss': best_val_loss,
                        'predictions_count': len(all_reg_preds)
                    })
        
        print(f"✅ Completed {symbol}: Acc: {results.get('direction_accuracy', 0):.3f}, "
              f"MAE: {results.get('mae', 0):.3f}, Time: {results['training_time']:.1f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'status': 'failed',
            'error': str(e),
            'training_time': time.time() - start_time if 'start_time' in locals() else 0
        }

def main():
    """Main training function - clean and simple"""
    
    # Simple configuration
    config = {
        'sequence_length': 10,      # Shorter sequences
        'hidden_size': 64,          # Smaller model
        'num_layers': 2,            # Simple architecture
        'dropout': 0.1,             # Light dropout
        'batch_size': 32,           # Reasonable batch size
        'epochs': 50,               # Fewer epochs
        'learning_rate': 0.001,     # Standard learning rate
        'patience': 15,             # Patience for early stopping
        'n_processes': min(10, mp.cpu_count())
    }
    
    print("🎯 CLEAN LSTM TRAINING - NO NORMALIZATION")
    print("="*60)
    print(f"📊 Sequence length: {config['sequence_length']} days")
    print(f"🧠 Model: {config['num_layers']} layers, {config['hidden_size']} hidden units")
    print(f"⚡ {config['n_processes']} parallel processes")
    
    # Load clean data (processed by preprocessing script)
    data_files = [
        r"C:\Users\Joshh\Projects\Stocks\clean_stock_data.parquet",
        "clean_stock_data.parquet",
        r"C:\Users\Joshh\Projects\Stocks\Data\stocks_df.csv"
    ]
    
    df = None
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"📁 Loading data from: {data_file}")
            try:
                if data_file.endswith('.parquet'):
                    df = pl.read_parquet(data_file)
                else:
                    df = pl.read_csv(data_file)
                print(f"✅ Data loaded successfully: {df.shape}")
                break
            except Exception as e:
                print(f"❌ Failed to load {data_file}: {e}")
                continue
    
    if df is None:
        print("❌ No data file found. Run preprocessing first!")
        return
    
    # Load features from preprocessing or use defaults
    features = []
    if os.path.exists('preprocessing_metadata.json'):
        with open('preprocessing_metadata.json', 'r') as f:
            metadata = json.load(f)
            features = metadata.get('features_kept', [])
        print(f"📊 Using {len(features)} features from preprocessing")
    else:
        # Use basic features
        basic_features = [
            'returns', 'intraday_return', 'daily_range', 'volume_ratio',
            'sma_20', 'sma_50', 'rsi_14', 'macd', 'volatility_10d', 'volatility_20d'
        ]
        features = [f for f in basic_features if f in df.columns]
        print(f"📊 Using {len(features)} basic features")
    
    if len(features) < 3:
        print("❌ Insufficient features available for training.")
        return
    
    print(f"🎯 Features: {features[:10]}{'...' if len(features) > 10 else ''}")
    
    # Get symbols for training
    symbols = df['symbol'].unique().to_list()[:10]
    print(f"🏢 Training models for {len(symbols)} symbols: {symbols}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Prepare data for each symbol
    print("\n📊 Preparing data for each symbol...")
    symbol_data = {}
    
    for symbol in symbols:
        try:
            print(f"   Processing {symbol}...")
            data_dict = prepare_data_simple(df, symbol, features, config['sequence_length'])
            if data_dict is not None:
                symbol_data[symbol] = data_dict
                print(f"   ✅ {symbol}: {data_dict['train_days']} train, {data_dict['test_days']} test days")
            else:
                print(f"   ⚠️ {symbol}: Insufficient data")
        except Exception as e:
            print(f"   ❌ {symbol}: Error preparing data - {e}")
    
    if not symbol_data:
        print("❌ No valid symbol data prepared.")
        return
    
    # Prepare multiprocessing arguments
    training_args = [(symbol, data_dict, config) for symbol, data_dict in symbol_data.items()]
    
    print(f"\n🚀 Starting clean training with {config['n_processes']} processes...")
    print("="*60)
    
    # Run training in parallel
    start_time = time.time()
    with Pool(processes=config['n_processes']) as pool:
        results = pool.map(train_simple_model, training_args)
    
    total_time = time.time() - start_time
    
    # Compile and save results
    results_df = pl.DataFrame(results)
    results_df.write_csv("clean_lstm_results.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 CLEAN TRAINING RESULTS")
    print("="*60)
    
    successful = results_df.filter(pl.col("status") == "completed")
    failed = results_df.filter(pl.col("status") == "failed")
    
    print(f"✅ Successfully trained: {len(successful)} models")
    print(f"❌ Failed: {len(failed)} models")
    print(f"⏱️ Total training time: {total_time:.1f}s")
    
    if len(successful) > 0:
        # Calculate metrics
        metrics = {}
        for metric in ['mae', 'rmse', 'direction_accuracy']:
            values = [x for x in successful[metric].to_list() if x is not None and not np.isnan(x)]
            if values:
                metrics[metric] = np.mean(values)
        
        avg_training_time = np.mean(successful['training_time'].to_list())
        
        print(f"\n📊 AVERAGE PERFORMANCE METRICS:")
        print(f"   MAE (Price Change): {metrics.get('mae', 0):.4f}%")
        print(f"   RMSE (Price Change): {metrics.get('rmse', 0):.4f}%")
        print(f"   Direction Accuracy: {metrics.get('direction_accuracy', 0):.4f}")
        print(f"   Average Training Time: {avg_training_time:.1f}s per model")
        
        print(f"\n🏆 TOP 5 MODELS BY DIRECTION ACCURACY:")
        top_models = successful.filter(pl.col("direction_accuracy").is_not_null()).sort("direction_accuracy", descending=True).head(5)
        for i, row in enumerate(top_models.iter_rows(named=True), 1):
            print(f"   {i}. {row['symbol']}: {row['direction_accuracy']:.4f} accuracy, "
                  f"MAE: {row.get('mae', 0):.4f}%, "
                  f"Features: {row.get('features_count', 0)}")
    
    if len(failed) > 0:
        print(f"\n❌ FAILED MODELS:")
        for row in failed.iter_rows(named=True):
            print(f"   {row['symbol']}: {row.get('error', 'Unknown error')}")
    
    print(f"\n📁 GENERATED FILES:")
    print(f"   • clean_lstm_results.csv - Training results and metrics")
    print(f"   • models/ - Trained model files (*_simple_model.pth)")
    
    print(f"\n✅ Clean training completed!")

if __name__ == "__main__":
    main()