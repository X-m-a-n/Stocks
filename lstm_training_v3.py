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
import warnings
import itertools
warnings.filterwarnings('ignore')

class StockDataset(Dataset):
    """Enhanced Dataset with feature scaling options"""
    
    def __init__(self, sequences, targets_regression, targets_classification, weights=None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets_regression = torch.FloatTensor(targets_regression)
        self.targets_classification = torch.LongTensor(targets_classification)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return (self.sequences[idx], 
                    self.targets_regression[idx], 
                    self.targets_classification[idx],
                    self.weights[idx])
        return (self.sequences[idx], 
                self.targets_regression[idx], 
                self.targets_classification[idx])

class AdvancedLSTM(nn.Module):
    """Advanced LSTM with multiple architectural improvements"""
    
    def __init__(self, input_size, config):
        super(AdvancedLSTM, self).__init__()
        
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.use_attention = config['use_attention']
        self.use_residual = config['use_residual']
        
        # Calculate actual hidden dimension
        self.lstm_hidden_dim = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Feature preprocessing layers
        if config['feature_preprocessing']:
            self.feature_norm = nn.LayerNorm(input_size)
            self.feature_projection = nn.Linear(input_size, config['projection_dim'])
            actual_input_size = config['projection_dim']
        else:
            self.feature_norm = None
            self.feature_projection = None
            actual_input_size = input_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=actual_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config['lstm_dropout'] if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.lstm_hidden_dim,
                num_heads=config['attention_heads'],
                dropout=config['attention_dropout'],
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(self.lstm_hidden_dim)
        
        # Residual connections
        if self.use_residual and config['feature_preprocessing']:
            self.residual_projection = nn.Linear(actual_input_size, self.lstm_hidden_dim)
        
        # Advanced pooling
        self.pooling_type = config['pooling_type']
        if self.pooling_type == 'attention_pooling':
            self.pooling_attention = nn.Linear(self.lstm_hidden_dim, 1)
        
        # Feature enhancement layers
        self.feature_enhancement = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, config['enhancement_dim']),
            nn.ReLU(),
            nn.Dropout(config['enhancement_dropout']),
            nn.Linear(config['enhancement_dim'], config['enhancement_dim']),
            nn.ReLU(),
            nn.Dropout(config['enhancement_dropout'])
        )
        
        # Separate task-specific networks
        # Regression head (price prediction)
        self.regression_net = nn.Sequential(
            nn.Linear(config['enhancement_dim'], config['regression_hidden']),
            nn.ReLU(),
            nn.Dropout(config['head_dropout']),
            nn.Linear(config['regression_hidden'], config['regression_hidden'] // 2),
            nn.ReLU(),
            nn.Dropout(config['head_dropout']),
            nn.Linear(config['regression_hidden'] // 2, 1)
        )
        
        # Classification head (direction prediction)
        self.classification_net = nn.Sequential(
            nn.Linear(config['enhancement_dim'], config['classification_hidden']),
            nn.ReLU(),
            nn.Dropout(config['head_dropout']),
            nn.Linear(config['classification_hidden'], config['classification_hidden'] // 2),
            nn.ReLU(), 
            nn.Dropout(config['head_dropout']),
            nn.Linear(config['classification_hidden'] // 2, 4)  # 4 classes
        )
        
        # Additional auxiliary heads for multi-task learning
        if config['use_auxiliary_tasks']:
            # Volatility prediction
            self.volatility_head = nn.Linear(config['enhancement_dim'], 1)
            # Volume prediction  
            self.volume_head = nn.Linear(config['enhancement_dim'], 1)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Feature preprocessing
        if self.feature_norm is not None:
            x = self.feature_norm(x)
        
        if self.feature_projection is not None:
            x = self.feature_projection(x)
            original_x = x.clone()  # For residual connections
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        if self.use_attention:
            # Self-attention over sequence
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Residual connection + layer norm
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Residual connection from input
        if self.use_residual and hasattr(self, 'residual_projection'):
            # Average the input sequence and project
            input_avg = original_x.mean(dim=1, keepdim=True)  # [batch, 1, features]
            input_projected = self.residual_projection(input_avg.squeeze(1))  # [batch, hidden]
        
        # Advanced pooling strategies
        if self.pooling_type == 'last':
            pooled = lstm_out[:, -1, :]
        elif self.pooling_type == 'mean':
            pooled = lstm_out.mean(dim=1)
        elif self.pooling_type == 'max':
            pooled = lstm_out.max(dim=1)[0]
        elif self.pooling_type == 'attention_pooling':
            # Learnable attention pooling
            attention_weights = torch.softmax(self.pooling_attention(lstm_out), dim=1)
            pooled = (lstm_out * attention_weights).sum(dim=1)
        elif self.pooling_type == 'concat':
            # Concatenate last, mean, and max
            last = lstm_out[:, -1, :]
            mean = lstm_out.mean(dim=1)
            max_pool = lstm_out.max(dim=1)[0]
            pooled = torch.cat([last, mean, max_pool], dim=1)
            # Adjust enhancement input size for concat
            if not hasattr(self, '_concat_projection'):
                self._concat_projection = nn.Linear(self.lstm_hidden_dim * 3, self.lstm_hidden_dim).to(x.device)
            pooled = self._concat_projection(pooled)
        
        # Add residual connection if enabled
        if self.use_residual and hasattr(self, 'residual_projection'):
            pooled = pooled + input_projected
        
        # Feature enhancement
        enhanced_features = self.feature_enhancement(pooled)
        
        # Main task predictions
        regression_output = self.regression_net(enhanced_features)
        classification_output = self.classification_net(enhanced_features)
        
        outputs = {
            'regression': regression_output,
            'classification': classification_output,
            'features': enhanced_features
        }
        
        # Auxiliary tasks
        if self.config['use_auxiliary_tasks']:
            outputs['volatility'] = self.volatility_head(enhanced_features)
            outputs['volume'] = self.volume_head(enhanced_features)
        
        return outputs

def create_sequences_advanced(data, features, target_cols, sequence_length=15):
    """Create sequences with additional targets for auxiliary tasks"""
    sequences = []
    targets = {}
    
    # Initialize target lists
    for col in target_cols:
        targets[col] = []
    
    for i in range(sequence_length, len(data)):
        # Create sequence of features
        seq = data.iloc[i-sequence_length:i][features].values
        
        # Check for NaN or inf values
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue
        
        # Collect all targets
        valid_targets = True
        current_targets = {}
        
        for col in target_cols:
            target_val = data.iloc[i][col]
            if pd.isna(target_val) or np.isnan(target_val) or np.isinf(target_val):
                valid_targets = False
                break
            current_targets[col] = target_val
        
        if not valid_targets:
            continue
        
        sequences.append(seq)
        for col in target_cols:
            targets[col].append(current_targets[col])
    
    # Convert to numpy arrays
    sequences = np.array(sequences)
    for col in target_cols:
        targets[col] = np.array(targets[col])
    
    return sequences, targets

def prepare_data_advanced(df, symbol, features, config):
    """Advanced data preparation with more sophisticated preprocessing"""
    
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
    if len(train_data) < config['sequence_length'] + 50:
        return None
    
    # Reset indices
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    # Generate additional targets for auxiliary tasks
    for data in [train_data, val_data, test_data]:
        if len(data) > 0:
            # Rolling volatility target (next day volatility)
            data['next_day_volatility'] = data['daily_range'].shift(-1)
            # Rolling volume target (next day volume change)
            data['next_day_volume_change'] = ((data['volume'].shift(-1) - data['volume']) / data['volume'] * 100)
    
    # Define target columns
    target_cols = ['next_day_change_pct', 'direction_numeric']
    if config['use_auxiliary_tasks']:
        target_cols.extend(['next_day_volatility', 'next_day_volume_change'])
    
    # Check features exist
    available_features = [f for f in features if f in train_data.columns]
    if len(available_features) < 5:
        return None
    
    # Create sequences
    try:
        train_seq, train_targets = create_sequences_advanced(
            train_data, available_features, target_cols, config['sequence_length']
        )
        
        if len(train_seq) == 0:
            return None
            
    except Exception as e:
        print(f"   Error creating training sequences for {symbol}: {e}")
        return None
    
    # Validation sequences
    val_seq, val_targets = None, None
    if len(val_data) > config['sequence_length']:
        try:
            val_seq, val_targets = create_sequences_advanced(
                val_data, available_features, target_cols, config['sequence_length']
            )
        except Exception as e:
            print(f"   Warning: Could not create validation sequences for {symbol}: {e}")
    
    # Test sequences
    test_seq, test_targets = None, None
    if len(test_data) > config['sequence_length']:
        try:
            test_seq, test_targets = create_sequences_advanced(
                test_data, available_features, target_cols, config['sequence_length']
            )
        except Exception as e:
            print(f"   Warning: Could not create test sequences for {symbol}: {e}")
    
    return {
        'symbol': symbol,
        'train': (train_seq, train_targets),
        'val': (val_seq, val_targets) if val_seq is not None else None,
        'test': (test_seq, test_targets) if test_seq is not None else None,
        'train_days': len(train_data),
        'test_days': len(test_data),
        'features_used': available_features
    }

def train_advanced_model(args):
    """Train advanced LSTM model with comprehensive configuration"""
    
    symbol, data_dict, config = args
    
    try:
        print(f"🚀 Training advanced model for {symbol}...")
        start_time = time.time()
        
        # Extract training data
        train_seq, train_targets = data_dict['train']
        val_data = data_dict['val']
        test_data = data_dict['test']
        
        # Create training dataset
        train_dataset = StockDataset(
            train_seq, 
            train_targets['next_day_change_pct'], 
            train_targets['direction_numeric']
        )
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Validation loader
        val_loader = None
        if val_data is not None:
            val_seq, val_targets = val_data
            if len(val_seq) > 0:
                val_dataset = StockDataset(
                    val_seq, 
                    val_targets['next_day_change_pct'], 
                    val_targets['direction_numeric']
                )
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize advanced model
        input_size = train_seq.shape[2]
        model = AdvancedLSTM(input_size, config)
        
        # Advanced loss functions
        regression_criterion = nn.SmoothL1Loss() if config['use_smooth_loss'] else nn.MSELoss()
        
        # Focal loss for classification (handles class imbalance better)
        if config['use_focal_loss']:
            class_weights = torch.FloatTensor([1.2, 0.8, 1.0, 1.5])  # down, neutral, up, big_up
        else:
            class_weights = None
        
        classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Auxiliary task losses
        if config['use_auxiliary_tasks']:
            volatility_criterion = nn.MSELoss()
            volume_criterion = nn.MSELoss()
        
        # Advanced optimizers
        if config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                betas=config['adam_betas']
            )
        elif config['optimizer'] == 'radam':
            optimizer = optim.RAdam(model.parameters(), lr=config['learning_rate'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Advanced schedulers
        if config['scheduler'] == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['learning_rate'] * 3,
                epochs=config['epochs'],
                steps_per_epoch=len(train_loader)
            )
        elif config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['epochs']
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience=config['scheduler_patience'], 
                factor=0.5
            )
        
        # Training loop with advanced features
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            model.train()
            epoch_losses = {'total': 0, 'regression': 0, 'classification': 0}
            if config['use_auxiliary_tasks']:
                epoch_losses.update({'volatility': 0, 'volume': 0})
            
            batch_count = 0
            
            for batch_seq, batch_reg, batch_class in train_loader:
                # Check for NaN in batch
                if torch.isnan(batch_seq).any() or torch.isnan(batch_reg).any():
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_seq)
                
                # Main task losses
                reg_loss = regression_criterion(outputs['regression'].squeeze(), batch_reg)
                class_loss = classification_criterion(outputs['classification'], batch_class)
                
                # Check for NaN in losses
                if torch.isnan(reg_loss) or torch.isnan(class_loss):
                    continue
                
                # Combined loss
                total_loss = (config['regression_weight'] * reg_loss + 
                             config['classification_weight'] * class_loss)
                
                # Auxiliary task losses
                if config['use_auxiliary_tasks']:
                    # Note: Would need auxiliary targets in DataLoader for full implementation
                    pass
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
                
                optimizer.step()
                
                if config['scheduler'] == 'onecycle':
                    scheduler.step()
                
                # Track losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['regression'] += reg_loss.item()
                epoch_losses['classification'] += class_loss.item()
                batch_count += 1
            
            if batch_count == 0:
                break
            
            # Validation
            if val_loader is not None and epoch % config['validation_freq'] == 0:
                model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for val_seq, val_reg, val_class in val_loader:
                        if torch.isnan(val_seq).any() or torch.isnan(val_reg).any():
                            continue
                        
                        outputs = model(val_seq)
                        
                        v_reg_loss = regression_criterion(outputs['regression'].squeeze(), val_reg)
                        v_class_loss = classification_criterion(outputs['classification'], val_class)
                        
                        if not (torch.isnan(v_reg_loss) or torch.isnan(v_class_loss)):
                            val_loss += (config['regression_weight'] * v_reg_loss + 
                                       config['classification_weight'] * v_class_loss).item()
                            val_batches += 1
                
                if val_batches > 0:
                    val_loss /= val_batches
                    
                    if config['scheduler'] != 'onecycle':
                        scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), f'models/{symbol}_advanced_model.pth')
                    else:
                        patience_counter += 1
                        if patience_counter >= config['patience']:
                            print(f"   Early stopping for {symbol} at epoch {epoch}")
                            break
        
        # Load best model and evaluate
        if os.path.exists(f'models/{symbol}_advanced_model.pth'):
            model.load_state_dict(torch.load(f'models/{symbol}_advanced_model.pth'))
        
        # Evaluation logic (similar to simple model but with advanced outputs)
        results = evaluate_advanced_model(model, test_data, data_dict, config, symbol, start_time)
        
        return results
        
    except Exception as e:
        print(f"❌ Failed {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'status': 'failed',
            'error': str(e),
            'training_time': time.time() - start_time if 'start_time' in locals() else 0
        }

def evaluate_advanced_model(model, test_data, data_dict, config, symbol, start_time):
    """Comprehensive evaluation of advanced model"""
    
    model.eval()
    results = {
        'symbol': symbol,
        'status': 'completed',
        'training_days': data_dict['train_days'],
        'test_days': data_dict['test_days'],
        'training_time': time.time() - start_time,
        'model_path': f'models/{symbol}_advanced_model.pth',
        'features_count': len(data_dict['features_used']),
        'config': config['config_name']
    }
    
    if test_data is not None:
        test_seq, test_targets = test_data
        if len(test_seq) > 0:
            test_dataset = StockDataset(
                test_seq, 
                test_targets['next_day_change_pct'], 
                test_targets['direction_numeric']
            )
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
            
            all_reg_preds = []
            all_class_preds = []
            all_reg_true = []
            all_class_true = []
            
            with torch.no_grad():
                for test_seq_batch, test_reg_batch, test_class_batch in test_loader:
                    if torch.isnan(test_seq_batch).any():
                        continue
                    
                    outputs = model(test_seq_batch)
                    
                    all_reg_preds.extend(outputs['regression'].squeeze().cpu().numpy())
                    all_class_preds.extend(torch.softmax(outputs['classification'], dim=1).argmax(dim=1).cpu().numpy())
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
                    'predictions_count': len(all_reg_preds)
                })
    
    return results

# Parameter tuning configurations
PARAMETER_CONFIGS = {
    'baseline_advanced': {
        'config_name': 'baseline_advanced',
        'sequence_length': 15,
        'hidden_size': 128,
        'num_layers': 3,
        'bidirectional': True,
        'use_attention': True,
        'attention_heads': 4,
        'attention_dropout': 0.1,
        'use_residual': True,
        'feature_preprocessing': True,
        'projection_dim': 64,
        'pooling_type': 'attention_pooling',
        'enhancement_dim': 256,
        'enhancement_dropout': 0.2,
        'regression_hidden': 128,
        'classification_hidden': 128,
        'head_dropout': 0.3,
        'use_auxiliary_tasks': False,
        'lstm_dropout': 0.2,
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'adam_betas': (0.9, 0.999),
        'optimizer': 'adamw',
        'scheduler': 'onecycle',
        'scheduler_patience': 10,
        'use_smooth_loss': True,
        'use_focal_loss': True,
        'regression_weight': 0.6,
        'classification_weight': 0.4,
        'grad_clip': 1.0,
        'patience': 20,
        'validation_freq': 5
    },
    
    'high_capacity': {
        'config_name': 'high_capacity',
        'sequence_length': 20,
        'hidden_size': 256,
        'num_layers': 4,
        'bidirectional': True,
        'use_attention': True,
        'attention_heads': 8,
        'attention_dropout': 0.1,
        'use_residual': True,
        'feature_preprocessing': True,
        'projection_dim': 128,
        'pooling_type': 'concat',
        'enhancement_dim': 512,
        'enhancement_dropout': 0.3,
        'regression_hidden': 256,
        'classification_hidden': 256,
        'head_dropout': 0.4,
        'use_auxiliary_tasks': True,
        'lstm_dropout': 0.3,
        'batch_size': 32,
        'epochs': 150,
        'learning_rate': 0.0005,
        'weight_decay': 1e-3,
        'adam_betas': (0.9, 0.999),
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'scheduler_patience': 15,
        'use_smooth_loss': True,
        'use_focal_loss': True,
        'regression_weight': 0.5,
        'classification_weight': 0.5,
        'grad_clip': 0.5,
        'patience': 25,
        'validation_freq': 10
    },
    
    'momentum_focused': {
        'config_name': 'momentum_focused',
        'sequence_length': 10,
        'hidden_size': 96,
        'num_layers': 2,
        'bidirectional': False,
        'use_attention': True,
        'attention_heads': 6,
        'attention_dropout': 0.05,
        'use_residual': False,
        'feature_preprocessing': False,
        'projection_dim': 64,
        'pooling_type': 'last',
        'enhancement_dim': 192,
        'enhancement_dropout': 0.15,
        'regression_hidden': 96,
        'classification_hidden': 96,
        'head_dropout': 0.2,
        'use_auxiliary_tasks': False,
        'lstm_dropout': 0.1,
        'batch_size': 128,
        'epochs': 80,
        'learning_rate': 0.002,
        'weight_decay': 5e-4,
        'adam_betas': (0.9, 0.99),
        'optimizer': 'adamw',
        'scheduler': 'onecycle',
        'scheduler_patience': 8,
        'use_smooth_loss': False,
        'use_focal_loss': False,
        'regression_weight': 0.7,
        'classification_weight': 0.3,
        'grad_clip': 1.5,
        'patience': 15,
        'validation_freq': 5
    }
}

def main():
    """Main function to run parameter tuning experiments"""
    
    print("🚀 ADVANCED LSTM PARAMETER TUNING")
    print("="*60)
    
    # Load clean data
    data_files = [
        r"C:\Users\Joshh\Projects\Stocks\clean_stock_data.parquet",
        "clean_stock_data.parquet"
    ]
    
    df = None
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"📁 Loading data from: {data_file}")
            df = pl.read_parquet(data_file)
            print(f"✅ Data loaded successfully: {df.shape}")
            break
    
    if df is None:
        print("❌ No clean data found. Run preprocessing first!")
        return
    
    # Load features from preprocessing
    if os.path.exists('preprocessing_metadata.json'):
        with open('preprocessing_metadata.json', 'r') as f:
            metadata = json.load(f)
            features = metadata.get('features_kept', [])[:20]  # Use top 20 features
    else:
        print("❌ No preprocessing metadata found!")
        return
    
    print(f"🎯 Using {len(features)} features for advanced models")
    
    # Use same symbols as simple model for comparison
    symbols = ['WISYNCO', 'BIL', 'KW', 'KEY', 'KREMI', 'TJH', 'CAC', 'JMMBGL']
    print(f"🏢 Testing {len(symbols)} symbols: {symbols}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Run experiments for each configuration
    all_results = []
    
    for config_name, config in PARAMETER_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"🧪 TESTING CONFIGURATION: {config_name.upper()}")
        print(f"{'='*60}")
        
        # Print key configuration details
        print(f"📊 Architecture: {config['num_layers']} layers, {config['hidden_size']} hidden")
        print(f"🔄 Sequence: {config['sequence_length']} days")
        print(f"🎯 Attention: {config['use_attention']}, Bidirectional: {config['bidirectional']}")
        print(f"⚡ Batch size: {config['batch_size']}, Epochs: {config['epochs']}")
        print(f"📈 LR: {config['learning_rate']}, Optimizer: {config['optimizer']}")
        
        # Prepare data for each symbol
        print(f"\n📊 Preparing data for {config_name}...")
        symbol_data = {}
        
        for symbol in symbols:
            try:
                print(f"   Processing {symbol}...")
                data_dict = prepare_data_advanced(df, symbol, features, config)
                if data_dict is not None:
                    symbol_data[symbol] = data_dict
                    print(f"   ✅ {symbol}: {data_dict['train_days']} train, {data_dict['test_days']} test days")
                else:
                    print(f"   ⚠️ {symbol}: Insufficient data")
            except Exception as e:
                print(f"   ❌ {symbol}: Error preparing data - {e}")
        
        if not symbol_data:
            print(f"❌ No valid data for {config_name}")
            continue
        
        # Prepare multiprocessing arguments
        training_args = [(symbol, data_dict, config) for symbol, data_dict in symbol_data.items()]
        
        print(f"\n🚀 Training {config_name} models...")
        
        # Run training (single process for debugging, can be made parallel)
        config_results = []
        for args in training_args:
            result = train_advanced_model(args)
            config_results.append(result)
        
        # Add configuration name to results
        for result in config_results:
            result['config'] = config_name
        
        all_results.extend(config_results)
        
        # Print configuration summary
        successful = [r for r in config_results if r['status'] == 'completed']
        failed = [r for r in config_results if r['status'] == 'failed']
        
        print(f"\n📊 {config_name.upper()} RESULTS:")
        print(f"✅ Successful: {len(successful)}/{len(config_results)}")
        
        if successful:
            avg_acc = np.mean([r.get('direction_accuracy', 0) for r in successful])
            avg_mae = np.mean([r.get('mae', 0) for r in successful if r.get('mae')])
            avg_time = np.mean([r.get('training_time', 0) for r in successful])
            
            print(f"📈 Avg Direction Accuracy: {avg_acc:.3f}")
            print(f"📊 Avg MAE: {avg_mae:.3f}%")
            print(f"⏱️ Avg Training Time: {avg_time:.1f}s")
            
            # Show top performers
            top_performers = sorted(successful, key=lambda x: x.get('direction_accuracy', 0), reverse=True)[:3]
            print(f"🏆 Top 3 performers:")
            for i, perf in enumerate(top_performers, 1):
                print(f"   {i}. {perf['symbol']}: {perf.get('direction_accuracy', 0):.3f} acc, {perf.get('mae', 0):.3f}% MAE")
    
    # Save all results
    results_df = pl.DataFrame(all_results)
    results_df.write_csv("advanced_lstm_comparison.csv")
    
    # Comprehensive comparison analysis
    print(f"\n{'='*80}")
    print("🏆 COMPREHENSIVE COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    # Group by configuration
    config_summary = {}
    for config_name in PARAMETER_CONFIGS.keys():
        config_results = [r for r in all_results if r.get('config') == config_name and r['status'] == 'completed']
        if config_results:
            config_summary[config_name] = {
                'count': len(config_results),
                'avg_accuracy': np.mean([r.get('direction_accuracy', 0) for r in config_results]),
                'avg_mae': np.mean([r.get('mae', 0) for r in config_results if r.get('mae')]),
                'avg_rmse': np.mean([r.get('rmse', 0) for r in config_results if r.get('rmse')]),
                'avg_training_time': np.mean([r.get('training_time', 0) for r in config_results]),
                'best_accuracy': max([r.get('direction_accuracy', 0) for r in config_results]),
                'best_symbol': max(config_results, key=lambda x: x.get('direction_accuracy', 0))['symbol']
            }
    
    # Print comparison table
    print(f"\n📊 CONFIGURATION COMPARISON:")
    print(f"{'Config':<20} {'Avg Acc':<10} {'Best Acc':<10} {'Avg MAE':<10} {'Avg Time':<12} {'Best Symbol':<10}")
    print("-" * 80)
    
    for config_name, stats in config_summary.items():
        print(f"{config_name:<20} {stats['avg_accuracy']:<10.3f} {stats['best_accuracy']:<10.3f} "
              f"{stats['avg_mae']:<10.3f} {stats['avg_training_time']:<12.1f} {stats['best_symbol']:<10}")
    
    # Symbol-wise comparison
    print(f"\n📈 SYMBOL-WISE COMPARISON:")
    symbol_comparison = {}
    
    for symbol in symbols:
        symbol_results = [r for r in all_results if r['symbol'] == symbol and r['status'] == 'completed']
        if symbol_results:
            symbol_comparison[symbol] = {}
            for result in symbol_results:
                config = result.get('config', 'unknown')
                symbol_comparison[symbol][config] = {
                    'accuracy': result.get('direction_accuracy', 0),
                    'mae': result.get('mae', 0)
                }
    
    for symbol, configs in symbol_comparison.items():
        print(f"\n🏢 {symbol}:")
        for config, metrics in configs.items():
            print(f"   {config:<20}: {metrics['accuracy']:.3f} acc, {metrics['mae']:.3f}% MAE")
    
    # Find overall best configuration
    if config_summary:
        best_config = max(config_summary.keys(), key=lambda x: config_summary[x]['avg_accuracy'])
        best_stats = config_summary[best_config]
        
        print(f"\n🎉 BEST OVERALL CONFIGURATION: {best_config.upper()}")
        print(f"📊 Average Accuracy: {best_stats['avg_accuracy']:.3f}")
        print(f"🎯 Best Single Result: {best_stats['best_accuracy']:.3f} ({best_stats['best_symbol']})")
        print(f"📈 Average MAE: {best_stats['avg_mae']:.3f}%")
        print(f"⏱️ Average Training Time: {best_stats['avg_training_time']:.1f}s")
    
    print(f"\n📁 RESULTS SAVED TO: advanced_lstm_comparison.csv")
    print(f"🎯 Use the best configuration for production models!")

if __name__ == "__main__":
    main()