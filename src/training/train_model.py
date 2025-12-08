"""
Training Script for TCANet-IDMS Trajectory Prediction

Implements PyTorch training pipeline for TCANet adapted to IDMS trajectory prediction,
compatible with existing experiment infrastructure.
"""

import sys
import os
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import logging

# Import our PyTorch models and data adapter
from pytorch_models.tcanet_idms import create_tcanet_idms_model
from pytorch_models.pytorch_data_adapter import create_idms_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TCANetIDMSTrainer:
    """
    PyTorch trainer for TCANet-IDMS models
    
    Provides similar interface to existing TensorFlow ElbowTrajectoryTrainer
    for compatibility with experiment infrastructure.
    """
    
    def __init__(self,
                 dataset_path: str,
                 model_config: Dict[str, Any],
                 save_dir: str = "pytorch_models/experiments",
                 experiment_name: str = None,
                 device: str = None):
        """
        Initialize TCANet-IDMS trainer
        
        Args:
            dataset_path: Path to HDF5 dataset
            model_config: Model configuration parameters
            save_dir: Directory to save models and results
            experiment_name: Name for this experiment
            device: Device to use ('cuda' or 'cpu')
        """
        self.dataset_path = Path(dataset_path)
        self.model_config = model_config
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name or f"tcanet_idms_{int(time.time())}"
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Create save directory
        self.experiment_dir = self.save_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'epoch': []
        }
        
    def create_model(self):
        """Create TCANet-IDMS model based on configuration"""
        
        config = self.model_config
        
        self.model = create_tcanet_idms_model(
            window_size=config.get('window_size', 1000),
            n_channels=config.get('n_channels', 4),
            trajectory_points=config.get('n_trajectory_points', 10),
            trajectory_horizon=config.get('horizon', 0.5),
            trajectory_delay=config.get('delay', 0.05),
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Created TCANet-IDMS model:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        return self.model
        
    def create_dataloaders(self):
        """Create PyTorch data loaders"""
        
        config = self.model_config
        
        self.train_loader, self.val_loader, self.test_loader = create_idms_dataloaders(
            dataset_path=str(self.dataset_path),
            window_size=config.get('window_size', 1000),
            horizon=config.get('horizon', 0.5),
            batch_size=config.get('batch_size', 32),
            emg_preproc=config.get('emg_preproc', None),
            subjects=config.get('subjects', None),
            trials=config.get('trials', None)
        )
        
        logger.info(f"Created data loaders:")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        logger.info(f"  Test batches: {len(self.test_loader)}")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler"""
        
        config = self.model_config
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 1e-2)
        )
        
        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'plateau')
        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=config.get('lr_patience', 10),
                min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('max_epochs', 200),
                eta_min=1e-7
            )
        else:
            self.scheduler = None
            
        logger.info(f"Created optimizer: {type(self.optimizer).__name__}")
        if self.scheduler:
            logger.info(f"Created scheduler: {type(self.scheduler).__name__}")
            
        return self.optimizer, self.scheduler
    
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        # Convert to numpy for metric calculation
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Flatten for overall metrics
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(target_flat, pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_flat, pred_flat)
        
        # R-squared
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            # Move to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = nn.MSELoss()(predictions, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            if self.model_config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_config['grad_clip'])
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(y_batch.detach())
            
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                # Move to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                predictions = self.model(X_batch)
                loss = nn.MSELoss()(predictions, y_batch)
                
                # Accumulate metrics
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(y_batch)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def test_model(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                # Move to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                predictions = self.model(X_batch)
                loss = nn.MSELoss()(predictions, y_batch)
                
                # Accumulate metrics
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(y_batch)
        
        # Calculate test metrics
        avg_loss = total_loss / len(self.test_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics, all_predictions, all_targets
    
    def save_model(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'model_config': self.model_config,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = self.experiment_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model at epoch {epoch}")
    
    def save_config(self):
        """Save model configuration"""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f, indent=2)
    
    def save_history(self):
        """Save training history"""
        history_path = self.experiment_dir / "training_history.csv"
        df = pd.DataFrame(self.history)
        df.to_csv(history_path, index=False)
    
    def train(self, 
              max_epochs: int = 200,
              patience: int = 20,
              min_improvement: float = 1e-6):
        """
        Main training loop
        
        Args:
            max_epochs: Maximum number of epochs
            patience: Early stopping patience
            min_improvement: Minimum improvement for early stopping
        """
        
        logger.info("Starting TCANet-IDMS training...")
        
        # Initialize components
        if self.model is None:
            self.create_model()
        if self.train_loader is None:
            self.create_dataloaders()
        if self.optimizer is None:
            self.create_optimizer_and_scheduler()
            
        # Save configuration
        self.save_config()
        
        # Training state
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        start_time = time.time()
        
        logger.info(f"Training for up to {max_epochs} epochs with early stopping (patience={patience})")
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch  
            val_metrics = self.validate_epoch()
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            
            # Check for best model
            is_best = val_metrics['loss'] < (best_val_loss - min_improvement)
            if is_best:
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
                self.save_model(epoch, is_best=True)
            else:
                epochs_without_improvement += 1
            
            # Log progress
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch:3d}/{max_epochs} ({epoch_time:.1f}s) | "
                       f"Train Loss: {train_metrics['loss']:.6f} RMSE: {train_metrics['rmse']:.6f} | "
                       f"Val Loss: {val_metrics['loss']:.6f} RMSE: {val_metrics['rmse']:.6f} | "
                       f"LR: {current_lr:.2e} | {'*' if is_best else ' '}")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
                
            # Save periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self.save_model(epoch, is_best=False)
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s ({total_time/3600:.1f}h)")
        
        # Final test evaluation
        logger.info("Evaluating on test set...")
        test_metrics, test_predictions, test_targets = self.test_model()
        
        logger.info(f"Test Results:")
        logger.info(f"  RMSE: {test_metrics['rmse']:.6f}")
        logger.info(f"  MAE: {test_metrics['mae']:.6f}")
        logger.info(f"  R²: {test_metrics['r2']:.6f}")
        
        # Save final results
        self.save_history()
        
        results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_time': total_time,
            'epochs_trained': epoch + 1,
            'best_epoch': epoch - epochs_without_improvement,
            'experiment_dir': str(self.experiment_dir)
        }
        
        # Save results summary
        results_path = self.experiment_dir / "results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    json_results[k] = {k2: float(v2) if isinstance(v2, np.floating) else v2 for k2, v2 in v.items()}
                else:
                    json_results[k] = float(v) if isinstance(v, np.floating) else v
            json.dump(json_results, f, indent=2)
        
        return results


def create_experiment_config(
    window_size: int = 1000,
    horizon: float = 0.5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    emg_preproc: str = None,
    subjects: List[str] = None,
    trials: List[str] = None
) -> Dict[str, Any]:
    """
    Create experiment configuration dictionary
    
    Returns:
        Configuration dictionary compatible with existing experiment infrastructure
    """
    
    config = {
        # Data parameters
        'window_size': window_size,
        'horizon': horizon,
        'delay': 0.05,
        'n_trajectory_points': 10,
        'n_channels': 4,
        'emg_preproc': emg_preproc,
        'subjects': subjects,
        'trials': trials,
        
        # Training parameters  
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': 1e-5,
        'max_epochs': 200,
        'patience': 20,
        'grad_clip': 1.0,
        
        # Scheduler parameters
        'scheduler': 'plateau',
        'lr_patience': 10,
        
        # Model parameters (TCANet-specific)
        'f1': 16,
        'pooling_size': 8,
        'drop_prob': 0.5,
        'tcn_depth': 3,
        'tcn_filters': 32,
        'transformer_heads': 4,
        'transformer_depth': 2,
    }
    
    return config


def main():
    """Example training script"""
    
    # Configuration
    dataset_path = "data/idms_ready_dataset.h5"
    
    config = create_experiment_config(
        window_size=1000,
        horizon=0.5,
        batch_size=32,
        learning_rate=0.001,
        emg_preproc=None,  # or 'denoise', 'bandpass', etc.
        subjects=None,  # Use all subjects
        trials=None     # Use all trials
    )
    
    # Create trainer
    trainer = TCANetIDMSTrainer(
        dataset_path=dataset_path,
        model_config=config,
        save_dir="pytorch_models/experiments",
        experiment_name=f"tcanet_idms_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Train model
    results = trainer.train(
        max_epochs=200,
        patience=20
    )
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {results['experiment_dir']}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.6f}")
    print(f"Test R²: {results['test_metrics']['r2']:.6f}")


if __name__ == "__main__":
    main()