#!/usr/bin/env python3
"""
Compare Training Strategies for IDMS Trajectory Prediction

Strategy A: Pre-train on all subjects → Fine-tune on target subject
Strategy B: Train directly on single target subject

This script evaluates which approach yields better performance.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Import components
from pytorch_models.tcanet_idms import create_tcanet_idms_model
from pytorch_models.pytorch_data_adapter import create_idms_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingStrategyComparator:
    """Compare different training strategies for IDMS trajectory prediction."""
    
    def __init__(self, 
                 dataset_path: str = "data/idms_ready_dataset.h5",
                 target_subject: str = "subject_003",
                 pretrained_model_path: str = None,
                 save_dir: str = "training_strategy_comparison"):
        """
        Initialize comparator.
        
        Args:
            dataset_path: Path to dataset
            target_subject: Subject to compare strategies on
            pretrained_model_path: Path to pre-trained all-subjects model
            save_dir: Directory to save results
        """
        self.dataset_path = dataset_path
        self.target_subject = target_subject
        self.pretrained_model_path = pretrained_model_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Model config (matching successful experiments)
        self.model_config = {
            'window_size': 1000,
            'horizon': 0.25,
            'delay': 0.05,
            'n_trajectory_points': 10,
            'n_channels': 4,
            'batch_size': 128,
            'learning_rate': 0.002,
            'weight_decay': 1e-5,
            'max_epochs': 50,  # Shorter for comparison
            'patience': 10,
            'grad_clip': 1.0
        }
        
        logger.info(f"Comparing strategies for target subject: {target_subject}")
        logger.info(f"Results will be saved to: {self.save_dir}")
    
    def strategy_a_pretrain_finetune(self) -> Dict:
        """Strategy A: Load pre-trained all-subjects model and fine-tune on target subject."""
        
        logger.info("="*60)
        logger.info("STRATEGY A: PRE-TRAIN + FINE-TUNE")
        logger.info("="*60)
        
        if not self.pretrained_model_path or not Path(self.pretrained_model_path).exists():
            logger.error(f"Pre-trained model not found at: {self.pretrained_model_path}")
            return {}
        
        # Load pre-trained model
        logger.info("Loading pre-trained model...")
        model = create_tcanet_idms_model(
            window_size=self.model_config['window_size'],
            n_channels=self.model_config['n_channels'],
            trajectory_points=self.model_config['n_trajectory_points'],
            trajectory_horizon=self.model_config['horizon'],
            trajectory_delay=self.model_config['delay']
        )
        
        # Load pre-trained weights
        pretrained_path = Path(self.pretrained_model_path)
        if pretrained_path.is_file():
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        else:
            # Assume it's a directory with best_model.pt
            checkpoint = torch.load(pretrained_path / "best_model.pt", map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded pre-trained state dict")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded pre-trained model")
        
        # Create target subject data loaders
        logger.info(f"Creating data loaders for {self.target_subject}...")
        train_loader, val_loader, test_loader = create_idms_dataloaders(
            dataset_path=self.dataset_path,
            subjects=[self.target_subject],
            window_size=self.model_config['window_size'],
            horizon=self.model_config['horizon'],
            batch_size=self.model_config['batch_size']
        )
        
        logger.info(f"Fine-tuning data: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
        
        # Fine-tune with lower learning rate
        optimizer = optim.Adam(model.parameters(), 
                              lr=self.model_config['learning_rate'] * 0.1,  # 10x lower LR for fine-tuning
                              weight_decay=self.model_config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        # Fine-tune for fewer epochs
        max_epochs = self.model_config['max_epochs'] // 2  # Half the epochs
        
        logger.info(f"Fine-tuning for {max_epochs} epochs with LR={self.model_config['learning_rate'] * 0.1:.5f}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.model_config['grad_clip'])
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_losses.append(loss.item())
                    all_preds.extend(y_pred.cpu().numpy().flatten())
                    all_targets.extend(y_batch.cpu().numpy().flatten())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_r2 = r2_score(all_targets, all_preds)
            
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_r2'].append(val_r2)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:2d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val R²={val_r2:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.model_config['patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        test_r2 = self._evaluate_model(model, test_loader)
        
        results = {
            'strategy': 'pretrain_finetune',
            'test_r2': test_r2,
            'best_val_r2': max(training_history['val_r2']),
            'final_epoch': len(training_history['train_loss']),
            'training_history': training_history
        }
        
        # Save model
        torch.save(model.state_dict(), self.save_dir / f"strategy_a_{self.target_subject}.pt")
        logger.info(f"Strategy A Test R²: {test_r2:.6f}")
        
        return results
    
    def strategy_b_single_subject(self) -> Dict:
        """Strategy B: Train from scratch on single target subject."""
        
        logger.info("="*60)
        logger.info("STRATEGY B: SINGLE SUBJECT FROM SCRATCH")
        logger.info("="*60)
        
        # Create fresh model
        model = create_tcanet_idms_model(
            window_size=self.model_config['window_size'],
            n_channels=self.model_config['n_channels'],
            trajectory_points=self.model_config['n_trajectory_points'],
            trajectory_horizon=self.model_config['horizon'],
            trajectory_delay=self.model_config['delay']
        )
        
        # Create target subject data loaders
        logger.info(f"Creating data loaders for {self.target_subject}...")
        train_loader, val_loader, test_loader = create_idms_dataloaders(
            dataset_path=self.dataset_path,
            subjects=[self.target_subject],
            window_size=self.model_config['window_size'],
            horizon=self.model_config['horizon'],
            batch_size=self.model_config['batch_size']
        )
        
        logger.info(f"Single-subject data: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
        
        # Train from scratch with full learning rate
        optimizer = optim.Adam(model.parameters(), 
                              lr=self.model_config['learning_rate'],
                              weight_decay=self.model_config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        logger.info(f"Training from scratch for {self.model_config['max_epochs']} epochs with LR={self.model_config['learning_rate']:.5f}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        
        for epoch in range(self.model_config['max_epochs']):
            # Training
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.model_config['grad_clip'])
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_losses.append(loss.item())
                    all_preds.extend(y_pred.cpu().numpy().flatten())
                    all_targets.extend(y_batch.cpu().numpy().flatten())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_r2 = r2_score(all_targets, all_preds)
            
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_r2'].append(val_r2)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:2d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val R²={val_r2:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.model_config['patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        test_r2 = self._evaluate_model(model, test_loader)
        
        results = {
            'strategy': 'single_subject',
            'test_r2': test_r2,
            'best_val_r2': max(training_history['val_r2']),
            'final_epoch': len(training_history['train_loss']),
            'training_history': training_history
        }
        
        # Save model
        torch.save(model.state_dict(), self.save_dir / f"strategy_b_{self.target_subject}.pt")
        logger.info(f"Strategy B Test R²: {test_r2:.6f}")
        
        return results
    
    def _evaluate_model(self, model, test_loader) -> float:
        """Evaluate model on test set and return R²."""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                all_preds.extend(y_pred.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
        
        return r2_score(all_targets, all_preds)
    
    def compare_strategies(self) -> Dict:
        """Compare both strategies and return results."""
        
        logger.info("🚀 TRAINING STRATEGY COMPARISON STARTED")
        logger.info(f"Target Subject: {self.target_subject}")
        logger.info(f"Pre-trained Model: {self.pretrained_model_path}")
        
        results = {}
        
        # Run Strategy A (if pre-trained model available)
        if self.pretrained_model_path and Path(self.pretrained_model_path).exists():
            results['strategy_a'] = self.strategy_a_pretrain_finetune()
        else:
            logger.warning("Pre-trained model not available, skipping Strategy A")
            results['strategy_a'] = {}
        
        # Run Strategy B
        results['strategy_b'] = self.strategy_b_single_subject()
        
        # Compare results
        self._plot_comparison(results)
        
        # Save results
        with open(self.save_dir / "strategy_comparison.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_comparison(self, results: Dict):
        """Plot training comparison."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Strategy Comparison: {self.target_subject}', fontsize=16, fontweight='bold')
        
        strategies = ['strategy_a', 'strategy_b']
        strategy_names = ['Pre-train + Fine-tune', 'Single Subject']
        colors = ['blue', 'red']
        
        # Plot 1: Training Loss
        ax1 = axes[0, 0]
        for i, (strategy, name, color) in enumerate(zip(strategies, strategy_names, colors)):
            if strategy in results and results[strategy]:
                history = results[strategy]['training_history']
                epochs = range(1, len(history['train_loss']) + 1)
                ax1.plot(epochs, history['train_loss'], color=color, label=f'{name}', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Validation Loss
        ax2 = axes[0, 1]
        for i, (strategy, name, color) in enumerate(zip(strategies, strategy_names, colors)):
            if strategy in results and results[strategy]:
                history = results[strategy]['training_history']
                epochs = range(1, len(history['val_loss']) + 1)
                ax2.plot(epochs, history['val_loss'], color=color, label=f'{name}', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation R²
        ax3 = axes[1, 0]
        for i, (strategy, name, color) in enumerate(zip(strategies, strategy_names, colors)):
            if strategy in results and results[strategy]:
                history = results[strategy]['training_history']
                epochs = range(1, len(history['val_r2']) + 1)
                ax3.plot(epochs, history['val_r2'], color=color, label=f'{name}', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation R²')
        ax3.set_title('Validation R² Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final Test R² Bar Chart
        ax4 = axes[1, 1]
        test_r2_values = []
        labels = []
        bar_colors = []
        
        for i, (strategy, name, color) in enumerate(zip(strategies, strategy_names, colors)):
            if strategy in results and results[strategy]:
                test_r2_values.append(results[strategy]['test_r2'])
                labels.append(name)
                bar_colors.append(color)
        
        if test_r2_values:
            bars = ax4.bar(labels, test_r2_values, color=bar_colors, alpha=0.7)
            ax4.set_ylabel('Test R²')
            ax4.set_title('Final Test Performance')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, test_r2_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = self.save_dir / f"strategy_comparison_{self.target_subject}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {plot_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("STRATEGY COMPARISON SUMMARY")
        logger.info("="*60)
        
        for strategy, name in zip(strategies, strategy_names):
            if strategy in results and results[strategy]:
                result = results[strategy]
                logger.info(f"{name}:")
                logger.info(f"  Test R²: {result['test_r2']:.6f}")
                logger.info(f"  Best Val R²: {result['best_val_r2']:.6f}")
                logger.info(f"  Training Epochs: {result['final_epoch']}")
        
        if len(test_r2_values) == 2:
            improvement = test_r2_values[0] - test_r2_values[1] if 'strategy_a' in results and results['strategy_a'] else 0
            if improvement > 0:
                logger.info(f"✅ Pre-training + Fine-tuning wins by {improvement:.4f} R² points!")
            elif improvement < 0:
                logger.info(f"✅ Single subject training wins by {abs(improvement):.4f} R² points!")
            else:
                logger.info("🤝 Both strategies perform equally well!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare training strategies')
    parser.add_argument('--pretrained-model', required=True,
                       help='Path to pre-trained all-subjects model (file or directory)')
    parser.add_argument('--target-subject', default='subject_003',
                       help='Subject to compare strategies on')
    parser.add_argument('--dataset', default='data/idms_ready_dataset.h5',
                       help='Path to dataset')
    parser.add_argument('--save-dir', default='training_strategy_comparison',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    comparator = TrainingStrategyComparator(
        dataset_path=args.dataset,
        target_subject=args.target_subject,
        pretrained_model_path=args.pretrained_model,
        save_dir=args.save_dir
    )
    
    results = comparator.compare_strategies()
    logger.info("🎯 Strategy comparison complete!")


if __name__ == "__main__":
    main()