#!/usr/bin/env python3
"""
Calculate baseline R² using average trajectory from training data as predictor.

This script:
1. Loads training data and calculates the average trajectory across all samples
2. Uses this average trajectory as a naive baseline predictor on test data
3. Calculates R² for comparison with trained models
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the data generator
from data_gen.idms_trajectory_datagenerator import IDMSTrajectoryDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineCalculator:
    """Calculate baseline R² using average trajectory predictor."""
    
    def __init__(self, dataset_path: str, experiment_config: Dict[str, Any]):
        """
        Initialize the baseline calculator.
        
        Args:
            dataset_path: Path to the HDF5 dataset
            experiment_config: Configuration from a representative experiment
        """
        self.dataset_path = Path(dataset_path)
        self.experiment_config = experiment_config
        
    def create_data_generator(self, split: str, shuffle: bool = False) -> IDMSTrajectoryDataGenerator:
        """
        Create a data generator for the specified split.
        
        Args:
            split: Data split ('train', 'val', 'test')
            shuffle: Whether to shuffle the data
            
        Returns:
            Configured data generator
        """
        params = self.experiment_config['key_parameters']
        
        generator = IDMSTrajectoryDataGenerator(
            dataset_path=str(self.dataset_path),
            subjects=None,  # Use all subjects
            trials=None,    # Use all trials
            window_size=params['window_size'],
            stride=50,      # Fixed stride
            delay=0.05,     # Fixed delay
            horizon=params['horizon'],
            n_trajectory_points=10,  # Fixed number of trajectory points
            batch_size=32,   # Fixed batch size
            shuffle=shuffle,
            emg_preproc=params.get('emg_preproc', None),
            split=split,
            seed=42         # Fixed seed for reproducibility
        )
        
        return generator
    
    def calculate_average_trajectory(self) -> np.ndarray:
        """
        Calculate the average trajectory across all available samples.
        
        Returns:
            Average trajectory shape: (n_trajectory_points,)
        """
        logger.info("Calculating average trajectory from ALL available data...")
        
        train_gen = self.create_data_generator('all', shuffle=False)
        
        all_trajectories = []
        n_batches = len(train_gen)
        
        for batch_idx in range(n_batches):
            try:
                X_batch, y_batch = train_gen[batch_idx]
                all_trajectories.append(y_batch)
                
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"  Processed {batch_idx + 1}/{n_batches} total batches")
                    
            except Exception as e:
                logger.error(f"Error processing training batch {batch_idx}: {e}")
                continue
        
        # Concatenate all trajectories and calculate mean
        all_trajectories = np.concatenate(all_trajectories, axis=0)
        avg_trajectory = np.mean(all_trajectories, axis=0)
        
        logger.info(f"Calculated average trajectory from {len(all_trajectories)} total samples (all data)")
        logger.info(f"Average trajectory shape: {avg_trajectory.shape}")
        logger.info(f"Average trajectory values: {avg_trajectory}")
        
        return avg_trajectory, all_trajectories
    
    def calculate_baseline_r2(self, avg_trajectory: np.ndarray) -> Dict[str, float]:
        """
        Calculate baseline R² using average trajectory as predictor on test data.
        
        Args:
            avg_trajectory: Average trajectory from training data
            
        Returns:
            Dictionary of baseline metrics
        """
        logger.info("Calculating baseline R² on test data...")
        
        test_gen = self.create_data_generator('test', shuffle=False)
        
        all_targets = []
        n_batches = len(test_gen)
        
        for batch_idx in range(n_batches):
            try:
                X_batch, y_batch = test_gen[batch_idx]
                all_targets.append(y_batch)
                
                if (batch_idx + 1) % 20 == 0:
                    logger.info(f"  Processed {batch_idx + 1}/{n_batches} test batches")
                    
            except Exception as e:
                logger.error(f"Error processing test batch {batch_idx}: {e}")
                continue
        
        # Concatenate all test targets
        all_targets = np.concatenate(all_targets, axis=0)
        n_test_samples = len(all_targets)
        
        # Create predictions using average trajectory (broadcast to all test samples)
        predictions = np.tile(avg_trajectory, (n_test_samples, 1))
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, all_targets)
        metrics['n_test_samples'] = n_test_samples
        
        logger.info(f"Baseline metrics calculated on {n_test_samples} test samples")
        logger.info(f"Baseline RMSE: {metrics['rmse']:.6f}")
        logger.info(f"Baseline MAE: {metrics['mae']:.6f}")
        logger.info(f"Baseline R²: {metrics['r2']:.6f}")
        
        return metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Baseline predictions (n_samples, n_trajectory_points)
            targets: Ground truth targets (n_samples, n_trajectory_points)
            
        Returns:
            Dictionary of metrics
        """
        # Flatten for overall metrics
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
            'mae': mean_absolute_error(target_flat, pred_flat),
            'mse': mean_squared_error(target_flat, pred_flat),
            'r2': 1 - (np.sum((target_flat - pred_flat) ** 2) / 
                      np.sum((target_flat - np.mean(target_flat)) ** 2))
        }
        
        # Calculate trajectory-point-wise metrics
        for t in range(predictions.shape[1]):
            pred_t = predictions[:, t]
            target_t = targets[:, t]
            
            metrics[f'rmse_t{t}'] = np.sqrt(mean_squared_error(target_t, pred_t))
            metrics[f'mae_t{t}'] = mean_absolute_error(target_t, pred_t)
            metrics[f'r2_t{t}'] = 1 - (np.sum((target_t - pred_t) ** 2) / 
                                      np.sum((target_t - np.mean(target_t)) ** 2))
        
        return metrics


def main():
    """Main function to calculate baseline R²."""
    
    # Load experiment configuration from summary
    experiment_summary_path = "Models/ElbowTrajectory/experiment_summary_20250825_145526.json"
    dataset_path = "data/idms_ready_dataset.h5"
    
    with open(experiment_summary_path, 'r') as f:
        experiment_summary = json.load(f)
    
    # Use the first successful experiment's configuration
    experiment_config = None
    for exp in experiment_summary['experiments']:
        if exp['success']:
            experiment_config = exp
            break
    
    if experiment_config is None:
        logger.error("No successful experiments found in summary!")
        return
    
    logger.info(f"Using configuration from experiment: {experiment_config['name']}")
    
    # Create baseline calculator
    calculator = BaselineCalculator(dataset_path, experiment_config)
    
    # Calculate average trajectory from training data
    avg_trajectory, all_training_trajectories = calculator.calculate_average_trajectory()
    
    # Calculate baseline R² on test data
    baseline_metrics = calculator.calculate_baseline_r2(avg_trajectory)
    
    # Print results
    print("\n" + "="*60)
    print("BASELINE R² CALCULATION RESULTS")
    print("="*60)
    print(f"Total samples used for average: {len(all_training_trajectories)}")
    print(f"Test samples evaluated: {baseline_metrics['n_test_samples']}")
    print(f"")
    print(f"Average trajectory: {avg_trajectory}")
    print(f"")
    print(f"📊 BASELINE METRICS:")
    print(f"   RMSE: {baseline_metrics['rmse']:.6f}")
    print(f"   MAE:  {baseline_metrics['mae']:.6f}")
    print(f"   R²:   {baseline_metrics['r2']:.6f}")
    print(f"")
    print("📈 Trajectory-wise R² scores:")
    for t in range(len(avg_trajectory)):
        print(f"   Point {t}: R² = {baseline_metrics[f'r2_t{t}']:.6f}")
    
    # Save results
    results = {
        'experiment_config': experiment_config['name'],
        'avg_trajectory': avg_trajectory.tolist(),
        'n_total_samples': len(all_training_trajectories),
        'baseline_metrics': baseline_metrics
    }
    
    output_file = f"baseline_r2_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Baseline results saved to: {output_file}")
    
    return baseline_metrics


if __name__ == "__main__":
    baseline_metrics = main()