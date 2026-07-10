#!/usr/bin/env python3
"""
Verify R² calculation from test_best_model.py

This script loads a model, makes predictions, and calculates R² using multiple methods
to verify the accuracy of the reported R² values.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Import the data generator
from data_gen.idms_trajectory_datagenerator import IDMSTrajectoryDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class R2Verifier:
    """Verify R² calculations using multiple methods."""
    
    def __init__(self, experiment_summary_path: str, dataset_path: str, models_base_dir: str):
        self.experiment_summary_path = Path(experiment_summary_path)
        self.dataset_path = Path(dataset_path)
        self.models_base_dir = Path(models_base_dir)
        
        # Load experiment summary
        with open(self.experiment_summary_path, 'r') as f:
            self.experiment_summary = json.load(f)
    
    def load_model(self, model_path: Path):
        """Load a trained model."""
        try:
            # Import custom objects
            from model_selection.modular_architectures import TrajEstimator, AdaptTrajEstimator
            from model_selection.uncertainty_trajectory_layers import JacobianTrajEstimator, AdaptJacobianTrajEstimator, UncertaintyTrajEstimator
            from model_selection.uncertainty_losses import aleatoric_trajectory_loss
            
            custom_objects = {
                'TrajEstimator': TrajEstimator,
                'AdaptTrajEstimator': AdaptTrajEstimator,
                'JacobianTrajEstimator': JacobianTrajEstimator,
                'AdaptJacobianTrajEstimator': AdaptJacobianTrajEstimator,
                'UncertaintyTrajEstimator': UncertaintyTrajEstimator,
                'aleatoric_trajectory_loss': aleatoric_trajectory_loss
            }
            
            model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def create_test_generator(self, experiment_config):
        """Create test data generator matching the experiment configuration."""
        params = experiment_config['key_parameters']
        
        test_gen = IDMSTrajectoryDataGenerator(
            dataset_path=str(self.dataset_path),
            subjects=None,
            trials=None,
            window_size=params['window_size'],
            stride=50,
            delay=0.05,
            horizon=params['horizon'],
            n_trajectory_points=10,
            batch_size=32,
            shuffle=False,
            emg_preproc=params.get('emg_preproc', None),
            split='test',
            test_ratio=0.05,  # Match training configuration
            val_ratio_from_trainval=0.2,
            seed=42
        )
        
        return test_gen
    
    def calculate_r2_multiple_methods(self, predictions: np.ndarray, targets: np.ndarray):
        """Calculate R² using multiple methods for verification."""
        
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Targets shape: {targets.shape}")
        logger.info(f"Predictions range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
        logger.info(f"Targets range: [{np.min(targets):.6f}, {np.max(targets):.6f}]")
        
        # Method 1: Exact replication of test_best_model.py
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        r2_original = 1 - (np.sum((target_flat - pred_flat) ** 2) / 
                          np.sum((target_flat - np.mean(target_flat)) ** 2))
        
        logger.info(f"Method 1 (Original): R² = {r2_original:.6f}")
        
        # Method 2: Using sklearn r2_score
        r2_sklearn = r2_score(target_flat, pred_flat)
        logger.info(f"Method 2 (Sklearn): R² = {r2_sklearn:.6f}")
        
        # Method 3: Manual calculation with detailed steps
        ss_res = np.sum((target_flat - pred_flat) ** 2)  # Sum of squared residuals
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)  # Total sum of squares
        r2_manual = 1 - (ss_res / ss_tot)
        
        logger.info(f"Method 3 (Manual): R² = {r2_manual:.6f}")
        logger.info(f"  SS_res (residuals): {ss_res:.2f}")
        logger.info(f"  SS_tot (total): {ss_tot:.2f}")
        logger.info(f"  Target mean: {np.mean(target_flat):.6f}")
        logger.info(f"  Target std: {np.std(target_flat):.6f}")
        
        # Method 4: Per-trajectory-point R²
        trajectory_r2s = []
        for t in range(predictions.shape[1]):
            pred_t = predictions[:, t]
            target_t = targets[:, t]
            r2_t = r2_score(target_t, pred_t)
            trajectory_r2s.append(r2_t)
            logger.info(f"  Point {t}: R² = {r2_t:.6f}")
        
        avg_trajectory_r2 = np.mean(trajectory_r2s)
        logger.info(f"Method 4 (Avg per-point): R² = {avg_trajectory_r2:.6f}")
        
        # Method 5: Check for potential issues
        logger.info("\nDiagnostic checks:")
        logger.info(f"  Any NaN in predictions: {np.any(np.isnan(predictions))}")
        logger.info(f"  Any NaN in targets: {np.any(np.isnan(targets))}")
        logger.info(f"  Any Inf in predictions: {np.any(np.isinf(predictions))}")
        logger.info(f"  Any Inf in targets: {np.any(np.isinf(targets))}")
        
        # Check if predictions are just predicting the mean
        pred_mean = np.mean(pred_flat)
        target_mean = np.mean(target_flat)
        logger.info(f"  Prediction mean: {pred_mean:.6f}")
        logger.info(f"  Target mean: {target_mean:.6f}")
        logger.info(f"  Mean difference: {abs(pred_mean - target_mean):.6f}")
        
        # Check prediction variance
        pred_var = np.var(pred_flat)
        target_var = np.var(target_flat)
        logger.info(f"  Prediction variance: {pred_var:.6f}")
        logger.info(f"  Target variance: {target_var:.6f}")
        
        # Method 6: Alternative R² formulation
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        r2_correlation = correlation ** 2
        logger.info(f"Method 5 (Correlation²): R² = {r2_correlation:.6f}")
        
        return {
            'r2_original': r2_original,
            'r2_sklearn': r2_sklearn,
            'r2_manual': r2_manual,
            'r2_avg_per_point': avg_trajectory_r2,
            'r2_correlation': r2_correlation,
            'ss_res': ss_res,
            'ss_tot': ss_tot,
            'trajectory_r2s': trajectory_r2s
        }
    
    def visualize_predictions_vs_targets(self, predictions: np.ndarray, targets: np.ndarray, 
                                       model_name: str, save_path: str = None):
        """Visualize predictions vs targets to understand the relationship."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Flatten for visualization
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Plot 1: Scatter plot of predictions vs targets
        # Sample for visualization if too many points
        n_points = min(len(pred_flat), 5000)
        indices = np.random.choice(len(pred_flat), n_points, replace=False)
        pred_sample = pred_flat[indices]
        target_sample = target_flat[indices]
        
        axes[0, 0].scatter(target_sample, pred_sample, alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(np.min(target_sample), np.min(pred_sample))
        max_val = max(np.max(target_sample), np.max(pred_sample))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
        
        axes[0, 0].set_xlabel('Target Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predictions vs Targets (Scatter)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals plot
        residuals = pred_flat - target_flat
        axes[0, 1].scatter(target_sample, residuals[indices], alpha=0.5, s=1)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Target Values')
        axes[0, 1].set_ylabel('Residuals (Pred - Target)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution comparison
        axes[1, 0].hist(target_flat, bins=50, alpha=0.7, label='Targets', density=True)
        axes[1, 0].hist(pred_flat, bins=50, alpha=0.7, label='Predictions', density=True)
        axes[1, 0].set_xlabel('Values')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Value Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Time series sample (first few samples)
        sample_size = min(200, predictions.shape[0])
        time_indices = np.arange(sample_size)
        
        # Show first trajectory point over time
        axes[1, 1].plot(time_indices, targets[:sample_size, 0], 'b-', alpha=0.7, label='Target (Point 0)')
        axes[1, 1].plot(time_indices, predictions[:sample_size, 0], 'r-', alpha=0.7, label='Prediction (Point 0)')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Trajectory Value')
        axes[1, 1].set_title('Time Series (First Trajectory Point)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Model: {model_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def verify_model_r2(self, model_name: str, n_batches: int = 50):
        """Verify R² calculation for a specific model."""
        
        # Find experiment
        experiment = None
        for exp in self.experiment_summary['experiments']:
            if exp['name'] == model_name:
                experiment = exp
                break
        
        if experiment is None:
            logger.error(f"Experiment {model_name} not found!")
            return None
        
        # Load model
        model_dir = self.models_base_dir / model_name
        model_path = model_dir / 'best_model.h5'
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        model = self.load_model(model_path)
        if model is None:
            return None
        
        logger.info(f"\nVerifying R² for model: {model_name}")
        logger.info(f"Model loaded: {model.name}")
        
        # Create test generator
        test_gen = self.create_test_generator(experiment)
        
        total_batches = min(len(test_gen), n_batches)
        logger.info(f"Processing {total_batches} batches from test set")
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        
        for batch_idx in range(total_batches):
            try:
                X_batch, y_batch = test_gen[batch_idx]
                
                # Make predictions
                y_pred = model.predict(X_batch, verbose=0)
                
                all_predictions.append(y_pred)
                all_targets.append(y_batch)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1}/{total_batches} batches")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        if not all_predictions:
            logger.error("No predictions generated!")
            return None
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        logger.info(f"Total samples: {len(predictions)}")
        
        # Calculate R² using multiple methods
        r2_results = self.calculate_r2_multiple_methods(predictions, targets)
        
        # Create visualization
        viz_path = f"r2_verification_{model_name}.png"
        self.visualize_predictions_vs_targets(predictions, targets, model_name, viz_path)
        
        return r2_results
    
    def calculate_average_trajectory_from_all_data(self, experiment_config):
        """Calculate average trajectory from all train/val/test data."""
        
        logger.info("Calculating average trajectory from all available data...")
        
        # Create data generator for all data (no split)
        params = experiment_config['key_parameters']
        
        # Create generators for train, val, and test to get all data
        train_gen = IDMSTrajectoryDataGenerator(
            dataset_path=str(self.dataset_path),
            subjects=None,
            trials=None,
            window_size=params['window_size'],
            stride=50,
            delay=0.05,
            horizon=params['horizon'],
            n_trajectory_points=10,
            batch_size=32,
            shuffle=False,
            emg_preproc=params.get('emg_preproc', None),
            split='train',
            test_ratio=0.05,
            val_ratio_from_trainval=0.2,
            seed=42
        )
        
        val_gen = IDMSTrajectoryDataGenerator(
            dataset_path=str(self.dataset_path),
            subjects=None,
            trials=None,
            window_size=params['window_size'],
            stride=50,
            delay=0.05,
            horizon=params['horizon'],
            n_trajectory_points=10,
            batch_size=32,
            shuffle=False,
            emg_preproc=params.get('emg_preproc', None),
            split='val',
            test_ratio=0.05,
            val_ratio_from_trainval=0.2,
            seed=42
        )
        
        test_gen = IDMSTrajectoryDataGenerator(
            dataset_path=str(self.dataset_path),
            subjects=None,
            trials=None,
            window_size=params['window_size'],
            stride=50,
            delay=0.05,
            horizon=params['horizon'],
            n_trajectory_points=10,
            batch_size=32,
            shuffle=False,
            emg_preproc=params.get('emg_preproc', None),
            split='test',
            test_ratio=0.05,
            val_ratio_from_trainval=0.2,
            seed=42
        )
        
        generators = [('train', train_gen), ('val', val_gen), ('test', test_gen)]
        
        total_batches = sum(len(gen) for _, gen in generators)
        logger.info(f"Collecting trajectories from {total_batches} batches across train/val/test...")
        
        all_trajectories = []
        processed_batches = 0
        
        for split_name, gen in generators:
            logger.info(f"Processing {split_name} split with {len(gen)} batches...")
            
            for batch_idx in range(len(gen)):
                try:
                    X_batch, y_batch = gen[batch_idx]
                    all_trajectories.append(y_batch)
                    processed_batches += 1
                    
                    if processed_batches % 100 == 0:
                        logger.info(f"  Processed {processed_batches}/{total_batches} batches")
                        
                except Exception as e:
                    logger.error(f"Error processing {split_name} batch {batch_idx}: {e}")
                    continue
        
        if not all_trajectories:
            logger.error("No trajectories collected!")
            return None
        
        # Calculate average trajectory
        all_trajectories = np.concatenate(all_trajectories, axis=0)
        avg_trajectory = np.mean(all_trajectories, axis=0)
        
        logger.info(f"Calculated average trajectory from {len(all_trajectories)} samples")
        logger.info(f"Average trajectory shape: {avg_trajectory.shape}")
        logger.info(f"Average trajectory values: {avg_trajectory}")
        
        return avg_trajectory, all_trajectories
    
    def calculate_baseline_r2(self, model_name: str, n_batches: int = 50):
        """Calculate baseline R² using average trajectory as naive predictor."""
        
        # Find experiment
        experiment = None
        for exp in self.experiment_summary['experiments']:
            if exp['name'] == model_name:
                experiment = exp
                break
        
        if experiment is None:
            logger.error(f"Experiment {model_name} not found!")
            return None
        
        logger.info(f"Calculating baseline R² for model configuration: {model_name}")
        
        # Calculate average trajectory from all data
        avg_trajectory, all_trajectories = self.calculate_average_trajectory_from_all_data(experiment)
        if avg_trajectory is None:
            return None
        
        # Create test generator for evaluation
        test_gen = self.create_test_generator(experiment)
        
        total_batches = min(len(test_gen), n_batches)
        logger.info(f"Evaluating baseline on {total_batches} test batches")
        
        # Collect test targets
        test_targets = []
        for batch_idx in range(total_batches):
            try:
                X_batch, y_batch = test_gen[batch_idx]
                test_targets.append(y_batch)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1}/{total_batches} test batches")
                    
            except Exception as e:
                logger.error(f"Error processing test batch {batch_idx}: {e}")
                continue
        
        if not test_targets:
            logger.error("No test targets collected!")
            return None
        
        # Concatenate test targets
        test_targets = np.concatenate(test_targets, axis=0)
        
        # Create baseline predictions (average trajectory repeated for each test sample)
        baseline_predictions = np.tile(avg_trajectory, (len(test_targets), 1))
        
        logger.info(f"Test samples: {len(test_targets)}")
        logger.info(f"Baseline predictions shape: {baseline_predictions.shape}")
        
        # Calculate baseline R² using the same method as model evaluation
        baseline_r2_results = self.calculate_r2_multiple_methods(baseline_predictions, test_targets)
        
        return {
            'avg_trajectory': avg_trajectory,
            'all_trajectories_stats': {
                'mean': np.mean(all_trajectories, axis=0),
                'std': np.std(all_trajectories, axis=0),
                'min': np.min(all_trajectories, axis=0),
                'max': np.max(all_trajectories, axis=0),
                'n_samples': len(all_trajectories)
            },
            'test_targets_stats': {
                'mean': np.mean(test_targets, axis=0),
                'std': np.std(test_targets, axis=0),
                'n_samples': len(test_targets)
            },
            'baseline_r2_results': baseline_r2_results
        }
    
    def compare_model_vs_baseline(self, model_name: str, n_batches: int = 50):
        """Compare model performance against baseline average trajectory predictor."""
        
        logger.info(f"Comparing model {model_name} vs baseline...")
        
        # Calculate model R²
        model_results = self.verify_model_r2(model_name, n_batches)
        if model_results is None:
            return None
        
        # Calculate baseline R²
        baseline_results = self.calculate_baseline_r2(model_name, n_batches)
        if baseline_results is None:
            return None
        
        # Compare results
        model_r2 = model_results['r2_sklearn']
        baseline_r2 = baseline_results['baseline_r2_results']['r2_sklearn']
        
        improvement = model_r2 - baseline_r2
        
        comparison_results = {
            'model_name': model_name,
            'model_r2': model_r2,
            'baseline_r2': baseline_r2,
            'improvement': improvement,
            'model_results': model_results,
            'baseline_results': baseline_results
        }
        
        # Print comparison
        print("\n" + "="*80)
        print("MODEL vs BASELINE COMPARISON")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Model R² (sklearn):    {model_r2:.6f}")
        print(f"Baseline R² (avg):     {baseline_r2:.6f}")
        print(f"Improvement:           {improvement:.6f}")
        
        if improvement > 0:
            print(f"✅ Model performs {improvement:.6f} R² points better than baseline")
        elif improvement < 0:
            print(f"❌ Model performs {abs(improvement):.6f} R² points worse than baseline")
        else:
            print("⚠️  Model performs exactly the same as baseline")
        
        # Additional statistics
        avg_traj = baseline_results['avg_trajectory']
        print(f"\nAverage trajectory statistics:")
        print(f"  Range: [{np.min(avg_traj):.6f}, {np.max(avg_traj):.6f}]")
        print(f"  Mean: {np.mean(avg_traj):.6f}")
        print(f"  Std: {np.std(avg_traj):.6f}")
        print(f"  Total samples used: {baseline_results['all_trajectories_stats']['n_samples']}")
        
        return comparison_results


def main():
    """Main function to verify R² calculations."""
    experiment_summary_path = "Models/ElbowTrajectory/experiment_summary_20250825_145526.json"
    dataset_path = "data/idms_ready_dataset.h5"
    models_base_dir = "Models/ElbowTrajectory"
    
    verifier = R2Verifier(
        experiment_summary_path=experiment_summary_path,
        dataset_path=dataset_path,
        models_base_dir=models_base_dir
    )
    
    # Verify the best model with baseline comparison
    model_to_check = "long_emg_short_horizon"  # Change this to check different models
    
    # Perform comprehensive model vs baseline comparison
    comparison_results = verifier.compare_model_vs_baseline(model_to_check, n_batches=50)
    
    if comparison_results:
        model_results = comparison_results['model_results']
        
        print("\n" + "="*60)
        print("R² VERIFICATION SUMMARY")
        print("="*60)
        print(f"Model: {model_to_check}")
        print(f"Original method (test_best_model.py): {model_results['r2_original']:.6f}")
        print(f"Sklearn r2_score:                    {model_results['r2_sklearn']:.6f}")
        print(f"Manual calculation:                  {model_results['r2_manual']:.6f}")
        print(f"Average per-trajectory-point:        {model_results['r2_avg_per_point']:.6f}")
        print(f"Correlation squared:                 {model_results['r2_correlation']:.6f}")
        print(f"\nSum of squared residuals:            {model_results['ss_res']:.2f}")
        print(f"Total sum of squares:                {model_results['ss_tot']:.2f}")
        
        # Check consistency
        methods = [model_results['r2_original'], model_results['r2_sklearn'], model_results['r2_manual']]
        if np.allclose(methods, methods[0], atol=1e-6):
            print("\n✅ All methods agree - R² calculation is consistent!")
        else:
            print("\n⚠️  Methods disagree - there may be an issue with the calculation!")
        
        return comparison_results
    else:
        return None


if __name__ == "__main__":
    main()