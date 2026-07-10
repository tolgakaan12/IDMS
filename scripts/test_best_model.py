#!/usr/bin/env python3
"""
Test all trained models on the test set and identify the best performing one.

This script:
1. Loads all models from the experiment summary
2. Creates a test data generator 
3. Evaluates each model on the test set
4. Reports performance metrics and identifies the best model
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the data generator
from idms.data.generator import IDMSTrajectoryDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate multiple trained models on the test set."""
    
    def __init__(self, 
                 experiment_summary_path: str,
                 dataset_path: str,
                 models_base_dir: str):
        """
        Initialize the model evaluator.
        
        Args:
            experiment_summary_path: Path to experiment summary JSON
            dataset_path: Path to the HDF5 dataset
            models_base_dir: Base directory containing all model folders
        """
        self.experiment_summary_path = Path(experiment_summary_path)
        self.dataset_path = Path(dataset_path)
        self.models_base_dir = Path(models_base_dir)
        
        # Load experiment summary
        with open(self.experiment_summary_path, 'r') as f:
            self.experiment_summary = json.load(f)
        
        self.results = []
        
    def create_test_generator(self, experiment_config: Dict[str, Any]) -> IDMSTrajectoryDataGenerator:
        """
        Create a test data generator based on experiment configuration.
        
        Args:
            experiment_config: Configuration from experiment summary
            
        Returns:
            Configured test data generator
        """
        params = experiment_config['key_parameters']
        
        # Create test generator with same parameters as training
        test_gen = IDMSTrajectoryDataGenerator(
            dataset_path=str(self.dataset_path),
            subjects=None,  # Use all subjects
            trials=None,    # Use all trials
            window_size=params['window_size'],
            stride=50,      # Fixed stride for testing
            delay=0.05,     # Fixed delay for testing
            horizon=params['horizon'],
            n_trajectory_points=10,  # Fixed number of trajectory points
            batch_size=32,   # Fixed batch size for testing
            shuffle=False,   # Don't shuffle test data
            emg_preproc=params.get('emg_preproc', None),
            split='test',    # Use test split
            seed=42         # Fixed seed for reproducibility
        )
        
        return test_gen
    
    def load_model(self, model_path: Path) -> tf.keras.Model:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded Keras model
        """
        try:
            # Import custom objects
            from idms.uncertainty.blocks import TrajEstimator, AdaptTrajEstimator
            from idms.uncertainty.jacobian_layers import JacobianTrajEstimator, AdaptJacobianTrajEstimator, UncertaintyTrajEstimator
            from idms.uncertainty.losses import aleatoric_trajectory_loss
            
            # Create custom objects dictionary
            custom_objects = {
                'TrajEstimator': TrajEstimator,
                'AdaptTrajEstimator': AdaptTrajEstimator,
                'JacobianTrajEstimator': JacobianTrajEstimator,
                'AdaptJacobianTrajEstimator': AdaptJacobianTrajEstimator,
                'UncertaintyTrajEstimator': UncertaintyTrajEstimator,
                'aleatoric_trajectory_loss': aleatoric_trajectory_loss
            }
            
            # Load model with custom objects
            model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
            return model
        except Exception as e:
            logger.warning(f"Failed to load model {model_path}: {e}")
            return None
    
    def evaluate_model(self, 
                      model: tf.keras.Model, 
                      test_generator: IDMSTrajectoryDataGenerator,
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model on the test set.
        
        Args:
            model: Trained Keras model
            test_generator: Test data generator
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        all_predictions = []
        all_targets = []
        
        # Evaluate on all test batches
        n_batches = len(test_generator)
        for batch_idx in range(n_batches):
            try:
                X_batch, y_batch = test_generator[batch_idx]
                
                # Make predictions
                y_pred = model.predict(X_batch, verbose=0)
                
                # Store for metric calculation
                all_predictions.append(y_pred)
                all_targets.append(y_batch)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1}/{n_batches} batches")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        if not all_predictions:
            logger.error(f"No predictions generated for model {model_name}")
            return {}
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        metrics['model_name'] = model_name
        metrics['n_samples'] = len(predictions)
        
        logger.info(f"  Model {model_name} - RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")
        
        return metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions (n_samples, n_trajectory_points)
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
        
        return metrics
    
    def run_evaluation(self) -> List[Dict[str, Any]]:
        """
        Run evaluation on all models from the experiment summary.
        
        Returns:
            List of evaluation results
        """
        logger.info(f"Starting evaluation of {len(self.experiment_summary['experiments'])} models")
        
        results = []
        
        for experiment in self.experiment_summary['experiments']:
            if not experiment['success']:
                logger.info(f"Skipping failed experiment: {experiment['name']}")
                continue
            
            model_name = experiment['name']
            model_dir = self.models_base_dir / model_name
            model_path = model_dir / 'best_model.h5'
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                continue
            
            # Create test generator for this experiment's configuration
            try:
                test_gen = self.create_test_generator(experiment)
                logger.info(f"Created test generator with {len(test_gen)} batches")
            except Exception as e:
                logger.error(f"Failed to create test generator for {model_name}: {e}")
                continue
            
            # Load and evaluate model
            model = self.load_model(model_path)
            if model is None:
                continue
            
            try:
                metrics = self.evaluate_model(model, test_gen, model_name)
                if metrics:
                    # Add experiment configuration to results
                    result = {
                        'experiment_name': model_name,
                        'experiment_number': experiment['experiment_number'],
                        'description': experiment['description'],
                        'key_parameters': experiment['key_parameters'],
                        **metrics
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}")
                continue
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
        
        self.results = results
        return results
    
    def find_best_model(self, metric: str = 'rmse') -> Dict[str, Any]:
        """
        Find the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for ranking (lower is better for RMSE/MAE)
            
        Returns:
            Best model result dictionary
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluation first.")
        
        # For RMSE and MAE, lower is better
        if metric in ['rmse', 'mae', 'mse']:
            best_model = min(self.results, key=lambda x: x[metric])
        else:
            # For R2, higher is better
            best_model = max(self.results, key=lambda x: x[metric])
        
        return best_model
    
    def save_results(self, output_path: str):
        """
        Save evaluation results to CSV and JSON files.
        
        Args:
            output_path: Output file path (without extension)
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_path = f"{output_path}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        
        # Save to JSON for complete information
        json_path = f"{output_path}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to: {json_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Sort by RMSE
        df_sorted = df.sort_values('rmse')
        
        # Print summary table
        summary_cols = ['experiment_name', 'rmse', 'mae', 'r2', 'n_samples']
        print(df_sorted[summary_cols].to_string(index=False, float_format='%.6f'))
        
        # Find and highlight best model
        best_model = self.find_best_model('rmse')
        print(f"\n🏆 BEST MODEL: {best_model['experiment_name']}")
        print(f"   Description: {best_model['description']}")
        print(f"   RMSE: {best_model['rmse']:.6f}")
        print(f"   MAE: {best_model['mae']:.6f}")
        print(f"   R²: {best_model['r2']:.6f}")
        print(f"   Test samples: {best_model['n_samples']}")
        
        return best_model


def main():
    """Main function to run the evaluation."""
    # Configuration
    experiment_summary_path = "Models/ElbowTrajectory/experiment_summary_20250825_145526.json"
    dataset_path = "data/idms_ready_dataset.h5"
    models_base_dir = "Models/ElbowTrajectory"
    
    # Create evaluator
    evaluator = ModelEvaluator(
        experiment_summary_path=experiment_summary_path,
        dataset_path=dataset_path,
        models_base_dir=models_base_dir
    )
    
    # Run evaluation
    logger.info("Starting model evaluation...")
    results = evaluator.run_evaluation()
    
    if not results:
        logger.error("No models were successfully evaluated!")
        return
    
    # Save results
    output_path = f"model_evaluation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    best_model = evaluator.save_results(output_path)
    
    logger.info("Evaluation completed!")
    return best_model


if __name__ == "__main__":
    best_model = main()