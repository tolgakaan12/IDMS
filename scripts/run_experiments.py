"""
PyTorch Experiment Runner for IDMS Trajectory Prediction

Integrates PyTorch TCANet-IDMS models with existing experiment infrastructure,
allowing direct comparison with TensorFlow models using the same evaluation framework.
"""

import sys
import os
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import logging

# Import PyTorch components
from pytorch_models.train_tcanet_idms import TCANetIDMSTrainer, create_experiment_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PyTorchExperimentRunner:
    """
    Experiment runner that integrates PyTorch models with existing infrastructure
    
    Provides similar interface to existing TensorFlow experiment management
    for seamless integration and comparison.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 models_base_dir: str = "Models/PyTorchTrajectory",
                 experiment_name: str = None):
        """
        Initialize PyTorch experiment runner
        
        Args:
            dataset_path: Path to IDMS dataset
            models_base_dir: Base directory for saving models
            experiment_name: Name for this experiment batch
        """
        self.dataset_path = Path(dataset_path)
        self.models_base_dir = Path(models_base_dir)
        
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"pytorch_experiments_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.models_base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.experiments = []
        self.results = []
        
        logger.info(f"Initialized PyTorch experiment runner: {experiment_name}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def create_experiment_grid(self) -> List[Dict[str, Any]]:
        """
        Create grid of experiment configurations to test different hyperparameters
        
        Returns:
            List of experiment configurations
        """
        
        # Base configuration
        base_config = {
            'window_size': 1000,
            'horizon': 0.5,
            'delay': 0.05,
            'n_trajectory_points': 10,
            'n_channels': 4,
            'batch_size': 32,
            'max_epochs': 200,
            'patience': 20,
        }
        
        # Experiment variations
        experiments = [
            # Experiment 1: Baseline TCANet-IDMS
            {
                **base_config,
                'name': 'tcanet_baseline',
                'description': 'Baseline TCANet-IDMS with standard parameters',
                'learning_rate': 0.001,
                'emg_preproc': None,
                'f1': 16,
                'tcn_depth': 3,
                'transformer_heads': 4,
                'transformer_depth': 2,
            },
            
            # Experiment 2: With EMG preprocessing
            {
                **base_config,
                'name': 'tcanet_preprocessed',
                'description': 'TCANet-IDMS with bandpass EMG preprocessing',
                'learning_rate': 0.001,
                'emg_preproc': 'bandpass',
                'f1': 16,
                'tcn_depth': 3,
                'transformer_heads': 4,
                'transformer_depth': 2,
            },
            
            # Experiment 3: Larger model
            {
                **base_config,
                'name': 'tcanet_large',
                'description': 'Larger TCANet-IDMS with more filters and deeper TCN',
                'learning_rate': 0.0008,
                'emg_preproc': None,
                'f1': 32,
                'tcn_depth': 4,
                'transformer_heads': 8,
                'transformer_depth': 3,
            },
            
            # Experiment 4: Different horizon
            {
                **base_config,
                'name': 'tcanet_long_horizon',
                'description': 'TCANet-IDMS with longer prediction horizon',
                'horizon': 0.8,
                'learning_rate': 0.001,
                'emg_preproc': None,
                'f1': 16,
                'tcn_depth': 3,
                'transformer_heads': 4,
                'transformer_depth': 2,
            },
            
            # Experiment 5: Smaller batch size, higher learning rate
            {
                **base_config,
                'name': 'tcanet_small_batch',
                'description': 'TCANet-IDMS with smaller batch size and higher LR',
                'batch_size': 16,
                'learning_rate': 0.002,
                'emg_preproc': None,
                'f1': 16,
                'tcn_depth': 3,
                'transformer_heads': 4,
                'transformer_depth': 2,
            },
        ]
        
        return experiments
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with given configuration
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment results
        """
        
        experiment_name = config['name']
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Description: {config['description']}")
        
        start_time = time.time()
        
        try:
            # Create trainer
            trainer = TCANetIDMSTrainer(
                dataset_path=str(self.dataset_path),
                model_config=config,
                save_dir=str(self.experiment_dir),
                experiment_name=experiment_name
            )
            
            # Train model
            results = trainer.train(
                max_epochs=config.get('max_epochs', 200),
                patience=config.get('patience', 20)
            )
            
            training_time = time.time() - start_time
            
            # Create experiment summary (compatible with TensorFlow format)
            experiment_summary = {
                'experiment_number': len(self.experiments) + 1,
                'name': experiment_name,
                'description': config['description'],
                'success': True,
                'start_time': datetime.datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'key_parameters': {
                    'window_size': config['window_size'],
                    'horizon': config['horizon'],
                    'batch_size': config['batch_size'],
                    'learning_rate': config['learning_rate'],
                    'emg_preproc': config.get('emg_preproc', None),
                    'framework': 'pytorch',
                    'model_type': 'tcanet_idms'
                },
                'results': {
                    'train_rmse': results['train_metrics']['rmse'],
                    'val_rmse': results['val_metrics']['rmse'], 
                    'test_rmse': results['test_metrics']['rmse'],
                    'test_mae': results['test_metrics']['mae'],
                    'test_r2': results['test_metrics']['r2'],
                    'epochs_trained': results['epochs_trained'],
                    'best_epoch': results['best_epoch']
                },
                'model_path': results['experiment_dir']
            }
            
            logger.info(f"✓ Experiment {experiment_name} completed successfully")
            logger.info(f"  Training time: {training_time:.1f}s")
            logger.info(f"  Test RMSE: {results['test_metrics']['rmse']:.6f}")
            logger.info(f"  Test R²: {results['test_metrics']['r2']:.6f}")
            
        except Exception as e:
            logger.error(f"✗ Experiment {experiment_name} failed: {e}")
            
            experiment_summary = {
                'experiment_number': len(self.experiments) + 1,
                'name': experiment_name,
                'description': config['description'],
                'success': False,
                'error': str(e),
                'start_time': datetime.datetime.now().isoformat(),
                'training_time_seconds': time.time() - start_time,
                'key_parameters': config,
                'results': {},
                'model_path': None
            }
        
        return experiment_summary
    
    def run_experiments(self, 
                       experiment_configs: List[Dict[str, Any]] = None,
                       max_experiments: int = None) -> List[Dict[str, Any]]:
        """
        Run multiple experiments
        
        Args:
            experiment_configs: List of configurations (None = use default grid)
            max_experiments: Maximum number of experiments to run
            
        Returns:
            List of experiment results
        """
        
        if experiment_configs is None:
            experiment_configs = self.create_experiment_grid()
        
        if max_experiments:
            experiment_configs = experiment_configs[:max_experiments]
        
        logger.info(f"Running {len(experiment_configs)} PyTorch experiments...")
        
        all_start_time = time.time()
        
        for i, config in enumerate(experiment_configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i+1}/{len(experiment_configs)}")
            logger.info(f"{'='*60}")
            
            # Run experiment
            result = self.run_single_experiment(config)
            
            # Store results
            self.experiments.append(result)
            self.results.append(result)
        
        total_time = time.time() - all_start_time
        successful_experiments = sum(1 for exp in self.experiments if exp['success'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"All experiments completed!")
        logger.info(f"  Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
        logger.info(f"  Successful: {successful_experiments}/{len(experiment_configs)}")
        logger.info(f"{'='*60}")
        
        # Save experiment summary
        self.save_experiment_summary()
        
        return self.results
    
    def save_experiment_summary(self):
        """Save experiment summary in format compatible with TensorFlow experiments"""
        
        summary = {
            'experiment_batch_name': self.experiment_name,
            'dataset_path': str(self.dataset_path),
            'total_experiments': len(self.experiments),
            'successful_experiments': sum(1 for exp in self.experiments if exp['success']),
            'framework': 'pytorch',
            'model_type': 'tcanet_idms',
            'created_at': datetime.datetime.now().isoformat(),
            'experiments': self.experiments
        }
        
        # Save JSON summary
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment summary saved: {summary_path}")
        
        # Save results CSV for easy analysis
        results_data = []
        for exp in self.experiments:
            if exp['success']:
                row = {
                    'experiment_name': exp['name'],
                    'description': exp['description'],
                    'window_size': exp['key_parameters'].get('window_size'),
                    'horizon': exp['key_parameters'].get('horizon'),
                    'batch_size': exp['key_parameters'].get('batch_size'),
                    'learning_rate': exp['key_parameters'].get('learning_rate'),
                    'emg_preproc': exp['key_parameters'].get('emg_preproc'),
                    'train_rmse': exp['results'].get('train_rmse'),
                    'val_rmse': exp['results'].get('val_rmse'),
                    'test_rmse': exp['results'].get('test_rmse'),
                    'test_mae': exp['results'].get('test_mae'),
                    'test_r2': exp['results'].get('test_r2'),
                    'epochs_trained': exp['results'].get('epochs_trained'),
                    'training_time_s': exp.get('training_time_seconds'),
                    'framework': 'pytorch'
                }
                results_data.append(row)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            results_csv_path = self.experiment_dir / "results_summary.csv"
            results_df.to_csv(results_csv_path, index=False)
            logger.info(f"Results CSV saved: {results_csv_path}")
            
            # Print summary statistics
            if len(results_data) > 1:
                logger.info("\nResults Summary:")
                logger.info(f"  Best Test RMSE: {results_df['test_rmse'].min():.6f} ({results_df.loc[results_df['test_rmse'].idxmin(), 'experiment_name']})")
                logger.info(f"  Best Test R²: {results_df['test_r2'].max():.6f} ({results_df.loc[results_df['test_r2'].idxmax(), 'experiment_name']})")
                logger.info(f"  Mean Test RMSE: {results_df['test_rmse'].mean():.6f} ± {results_df['test_rmse'].std():.6f}")
                logger.info(f"  Mean Test R²: {results_df['test_r2'].mean():.6f} ± {results_df['test_r2'].std():.6f}")
    
    def compare_with_tensorflow_results(self, tensorflow_summary_path: str):
        """
        Compare PyTorch results with TensorFlow experiment results
        
        Args:
            tensorflow_summary_path: Path to TensorFlow experiment summary JSON
        """
        
        try:
            # Load TensorFlow results
            with open(tensorflow_summary_path, 'r') as f:
                tf_summary = json.load(f)
            
            # Extract TensorFlow results
            tf_results = []
            for exp in tf_summary['experiments']:
                if exp['success']:
                    # Assuming similar result structure - adapt as needed
                    tf_results.append({
                        'experiment_name': exp['name'],
                        'framework': 'tensorflow',
                        'test_rmse': exp.get('test_rmse', np.nan),  # Adapt field names
                        'test_r2': exp.get('test_r2', np.nan),
                    })
            
            # Extract PyTorch results
            pytorch_results = []
            for exp in self.experiments:
                if exp['success']:
                    pytorch_results.append({
                        'experiment_name': exp['name'],
                        'framework': 'pytorch',
                        'test_rmse': exp['results']['test_rmse'],
                        'test_r2': exp['results']['test_r2'],
                    })
            
            # Combine and save comparison
            all_results = tf_results + pytorch_results
            comparison_df = pd.DataFrame(all_results)
            
            comparison_path = self.experiment_dir / "framework_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            
            # Print comparison summary
            logger.info("\nFramework Comparison:")
            tf_mean_rmse = comparison_df[comparison_df['framework'] == 'tensorflow']['test_rmse'].mean()
            pt_mean_rmse = comparison_df[comparison_df['framework'] == 'pytorch']['test_rmse'].mean()
            tf_mean_r2 = comparison_df[comparison_df['framework'] == 'tensorflow']['test_r2'].mean()
            pt_mean_r2 = comparison_df[comparison_df['framework'] == 'pytorch']['test_r2'].mean()
            
            logger.info(f"  TensorFlow - Mean RMSE: {tf_mean_rmse:.6f}, Mean R²: {tf_mean_r2:.6f}")
            logger.info(f"  PyTorch    - Mean RMSE: {pt_mean_rmse:.6f}, Mean R²: {pt_mean_r2:.6f}")
            
            logger.info(f"Comparison saved: {comparison_path}")
            
        except Exception as e:
            logger.warning(f"Could not compare with TensorFlow results: {e}")


def main():
    """Main experiment runner"""
    
    # Configuration
    dataset_path = "data/idms_ready_dataset.h5"
    models_base_dir = "Models/PyTorchTrajectory"
    
    # Create experiment runner
    runner = PyTorchExperimentRunner(
        dataset_path=dataset_path,
        models_base_dir=models_base_dir,
        experiment_name=f"pytorch_tcanet_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run experiments
    results = runner.run_experiments(
        max_experiments=3  # Limit for testing - remove to run all
    )
    
    # Optional: Compare with existing TensorFlow results
    tensorflow_summary = "Models/ElbowTrajectory/experiment_summary_20250825_145526.json"
    if os.path.exists(tensorflow_summary):
        runner.compare_with_tensorflow_results(tensorflow_summary)
    
    print(f"\n🎉 PyTorch experiments completed!")
    print(f"Results saved in: {runner.experiment_dir}")
    print(f"Successful experiments: {sum(1 for r in results if r['success'])}/{len(results)}")


if __name__ == "__main__":
    main()