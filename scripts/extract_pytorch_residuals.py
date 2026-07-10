#!/usr/bin/env python3
"""
PyTorch Elbow Model Residual Extractor
======================================

Extracts residuals from the PyTorch elbow trajectory model by comparing
predicted v0 (velocity parameter) with ground truth velocities from the dataset.

Key steps:
1. Load finetuned subject 005 model
2. Run inference to get v0 predictions (penultimate layer)
3. Load corresponding ground truth velocities from IDMS dataset
4. Handle timing alignment (model predicts at t+delay)
5. Calculate residuals: true_v0 - predicted_v0
6. Save residuals with trial metadata (train/val/test splits)

Date: 2025-01-15
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch_models'))

import h5py
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
# import pandas as pd  # Will use csv module instead
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch model and data utilities
from idms.estimator.models.tcanet import create_tcanet_idms_model
from idms.data.generator import IDMSTrajectoryDataGenerator


class PyTorchResidualExtractor:
    """
    Extracts residuals from PyTorch elbow trajectory model.
    """
    
    def __init__(self, 
                 model_path: str,
                 dataset_path: str = "data/idms_ready_dataset.h5"):
        """
        Initialize residual extractor.
        
        Args:
            model_path: Path to model directory containing best_model.pt and config.json
            dataset_path: Path to IDMS dataset HDF5 file
        """
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        
        # Load model configuration
        config_path = self.model_path / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load data splits
        splits_path = self.model_path / "data_splits.json"
        with open(splits_path, 'r') as f:
            self.data_splits = json.load(f)
            
        # Initialize model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initialized extractor for model: {model_path}")
        print(f"Config: {self.config}")
        print(f"Device: {self.device}")
        
    def load_model(self) -> None:
        """Load the trained PyTorch model."""
        print("Loading PyTorch model...")
        
        # Create model architecture using the correct function
        self.model = create_tcanet_idms_model(
            window_size=self.config['window_size'],
            n_channels=self.config.get('n_channels', len(self.config['emg_channels'])),
            trajectory_points=self.config['n_trajectory_points'],
            trajectory_horizon=self.config['horizon'],
            trajectory_delay=self.config['delay']
        ).to(self.device)
        
        # Load trained weights
        model_checkpoint = torch.load(
            self.model_path / "best_model.pt", 
            map_location=self.device,
            weights_only=False
        )
        
        # Handle checkpoint format (from visualize_pytorch_predictions.py)
        if isinstance(model_checkpoint, dict) and 'model_state_dict' in model_checkpoint:
            # Full checkpoint format
            state_dict = model_checkpoint['model_state_dict']
            print(f"Loaded checkpoint from epoch {model_checkpoint.get('epoch', 'unknown')}")
        else:
            # Direct state dict format
            state_dict = model_checkpoint
            
        # Clean state dict keys (handle _orig_mod. prefixes from compiled models)
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace('_orig_mod.', '')
            clean_state_dict[clean_key] = value
            
        self.model.load_state_dict(clean_state_dict)
            
        self.model.eval()
        
        print(f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
    def get_trial_split(self, trial_name: str) -> str:
        """Get the data split (train/val/test) for a trial."""
        splits = self.data_splits['data_splits']
        
        if trial_name in splits['test_trials']:
            return 'test'
        elif trial_name in splits['val_trials']:
            return 'val'
        elif trial_name in splits['train_trials']:
            return 'train'
        else:
            return 'unknown'
            
    def extract_v0_predictions(self, emg_windows: np.ndarray) -> np.ndarray:
        """
        Extract v0 predictions using first trajectory point / delta_t approach.
        
        Args:
            emg_windows: EMG data windows (N, 1, 4, window_size)
            
        Returns:
            v0_predictions: Velocity predictions (N,) in degrees/second
        """
        if self.model is None:
            self.load_model()
            
        self.model.eval()
        
        with torch.no_grad():
            # Convert to torch tensor
            emg_tensor = torch.tensor(emg_windows, dtype=torch.float32).to(self.device)
            
            # Get full model output (trajectory predictions)
            trajectory_output = self.model(emg_tensor)  # Shape: (N, n_trajectory_points)
            
            # Calculate trajectory times (same as visualize_pytorch_velocity_predictions.py)
            trajectory_times = np.linspace(
                self.config['delay'], 
                self.config['delay'] + self.config['horizon'], 
                self.config['n_trajectory_points']
            )
            
            # Use first trajectory point with its actual time (not spacing!)
            dt = trajectory_times[0]  # Time from current to first prediction point (should be delay)
            
            # Extract v0 using first trajectory point / delta_t
            # This gives instantaneous velocity at the first prediction time
            first_trajectory_point = trajectory_output[:, 0].cpu().numpy()  # (N,)
            v0_predictions = first_trajectory_point / dt
            
        return v0_predictions
    
    def load_ground_truth_velocity(self, subject: str, trial: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ground truth elbow velocity from dataset.
        
        Args:
            subject: Subject ID (e.g., 'subject_005')
            trial: Trial ID (e.g., 'trial_001')
            
        Returns:
            velocity: Ground truth elbow velocity (N_samples,)
            time: Time vector (N_samples,)
        """
        with h5py.File(self.dataset_path, 'r') as f:
            trial_data = f[f'subjects/{subject}/{trial}']
            
            # Load velocity and time data
            velocity = trial_data['mocap_data/elbow_velocity'][:]
            time = trial_data['mocap_data/time'][:]
            
        return velocity, time
    
    def extract_trial_residuals(self, subject: str, trial: str) -> Dict:
        """
        Extract residuals for a single trial.
        
        Args:
            subject: Subject ID
            trial: Trial ID
            
        Returns:
            Dict containing residuals and metadata
        """
        trial_name = f"{subject}/{trial}"
        print(f"Processing trial: {trial_name}")
        
        # Create data generator for this specific trial
        data_gen = IDMSTrajectoryDataGenerator(
            dataset_path=str(self.dataset_path),
            subjects=[subject],
            trials=[trial],
            window_size=self.config['window_size'],
            stride=self.config.get('stride', 25),  # FIX: Use config stride, default to 25 if missing
            delay=self.config['delay'],
            horizon=self.config['horizon'],
            n_trajectory_points=self.config['n_trajectory_points'],
            batch_size=64,  # Process in batches for efficiency
            shuffle=False,  # Keep temporal order
            emg_channels=self.config['emg_channels']
        )
        
        # Get all EMG windows and trajectory targets for this trial
        all_emg_windows = []
        all_target_velocities = []
        all_window_times = []
        
        for batch_idx in range(len(data_gen)):
            emg_batch, trajectory_batch = data_gen[batch_idx]
            
            # Convert from TensorFlow format to PyTorch format 
            # Expected PyTorch input: (batch, 1, channels, time)
            if emg_batch.ndim == 4:
                # From (batch, time, channels, 1) to (batch, 1, channels, time)
                emg_batch = np.transpose(emg_batch, (0, 3, 2, 1))
            elif emg_batch.ndim == 3:
                # From (batch, time, channels) to (batch, 1, channels, time)  
                emg_batch = emg_batch[:, None, :, :].transpose(0, 1, 3, 2)
            
            all_emg_windows.append(emg_batch)
            all_target_velocities.append(trajectory_batch)
            
            # Calculate window center times for alignment
            batch_start_idx = batch_idx * data_gen.batch_size
            batch_size = emg_batch.shape[0]
            window_times = np.arange(batch_start_idx, batch_start_idx + batch_size) * data_gen.stride
            all_window_times.append(window_times)
        
        # Concatenate all batches
        all_emg_windows = np.concatenate(all_emg_windows, axis=0)
        all_target_velocities = np.concatenate(all_target_velocities, axis=0)
        all_window_times = np.concatenate(all_window_times, axis=0)
        
        print(f"  Total windows: {len(all_emg_windows)}")
        
        # Extract v0 predictions from model
        v0_predictions = self.extract_v0_predictions(all_emg_windows)
        
        # Load ground truth velocity for comparison
        gt_velocity, gt_time = self.load_ground_truth_velocity(subject, trial)
        
        # Calculate ground truth v0 using trajectory method (same as predictions)
        # This matches the approach in visualize_pytorch_velocity_predictions.py
        
        # Load angle data for trajectory-based calculation
        with h5py.File(self.dataset_path, 'r') as f:
            trial_data = f[f'subjects/{subject}/{trial}']
            angles = trial_data['mocap_data/elbow_angle_filtered'][:]
            times = trial_data['mocap_data/time'][:]
        
        gt_v0_at_prediction_times = []
        dt = self.config['delay']  # Same dt used for predictions
        
        for window_idx, window_time in enumerate(all_window_times):
            # Calculate indices
            emg_end = window_time + self.config['window_size']  # End of EMG window (in samples)
            pred_time_idx = emg_end + int(dt * 2000)  # Prediction time (in samples)
            
            # Check bounds
            if emg_end < len(angles) and pred_time_idx < len(angles):
                # Method 2: Calculate velocity from trajectory (same as prediction method)
                angle_start = angles[emg_end]  # Angle at end of EMG window
                angle_future = angles[pred_time_idx]  # Angle at prediction time
                angle_diff = angle_future - angle_start  # Angular difference
                gt_velocity_trajectory = angle_diff / dt  # Angular velocity
                gt_v0_at_prediction_times.append(gt_velocity_trajectory)
            else:
                # Handle edge case - use last valid velocity or zero
                gt_v0_at_prediction_times.append(0.0)
        
        gt_v0_at_prediction_times = np.array(gt_v0_at_prediction_times)
        
        # Calculate residuals: true_v0 - predicted_v0
        residuals = gt_v0_at_prediction_times - v0_predictions
        
        # Get trial split info
        split_type = self.get_trial_split(trial_name)
        
        result = {
            'subject': subject,
            'trial': trial,
            'trial_name': trial_name,
            'split': split_type,
            'residuals': residuals,
            'predicted_v0': v0_predictions,
            'ground_truth_v0': gt_v0_at_prediction_times,
            'n_windows': len(residuals),
            'trial_duration': times[-1] - times[0]
        }
        
        print(f"  Split: {split_type}")
        print(f"  Residual stats: mean={np.mean(residuals):.6f}, std={np.std(residuals):.6f}")
        print(f"  Prediction range: [{np.min(v0_predictions):.3f}, {np.max(v0_predictions):.3f}]")
        print(f"  Ground truth range: [{np.min(gt_v0_at_prediction_times):.3f}, {np.max(gt_v0_at_prediction_times):.3f}]")
        
        return result
        
    def extract_residuals_for_trial(self, trial_name: str) -> Optional[np.ndarray]:
        """
        Extract residuals for a single trial by name (e.g. 'subject_005/trial_001').
        
        Args:
            trial_name: Trial name in format 'subject_xxx/trial_yyy'
            
        Returns:
            Residuals array or None if extraction fails
        """
        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()
                
            subject, trial = trial_name.split('/')
            result = self.extract_trial_residuals(subject, trial)
            return result['residuals'] if result else None
        except Exception as e:
            print(f"Error extracting residuals for {trial_name}: {e}")
            return None
        
    def extract_all_residuals(self, max_trials_per_split: Optional[Dict[str, int]] = None) -> Dict[str, List]:
        """
        Extract residuals from all trials in the data splits.
        
        Args:
            max_trials_per_split: Optional limits per split type {'test': 4, 'val': 8, 'train': 10}
            
        Returns:
            Dictionary organized by split type
        """
        if self.model is None:
            self.load_model()
            
        all_results = {'test': [], 'val': [], 'train': []}
        
        # Process trials by split priority (test first, then val, then train)
        splits_data = self.data_splits['data_splits']
        
        for split_name in ['test_trials', 'val_trials', 'train_trials']:
            split_type = split_name.replace('_trials', '')
            trials_list = splits_data[split_name]
            
            # Apply trial limits if specified
            if max_trials_per_split and split_type in max_trials_per_split:
                trials_list = trials_list[:max_trials_per_split[split_type]]
            
            print(f"\n{'='*50}")
            print(f"Processing {split_type.upper()} trials ({len(trials_list)} trials)")
            print(f"{'='*50}")
            
            for trial_name in tqdm(trials_list, desc=f"Extracting {split_type} residuals"):
                try:
                    subject, trial = trial_name.split('/')
                    result = self.extract_trial_residuals(subject, trial)
                    all_results[split_type].append(result)
                    
                except Exception as e:
                    print(f"  ERROR processing {trial_name}: {e}")
                    continue
                    
        return all_results
    
    def save_residuals(self, results: Dict[str, List], output_dir: str = "residual_analysis") -> None:
        """Save extracted residuals to files for analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving residuals to {output_path}")
        
        # Create summary statistics
        summary_data = []
        
        for split_type, trials_data in results.items():
            if not trials_data:
                continue
                
            print(f"\n{split_type.upper()} split:")
            print(f"  Trials: {len(trials_data)}")
            
            # Combine residuals from all trials in this split
            all_residuals = np.concatenate([t['residuals'] for t in trials_data])
            all_predictions = np.concatenate([t['predicted_v0'] for t in trials_data]) 
            all_ground_truth = np.concatenate([t['ground_truth_v0'] for t in trials_data])
            
            # Save residuals array for this split
            np.save(output_path / f"{split_type}_residuals.npy", all_residuals)
            np.save(output_path / f"{split_type}_predictions.npy", all_predictions)
            np.save(output_path / f"{split_type}_ground_truth.npy", all_ground_truth)
            
            # Calculate statistics
            stats = {
                'split': split_type,
                'n_trials': len(trials_data),
                'n_windows': len(all_residuals),
                'residual_mean': np.mean(all_residuals),
                'residual_std': np.std(all_residuals),
                'residual_min': np.min(all_residuals),
                'residual_max': np.max(all_residuals),
                'prediction_mean': np.mean(all_predictions),
                'prediction_std': np.std(all_predictions),
                'gt_mean': np.mean(all_ground_truth),
                'gt_std': np.std(all_ground_truth),
                'correlation': np.corrcoef(all_predictions, all_ground_truth)[0, 1],
                'mae': np.mean(np.abs(all_residuals)),
                'rmse': np.sqrt(np.mean(all_residuals**2))
            }
            summary_data.append(stats)
            
            print(f"  Windows: {stats['n_windows']:,}")
            print(f"  Residual: μ={stats['residual_mean']:.6f}, σ={stats['residual_std']:.6f}")
            print(f"  MAE: {stats['mae']:.6f}, RMSE: {stats['rmse']:.6f}")
            print(f"  Correlation: {stats['correlation']:.4f}")
            
        # Save summary statistics using csv module
        import csv
        with open(output_path / "residual_summary.csv", 'w', newline='') as f:
            if summary_data:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
        
        # Save detailed trial information
        trial_details = []
        for split_type, trials_data in results.items():
            for trial_data in trials_data:
                trial_details.append({
                    'split': trial_data['split'],
                    'subject': trial_data['subject'],
                    'trial': trial_data['trial'],
                    'trial_name': trial_data['trial_name'],
                    'n_windows': trial_data['n_windows'],
                    'trial_duration': trial_data['trial_duration'],
                    'residual_mean': np.mean(trial_data['residuals']),
                    'residual_std': np.std(trial_data['residuals']),
                    'prediction_mean': np.mean(trial_data['predicted_v0']),
                    'gt_mean': np.mean(trial_data['ground_truth_v0'])
                })
                
        # Save trial details using csv module
        with open(output_path / "trial_details.csv", 'w', newline='') as f:
            if trial_details:
                writer = csv.DictWriter(f, fieldnames=trial_details[0].keys())
                writer.writeheader()
                writer.writerows(trial_details)
        
        print(f"\nFiles saved:")
        print(f"  - residual_summary.csv: Overall statistics by split")
        print(f"  - trial_details.csv: Per-trial statistics")
        for split_type in results.keys():
            if results[split_type]:
                print(f"  - {split_type}_residuals.npy: Residual arrays")
                
    def plot_residual_diagnostics(self, results: Dict[str, List], output_dir: str = "residual_analysis") -> None:
        """Create diagnostic plots for residuals."""
        output_path = Path(output_dir)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PyTorch Elbow Model - Residual Analysis', fontsize=16)
        
        colors = {'test': 'red', 'val': 'blue', 'train': 'green'}
        
        for split_type, trials_data in results.items():
            if not trials_data:
                continue
                
            all_residuals = np.concatenate([t['residuals'] for t in trials_data])
            all_predictions = np.concatenate([t['predicted_v0'] for t in trials_data])
            all_ground_truth = np.concatenate([t['ground_truth_v0'] for t in trials_data])
            
            color = colors[split_type]
            
            # Residual histogram
            axes[0, 0].hist(all_residuals, bins=50, alpha=0.7, 
                          color=color, label=f'{split_type} (n={len(all_residuals):,})')
            
            # Prediction vs Ground Truth scatter
            axes[0, 1].scatter(all_ground_truth, all_predictions, 
                             alpha=0.5, s=1, color=color, label=split_type)
            
            # Residuals vs Predictions
            axes[0, 2].scatter(all_predictions, all_residuals,
                             alpha=0.5, s=1, color=color, label=split_type)
            
            # Time series of first trial
            if trials_data:
                first_trial = trials_data[0]
                # Create time indices for the residuals
                times = np.arange(len(first_trial['residuals'])) / 2000.0  # Convert to seconds assuming 2kHz
                axes[1, 0].plot(times[:500], first_trial['residuals'][:500],  # Plot first 500 points
                              color=color, alpha=0.8, label=f"{split_type} - {first_trial['trial_name']}")
            
        # Format plots
        axes[0, 0].set_xlabel('Residual')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Residual Distribution')
        axes[0, 0].legend()
        
        axes[0, 1].set_xlabel('Ground Truth v0')
        axes[0, 1].set_ylabel('Predicted v0')
        axes[0, 1].set_title('Prediction vs Ground Truth')
        axes[0, 1].plot([-1, 1], [-1, 1], 'k--', alpha=0.5)  # Perfect prediction line
        axes[0, 1].legend()
        
        axes[0, 2].set_xlabel('Predicted v0')
        axes[0, 2].set_ylabel('Residual')
        axes[0, 2].set_title('Residuals vs Predictions')
        axes[0, 2].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[0, 2].legend()
        
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].set_title('Residual Time Series (Sample Trials)')
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].legend()
        
        # QQ plots for normality check
        from scipy import stats
        for i, (split_type, trials_data) in enumerate(results.items()):
            if not trials_data:
                continue
                
            all_residuals = np.concatenate([t['residuals'] for t in trials_data])
            stats.probplot(all_residuals, dist="norm", plot=axes[1, 1])
            
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        
        # Box plots by split
        residual_data = []
        split_labels = []
        for split_type, trials_data in results.items():
            if trials_data:
                all_residuals = np.concatenate([t['residuals'] for t in trials_data])
                residual_data.append(all_residuals)
                split_labels.append(split_type)
                
        axes[1, 2].boxplot(residual_data, labels=split_labels)
        axes[1, 2].set_ylabel('Residual')
        axes[1, 2].set_title('Residuals by Data Split')
        axes[1, 2].axhline(0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path / "residual_diagnostics.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - residual_diagnostics.png: Diagnostic plots saved")


def main():
    """Main function for extracting residuals."""
    print("PyTorch Elbow Model Residual Extraction")
    print("=" * 50)
    
    # Configuration
    model_path = "pytorch_models/experiments/multistage_3108/stage2_finetune_subject_005_tcn_frozen"
    
    # Initialize extractor
    extractor = PyTorchResidualExtractor(
        model_path=model_path,
        dataset_path="data/idms_ready_dataset.h5"
    )
    
    # Extract residuals (start with limited trials for validation)
    print("\nExtracting residuals from limited trials for validation...")
    results = extractor.extract_all_residuals(
        max_trials_per_split={'test': 2, 'val': 3, 'train': 5}
    )
    
    # Save results
    output_dir = "residual_analysis/pytorch_elbow_residuals"
    extractor.save_residuals(results, output_dir)
    extractor.plot_residual_diagnostics(results, output_dir)
    
    print(f"\nResidual extraction complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()