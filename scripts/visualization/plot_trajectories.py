#!/usr/bin/env python3
"""
Visualize model performance on test trials with GIF animations.

Creates GIF animations showing ground truth vs predicted trajectories for each test trial
of each subject, with individual R² values for each trajectory point.
"""

import numpy as np
import torch
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Import model components
from pytorch_models.tcanet_idms import create_tcanet_idms_model
from pytorch_models.pytorch_data_adapter import PyTorchIDMSDataset
from data_gen.idms_trajectory_datagenerator import IDMSTrajectoryDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})


class TestTrajectoryVisualizer:
    """Visualize test trajectory predictions with GIF animations."""
    
    def __init__(self, experiment_dir: str, dataset_path: str, cross_subject: bool = False):
        """
        Initialize visualizer.
        
        Args:
            experiment_dir: Path to PyTorch experiment directory
            dataset_path: Path to the HDF5 dataset
            cross_subject: If True, use general model's test splits for cross-subject evaluation
        """
        self.experiment_dir = Path(experiment_dir)
        self.dataset_path = dataset_path
        self.cross_subject = cross_subject
        
        # Load configuration and model
        self.config = self._load_config()
        self.data_splits = self._load_data_splits()
        self.model = self._load_model()
        
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def _load_config(self):
        """Load model configuration."""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_data_splits(self):
        """Load predefined data splits."""
        if self.cross_subject:
            # Use general model's data splits for cross-subject evaluation
            general_splits_path = Path.cwd() / "pytorch_models/experiments/multistage_3108/stage1_general_model/data_splits.json"
            if general_splits_path.exists():
                logger.info(f"Using cross-subject splits from: {general_splits_path}")
                with open(general_splits_path, 'r') as f:
                    splits_data = json.load(f)
                    return splits_data.get('data_splits', {})
            else:
                logger.warning(f"Cross-subject splits not found at {general_splits_path}")
                return None
        else:
            # Use model's own data splits
            splits_path = self.experiment_dir / "data_splits.json"
            if splits_path.exists():
                with open(splits_path, 'r') as f:
                    splits_data = json.load(f)
                    return splits_data.get('data_splits', {})
            return None
    
    def _load_model(self):
        """Load the trained model."""
        model = create_tcanet_idms_model(
            window_size=self.config['window_size'],
            n_channels=len(self.config['emg_channels']),
            trajectory_points=self.config['n_trajectory_points'],
            trajectory_horizon=self.config['horizon'],
            trajectory_delay=self.config['delay']
        )
        
        model_path = self.experiment_dir / "best_model.pt"
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Handle compiled models (with _orig_mod. prefix)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            logger.info("Detected compiled model, removing _orig_mod. prefixes")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # Remove '_orig_mod.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def _get_test_trials_for_subject(self, subject):
        """Get predefined test trials for a subject."""
        if self.data_splits is None:
            return None
        
        test_trials = self.data_splits.get('test_trials', [])
        subject_test_trials = []
        for trial in test_trials:
            if trial.startswith(f"{subject}/"):
                trial_name = trial.split("/")[1]
                subject_test_trials.append(trial_name)
        
        return subject_test_trials if subject_test_trials else None
    
    def _get_trial_data(self, subject: str, trial: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get EMG and trajectory data for a specific trial.
        
        Returns:
            Tuple of (emg_data, trajectory_data)
        """
        # Create dataset for this specific trial
        dataset = PyTorchIDMSDataset(
            dataset_path=self.dataset_path,
            subjects=[subject],
            trials=[trial],
            window_size=self.config['window_size'],
            stride=50,  # Use smaller stride for more data points
            delay=self.config['delay'],
            horizon=self.config['horizon'],
            n_trajectory_points=self.config['n_trajectory_points'],
            batch_size=1,
            shuffle=False,
            emg_preproc=self.config.get('emg_preproc'),
            split='all'
        )
        
        # Extract all EMG windows and trajectories
        emg_windows = []
        trajectories = []
        
        for i in range(len(dataset)):
            emg, traj = dataset[i]
            # emg shape: (1, n_channels, window_size), we want (window_size, n_channels)
            emg_data = emg[0].numpy().T  # (n_channels, window_size) -> (window_size, n_channels)
            emg_windows.append(emg_data)
            trajectories.append(traj.numpy())  # (n_trajectory_points,)
        
        return np.array(emg_windows), np.array(trajectories)
    
    def _predict_trajectories(self, emg_windows: np.ndarray) -> np.ndarray:
        """Predict trajectories for EMG windows."""
        predictions = []
        
        with torch.no_grad():
            for emg in emg_windows:
                # emg is shape (window_size, n_channels), we need (1, 1, n_channels, window_size)
                emg_tensor = torch.tensor(emg.T, dtype=torch.float32)  # (n_channels, window_size)
                emg_tensor = emg_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, n_channels, window_size)
                pred = self.model(emg_tensor)
                predictions.append(pred.squeeze().numpy())
        
        return np.array(predictions)
    
    def _calculate_r2_metrics(self, true_traj: np.ndarray, pred_traj: np.ndarray) -> Dict[str, Any]:
        """
        Calculate R² using multiple methods for trajectory predictions.
        
        Args:
            true_traj: Ground truth trajectories (n_samples, n_trajectory_points)
            pred_traj: Predicted trajectories (n_samples, n_trajectory_points)
            
        Returns:
            Dictionary with different R² calculations
        """
        
        # Method 1: R² per trajectory point (what sklearn would recommend)
        r2_per_point = []
        for point_idx in range(true_traj.shape[1]):
            true_points = true_traj[:, point_idx]
            pred_points = pred_traj[:, point_idx]
            r2 = r2_score(true_points, pred_points)
            r2_per_point.append(r2)
        r2_per_point = np.array(r2_per_point)
        
        # Method 2: R² per trajectory (each 10-point trajectory as one unit)
        r2_per_trajectory = []
        for sample_idx in range(true_traj.shape[0]):
            true_sample = true_traj[sample_idx, :]
            pred_sample = pred_traj[sample_idx, :]
            if np.var(true_sample) > 1e-10:  # Check for variance
                r2 = r2_score(true_sample, pred_sample)
                r2_per_trajectory.append(r2)
        r2_per_trajectory_mean = np.mean(r2_per_trajectory) if r2_per_trajectory else 0.0
        
        # Method 3: sklearn multi-output R² (proper way)
        r2_multioutput_uniform = r2_score(true_traj, pred_traj, multioutput='uniform_average')
        r2_multioutput_variance = r2_score(true_traj, pred_traj, multioutput='variance_weighted')
        
        # Method 4: Current test_cross_subject method (flattened - for comparison)
        true_flat = true_traj.flatten()
        pred_flat = pred_traj.flatten()
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        r2_flattened = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Method 5: Global R² treating all as independent (worst method but for comparison)
        r2_global = r2_score(true_flat, pred_flat)
        
        return {
            'r2_per_point': r2_per_point,  # Array of R² for each trajectory point
            'r2_per_point_mean': np.mean(r2_per_point),
            'r2_per_trajectory_mean': r2_per_trajectory_mean,
            'r2_multioutput_uniform': r2_multioutput_uniform,
            'r2_multioutput_variance': r2_multioutput_variance,
            'r2_flattened': r2_flattened,  # Current test_cross_subject method
            'r2_global': r2_global,
            'n_samples': true_traj.shape[0],
            'n_points': true_traj.shape[1]
        }
    
    def _create_trajectory_gif(self, subject: str, trial: str, true_traj: np.ndarray, 
                             pred_traj: np.ndarray, r2_metrics: Dict[str, Any], 
                             output_path: Path, max_frames: int = 50):
        """Create GIF animation for trajectory comparison with improved speed."""
        
        # Limit number of frames to avoid very long GIFs
        n_frames = min(max_frames, len(true_traj))
        frame_indices = np.linspace(0, len(true_traj)-1, n_frames, dtype=int)
        
        # Create figure and axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        fig.suptitle(f'{subject} - {trial} (Showing {n_frames}/{len(true_traj)} frames)', fontsize=14, fontweight='bold')
        
        # Extract R² metrics
        r2_per_point = r2_metrics['r2_per_point']
        
        # Trajectory time points
        time_points = np.linspace(self.config['delay'], 
                                self.config['delay'] + self.config['horizon'], 
                                self.config['n_trajectory_points'])
        
        def animate(frame_idx):
            frame = frame_indices[frame_idx]
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Plot trajectory comparison
            ax1.plot(time_points, true_traj[frame], 'b-', linewidth=2, label='Ground Truth', marker='o')
            ax1.plot(time_points, pred_traj[frame], 'r--', linewidth=2, label='Prediction', marker='s')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Elbow Angle (rad)')
            ax1.set_title(f'Trajectory Comparison (Frame {frame + 1}/{len(true_traj)})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits
            all_data = np.concatenate([true_traj.flatten(), pred_traj.flatten()])
            y_margin = 0.1 * (np.max(all_data) - np.min(all_data))
            ax1.set_ylim(np.min(all_data) - y_margin, np.max(all_data) + y_margin)
            
            # Plot R² scores per trajectory point
            colors = plt.cm.viridis(np.linspace(0, 1, len(r2_per_point)))
            bars = ax2.bar(range(len(r2_per_point)), r2_per_point, color=colors, alpha=0.7)
            ax2.set_xlabel('Trajectory Point')
            ax2.set_ylabel('R² Score')
            ax2.set_title('R² per Trajectory Point (Method 1: Per Point)')
            ax2.set_ylim(min(r2_per_point.min() - 0.1, 0), max(r2_per_point.max() + 0.1, 1))
            ax2.grid(True, alpha=0.3)
            
            # Add R² values on top of bars
            for i, (bar, r2_val) in enumerate(zip(bars, r2_per_point)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if r2_val >= 0 else bar.get_height() - 0.05, 
                        f'{r2_val:.3f}', ha='center', va='bottom' if r2_val >= 0 else 'top', fontsize=8)
            
            # Add time labels for trajectory points
            ax2_twin = ax2.twiny()
            ax2_twin.set_xlim(ax2.get_xlim())
            ax2_twin.set_xticks(range(len(time_points)))
            ax2_twin.set_xticklabels([f'{t:.2f}s' for t in time_points], rotation=45)
            ax2_twin.set_xlabel('Time')
            
            # Show R² comparison table
            methods = ['Per Point (Mean)', 'Per Trajectory', 'Multi-output Uniform', 'Multi-output Variance', 'Flattened (Current)', 'Global']
            values = [
                r2_metrics['r2_per_point_mean'],
                r2_metrics['r2_per_trajectory_mean'], 
                r2_metrics['r2_multioutput_uniform'],
                r2_metrics['r2_multioutput_variance'],
                r2_metrics['r2_flattened'],
                r2_metrics['r2_global']
            ]
            
            y_pos = np.arange(len(methods))
            colors_methods = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(methods)))
            bars3 = ax3.barh(y_pos, values, color=colors_methods, alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(methods)
            ax3.set_xlabel('R² Score')
            ax3.set_title('R² Comparison: Different Calculation Methods')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add values on bars
            for i, (bar, value) in enumerate(zip(bars3, values)):
                ax3.text(value + 0.01 if value >= 0 else value - 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center', fontsize=9, fontweight='bold')
        
        # Create animation with fewer frames and faster speed
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                     interval=300, repeat=True)
        
        # Save as GIF with optimization
        logger.info(f"Saving GIF: {output_path} ({n_frames} frames)")
        anim.save(output_path, writer='pillow', fps=3)  # Slower FPS for readability
        plt.close(fig)
    
    def visualize_all_test_trials(self, output_dir: str = None):
        """Create GIF visualizations for all test trials."""
        
        if output_dir is None:
            if self.cross_subject:
                output_dir = self.experiment_dir / "cross_subject_trajectory_gifs"
            else:
                output_dir = self.experiment_dir / "trajectory_gifs"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Determine which subjects to test
        if self.cross_subject:
            # For cross-subject evaluation, determine training subjects from model config
            training_subjects = self.config.get('subjects', None)
            all_subjects = ['subject_001', 'subject_002', 'subject_003', 'subject_004', 'subject_005']
            
            if training_subjects is None:
                # Model was trained on all subjects, test all subjects
                subjects = all_subjects
                logger.info("Cross-subject mode: Model trained on ALL subjects, testing all subjects")
            else:
                # Model was trained on specific subjects, test on OTHER subjects
                subjects = [s for s in all_subjects if s not in training_subjects]
                logger.info(f"Cross-subject mode: Model trained on {training_subjects}, testing on {subjects}")
        else:
            # Regular mode: test on all subjects
            subjects = ['subject_001', 'subject_002', 'subject_003', 'subject_004', 'subject_005']
        
        for subject in subjects:
            logger.info(f"\n=== Processing {subject} ===")
            
            # Get test trials for this subject
            test_trials = self._get_test_trials_for_subject(subject)
            if not test_trials:
                logger.warning(f"No test trials found for {subject}")
                continue
            
            logger.info(f"Test trials for {subject}: {test_trials}")
            
            subject_dir = output_dir / subject
            subject_dir.mkdir(exist_ok=True)
            
            for trial in test_trials:
                logger.info(f"Processing {subject}/{trial}...")
                
                try:
                    # Get trial data
                    emg_windows, true_trajectories = self._get_trial_data(subject, trial)
                    
                    if len(emg_windows) == 0:
                        logger.warning(f"No data found for {subject}/{trial}")
                        continue
                    
                    # Get predictions
                    predicted_trajectories = self._predict_trajectories(emg_windows)
                    
                    # Calculate R² metrics using multiple methods
                    r2_metrics = self._calculate_r2_metrics(true_trajectories, predicted_trajectories)
                    
                    # Create GIF with limited frames for speed
                    gif_path = subject_dir / f"{trial}.gif"
                    self._create_trajectory_gif(subject, trial, true_trajectories, 
                                              predicted_trajectories, r2_metrics, gif_path, max_frames=60)
                    
                    logger.info(f"Created GIF: {gif_path}")
                    logger.info(f"R² Metrics for {subject}/{trial}:")
                    logger.info(f"  Per Point (Mean): {r2_metrics['r2_per_point_mean']:.3f}")
                    logger.info(f"  Per Trajectory: {r2_metrics['r2_per_trajectory_mean']:.3f}")
                    logger.info(f"  Multi-output Uniform: {r2_metrics['r2_multioutput_uniform']:.3f}")
                    logger.info(f"  Multi-output Variance: {r2_metrics['r2_multioutput_variance']:.3f}")
                    logger.info(f"  Flattened (Current method): {r2_metrics['r2_flattened']:.3f}")
                    logger.info(f"  Global: {r2_metrics['r2_global']:.3f}")
                    logger.info(f"  Per Point Details: {r2_metrics['r2_per_point']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {subject}/{trial}: {e}")
                    continue
        
        logger.info(f"\nAll GIF visualizations saved to: {output_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize test trajectory predictions')
    parser.add_argument('experiment_dir', help='Path to experiment directory')
    parser.add_argument('--dataset', default='data/idms_ready_dataset.h5', 
                       help='Path to HDF5 dataset')
    parser.add_argument('--output_dir', help='Output directory for GIFs')
    parser.add_argument('--cross', action='store_true', 
                       help='Use cross-subject evaluation with general model test splits')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TestTrajectoryVisualizer(args.experiment_dir, args.dataset, cross_subject=args.cross)
    
    # Generate all visualizations
    visualizer.visualize_all_test_trials(args.output_dir)


if __name__ == "__main__":
    main()