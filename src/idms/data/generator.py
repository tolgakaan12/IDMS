#!/usr/bin/env python3
"""
IDMS Trajectory Data Generator

A data generator that reads from idms_ready_dataset.h5 and provides:
- EMG windows (normalized)
- Real future trajectory points sampled at specified intervals

Key features:
- Uses ACTUAL future data points (no extrapolation)
- Configurable delay and horizon
- Flexible trajectory sampling
- Proper train/validation/test splits
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
from idms.data.preproc import norm_emg

logger = logging.getLogger(__name__)


class IDMSTrajectoryDataGenerator(Sequence):
    """
    Data generator for trajectory prediction using real future data points.
    
    For each EMG window at time t:
    - Input: EMG[t-window_size:t] (normalized)
    - Output: Trajectory points at [t+delay, t+delay+dt, ..., t+delay+horizon]
              RELATIVE to the elbow angle at time t (end of EMG window)
    
    No extrapolation - only uses real future data that exists in the dataset.
    Trajectory targets are relative changes from the current elbow position.
    """
    
    def __init__(self,
                 dataset_path: str,
                 subjects: Optional[List[str]] = None,
                 trials: Optional[List[str]] = None,
                 window_size: int = 1000,  # 0.5s at 2000Hz
                 stride: int = 50,        # 0.025s between windows  
                 delay: float = 0.05,      # Start future prediction 0.2s ahead
                 horizon: float = 0.5,    # Predict 0.5s into future
                 n_trajectory_points: int = 10,  # Number of trajectory samples
                 batch_size: int = 32,
                 shuffle: bool = True,
                 shuffle_method: str = 'windows',  # 'windows' or 'trials' 
                 emg_channels: Optional[List[str]] = None,
                 normalize_emg: bool = True,
                 emg_preproc: Optional[str] = None,  # 'denoise', 'hp_filter', etc.
                 split: str = 'all',  # 'all', 'train', 'val', 'test'
                 test_ratio: float = 0.2,
                 val_ratio_from_trainval: float = 0.2,
                 seed: int = 42):
        """
        Initialize IDMS Trajectory Data Generator.
        
        Args:
            dataset_path: Path to idms_ready_dataset.h5
            subjects: List of subjects to include (None = all)
            trials: List of trial patterns to include (None = all)
            window_size: EMG window size in samples
            stride: Stride between windows in samples
            delay: Delay before trajectory starts (seconds)
            horizon: Total trajectory duration (seconds)
            n_trajectory_points: Number of trajectory points to sample
            batch_size: Batch size for training
            shuffle: Shuffle data between epochs
            shuffle_method: 'windows' = shuffle all windows freely, 'trials' = shuffle trials but keep windows in order
            emg_channels: EMG channels to use (None = all available)
            normalize_emg: Apply EMG normalization
            emg_preproc: Additional EMG preprocessing ('denoise', 'hp_filter', None)
            split: Which data split to use
            test_ratio: Ratio of data for test set (e.g., 0.05 = 5%)
            val_ratio_from_trainval: Ratio of train+val data for validation (e.g., 0.2 = 20% of remaining 95%)
            seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.subjects = subjects
        self.trials = trials
        self.window_size = window_size
        self.stride = stride
        self.delay = delay
        self.horizon = horizon
        self.n_trajectory_points = n_trajectory_points
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_method = shuffle_method
        self.emg_channels = emg_channels or ['biceps', 'triceps', 'bra', 'ecu']
        self.normalize_emg = normalize_emg
        self.emg_preproc = emg_preproc
        self.split = split
        self.test_ratio = test_ratio
        self.val_ratio_from_trainval = val_ratio_from_trainval
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        
        # Calculate trajectory time points
        self.trajectory_times = np.linspace(delay, delay + horizon, n_trajectory_points)
        
        # Initialize data storage
        self.trials_data = []
        self.window_indices = []  # For 'windows' shuffle method
        self.trial_windows = []   # For 'trials' shuffle method - list of (trial_idx, windows_in_trial)
        self.n_channels = len(self.emg_channels)
        
        # Load data
        self._load_dataset()
        self._create_windows()
        self._apply_split()
        
        # Shuffle if requested
        self.on_epoch_end()
        
        logger.info(f"IDMS Trajectory Generator initialized:")
        logger.info(f"  Dataset: {dataset_path}")
        logger.info(f"  Trials loaded: {len(self.trials_data)}")
        logger.info(f"  Windows available: {len(self.window_indices)}")
        logger.info(f"  EMG window: {window_size} samples ({window_size/2000:.3f}s)")
        logger.info(f"  Trajectory: {n_trajectory_points} points from {delay:.3f}s to {delay+horizon:.3f}s")
        logger.info(f"  Split: {split}")
        
    def _load_dataset(self):
        """Load data from HDF5 dataset."""
        with h5py.File(self.dataset_path, 'r') as f:
            subjects_group = f['subjects']
            
            # Get subject list
            if self.subjects is None:
                subject_names = list(subjects_group.keys())
            else:
                subject_names = [s for s in self.subjects if s in subjects_group]
            
            for subject_name in subject_names:
                subject_group = subjects_group[subject_name]
                
                # Get trial list
                trial_names = list(subject_group.keys())
                if self.trials is not None:
                    # Filter trials based on patterns
                    filtered_trials = []
                    for pattern in self.trials:
                        filtered_trials.extend([t for t in trial_names if pattern in t])
                    trial_names = filtered_trials
                
                for trial_name in trial_names:
                    trial_group = subject_group[trial_name]
                    
                    # Load EMG data
                    emg_data = {}
                    for channel in self.emg_channels:
                        if channel in trial_group['emg_data']:
                            emg_data[channel] = trial_group['emg_data'][channel][:]
                        else:
                            logger.warning(f"Channel {channel} not found in {subject_name}/{trial_name}")
                    
                    # Apply EMG preprocessing and normalization (channel-specific across entire trial)
                    if emg_data:
                        # Stack channels for processing (channels x samples)
                        emg_array = np.stack([emg_data[ch] for ch in self.emg_channels], axis=0)
                        
                        # Apply additional preprocessing if specified
                        if self.emg_preproc == 'denoise':
                            from idms.data.preproc import denoise
                            emg_array = denoise(emg_array, sfreq=2000, high_band=20, low_band=450)
                        elif self.emg_preproc == 'hp_filter':
                            from idms.data.preproc import hp_filter_120hz
                            emg_array = hp_filter_120hz(emg_array, cutoff=120, sfreq=2000)
                        elif self.emg_preproc == 'bandpass':
                            from idms.data.preproc import bp_filter
                            emg_array = bp_filter(emg_array, high_band=7, low_band=400, sfreq=2000)
                        
                        # Apply normalization after preprocessing
                        if self.normalize_emg:
                            emg_array = norm_emg(emg_array)  # Normalize using trial-global statistics
                        
                        # Update emg_data dict with processed values
                        for i, ch in enumerate(self.emg_channels):
                            emg_data[ch] = emg_array[i]
                    
                    # Load mocap data  
                    angle_data = trial_group['mocap_data']['elbow_angle_filtered'][:]
                    velocity_data = trial_group['mocap_data']['elbow_velocity'][:]
                    time_data = trial_group['mocap_data']['time'][:]
                    
                    # Store trial data
                    if emg_data:  # Only store if EMG data exists
                        trial_info = {
                            'subject': subject_name,
                            'trial': trial_name,
                            'emg': np.stack([emg_data[ch] for ch in self.emg_channels], axis=0),
                            'angle': angle_data,
                            'velocity': velocity_data, 
                            'time': time_data,
                            'n_samples': len(angle_data)
                        }
                        self.trials_data.append(trial_info)
        
        logger.info(f"Loaded {len(self.trials_data)} trials")
    
    def _create_windows(self):
        """Create sliding windows from trials data."""
        self.window_indices = []
        self.trial_windows = []
        
        # Convert time delays to sample indices (assuming 2000 Hz)
        fs = 2000.0
        delay_samples = int(self.delay * fs)
        horizon_samples = int(self.horizon * fs)
        max_future_samples = delay_samples + horizon_samples
        
        total_rejected = 0
        
        for trial_idx, trial in enumerate(self.trials_data):
            n_samples = trial['n_samples']
            
            # Calculate valid window range
            # Need window_size samples for EMG + max_future_samples for trajectory
            max_start_idx = n_samples - self.window_size - max_future_samples
            
            if max_start_idx <= 0:
                logger.warning(f"Trial {trial['subject']}/{trial['trial']} too short - skipping")
                continue
            
            # Find NaN regions in angle data
            angle_data = trial['angle']
            nan_mask = np.isnan(angle_data)
            
            # Create windows with stride, but skip those with NaN trajectory data
            window_starts = np.arange(0, max_start_idx, self.stride)
            trial_window_list = []
            trial_rejected = 0
            
            for start_idx in window_starts:
                # Calculate trajectory sample indices
                traj_samples = [start_idx + self.window_size + int(t * fs) for t in self.trajectory_times]
                
                # Check if any trajectory samples fall in NaN regions
                trajectory_valid = True
                for traj_idx in traj_samples:
                    if traj_idx >= n_samples or nan_mask[traj_idx]:
                        trajectory_valid = False
                        break
                
                # Also check EMG window for NaN (if EMG has NaN, skip too)
                emg_start = start_idx
                emg_end = start_idx + self.window_size
                emg_data_slice = trial['emg'][:, emg_start:emg_end]
                emg_has_nan = np.any(np.isnan(emg_data_slice))
                
                if trajectory_valid and not emg_has_nan:
                    window_info = {
                        'trial_idx': trial_idx,
                        'start_idx': start_idx,
                        'emg_start': emg_start,
                        'emg_end': emg_end,
                        'traj_start': start_idx + self.window_size + delay_samples,
                        'traj_samples': traj_samples
                    }
                    
                    # Add to both lists
                    self.window_indices.append(window_info)
                    trial_window_list.append(len(self.window_indices) - 1)  # Store index into window_indices
                else:
                    trial_rejected += 1
            
            # Store trial's window indices
            if trial_window_list:
                self.trial_windows.append({
                    'trial_idx': trial_idx,
                    'window_indices': trial_window_list
                })
            
            total_rejected += trial_rejected
            if trial_rejected > 0:
                logger.info(f"Trial {trial['subject']}/{trial['trial']}: rejected {trial_rejected} windows due to NaN data")
        
        logger.info(f"Created {len(self.window_indices)} windows across {len(self.trial_windows)} trials")
        if total_rejected > 0:
            logger.info(f"Rejected {total_rejected} windows total due to NaN data ({total_rejected/(len(self.window_indices)+total_rejected)*100:.1f}%)")
    
    def _apply_split(self):
        """Apply train/validation/test split."""
        if self.split == 'all':
            # Initialize shuffle orders for all data
            if self.shuffle_method == 'windows':
                self.current_window_order = list(range(len(self.window_indices)))
            elif self.shuffle_method == 'trials':
                self.current_trial_order = list(range(len(self.trial_windows)))
            return
        
        # Split by trials to avoid data leakage
        # First: split test set (e.g., 5%)
        # Second: split remaining train+val (e.g., 80/20 = 76% train, 19% val)
        trial_indices = np.arange(len(self.trials_data))
        np.random.shuffle(trial_indices)
        
        n_test = int(len(trial_indices) * self.test_ratio)
        n_trainval = len(trial_indices) - n_test
        n_val = int(n_trainval * self.val_ratio_from_trainval)
        n_train = n_trainval - n_val
        
        if self.split == 'train':
            valid_trials = set(trial_indices[:n_train])
        elif self.split == 'val':
            valid_trials = set(trial_indices[n_train:n_train + n_val])
        elif self.split == 'test':
            valid_trials = set(trial_indices[n_train + n_val:])
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        logger.info(f"Split ratios - Train: {n_train}/{len(trial_indices)} ({n_train/len(trial_indices):.1%}), "
                   f"Val: {n_val}/{len(trial_indices)} ({n_val/len(trial_indices):.1%}), "
                   f"Test: {n_test}/{len(trial_indices)} ({n_test/len(trial_indices):.1%})")
        
        # Filter data based on shuffle method
        if self.shuffle_method == 'windows':
            # Filter windows to only include valid trials
            self.window_indices = [w for w in self.window_indices if w['trial_idx'] in valid_trials]
            self.current_window_order = list(range(len(self.window_indices)))
        elif self.shuffle_method == 'trials':
            # Filter trial_windows to only include valid trials
            self.trial_windows = [tw for tw in self.trial_windows if tw['trial_idx'] in valid_trials]
            self.current_trial_order = list(range(len(self.trial_windows)))
            # Count total windows
            total_windows = sum(len(tw['window_indices']) for tw in self.trial_windows)
            logger.info(f"Split '{self.split}': {total_windows} windows across {len(self.trial_windows)} trials")
            return
        
        logger.info(f"Split '{self.split}': {len(self.window_indices)} windows")
    
    def __len__(self):
        """Number of batches per epoch."""
        if self.shuffle_method == 'windows':
            return int(np.ceil(len(self.window_indices) / self.batch_size))
        elif self.shuffle_method == 'trials':
            total_windows = sum(len(tw['window_indices']) for tw in self.trial_windows)
            return int(np.ceil(total_windows / self.batch_size))
        else:
            raise ValueError(f"Invalid shuffle method: {self.shuffle_method}")
    
    def __getitem__(self, batch_idx):
        """Generate one batch of data."""
        if self.shuffle_method == 'windows':
            return self._getitem_windows(batch_idx)
        elif self.shuffle_method == 'trials':
            return self._getitem_trials(batch_idx)
        else:
            raise ValueError(f"Invalid shuffle method: {self.shuffle_method}")
    
    def _getitem_windows(self, batch_idx):
        """Generate batch using window-level shuffling."""
        # Get batch window indices
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.current_window_order))
        
        batch_windows = [self.window_indices[self.current_window_order[i]] 
                        for i in range(start_idx, end_idx)]
        
        return self._generate_batch_data(batch_windows)
    
    def _getitem_trials(self, batch_idx):
        """Generate batch using trial-level shuffling (windows within trials stay in order)."""
        # Flatten trial windows according to current trial order
        all_window_indices = []
        for trial_order_idx in self.current_trial_order:
            trial_windows = self.trial_windows[trial_order_idx]
            # Add windows from this trial in order
            for window_idx in trial_windows['window_indices']:
                all_window_indices.append(self.window_indices[window_idx])
        
        # Get batch
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(all_window_indices))
        batch_windows = all_window_indices[start_idx:end_idx]
        
        return self._generate_batch_data(batch_windows)
    
    def _generate_batch_data(self, batch_windows):
        """Generate EMG and trajectory data for a batch of windows."""
        X_batch = []
        y_batch = []
        
        for window_info in batch_windows:
            trial = self.trials_data[window_info['trial_idx']]
            
            # Extract EMG window (already normalized at trial level if normalize_emg=True)
            emg_window = trial['emg'][
                :, window_info['emg_start']:window_info['emg_end']
            ].T  # Shape: (window_size, n_channels)
            
            # Get initial angle at the end of EMG window (reference point for trajectory)
            initial_sample_idx = window_info['emg_end']  # End of EMG window
            if initial_sample_idx < trial['n_samples']:
                initial_angle = trial['angle'][initial_sample_idx]
                # Check if initial angle is valid
                if np.isnan(initial_angle):
                    logger.debug(f"NaN detected in initial angle for trial {trial['subject']}/{trial['trial']} at sample {initial_sample_idx}")
                    continue
            else:
                # Fallback to last available sample
                initial_angle = trial['angle'][-1]
                if np.isnan(initial_angle):
                    logger.debug(f"NaN detected in fallback initial angle for trial {trial['subject']}/{trial['trial']}")
                    continue
            
            # Extract trajectory points at specified future times (relative to initial angle)
            trajectory_points = []
            for sample_idx in window_info['traj_samples']:
                if sample_idx < trial['n_samples']:
                    angle_value = trial['angle'][sample_idx]
                    # Final safety check for NaN values
                    if np.isnan(angle_value):
                        logger.debug(f"NaN detected in trajectory for trial {trial['subject']}/{trial['trial']} at sample {sample_idx}")
                        # Skip this window entirely by returning early
                        continue
                    # Make trajectory relative to initial angle
                    relative_angle = angle_value - initial_angle
                    trajectory_points.append(relative_angle)
                else:
                    # This shouldn't happen with proper window creation, but safety check
                    angle_value = trial['angle'][-1]
                    if np.isnan(angle_value):
                        logger.debug(f"NaN detected in fallback trajectory for trial {trial['subject']}/{trial['trial']}")
                        continue
                    # Make trajectory relative to initial angle
                    relative_angle = angle_value - initial_angle
                    trajectory_points.append(relative_angle)
            
            # Only add window if we got all trajectory points without NaN
            if len(trajectory_points) == len(window_info['traj_samples']):
                X_batch.append(emg_window)
                y_batch.append(trajectory_points)
        
        return np.array(X_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        """Shuffle data after each epoch."""
        if self.shuffle:
            if self.shuffle_method == 'windows':
                # Shuffle all windows freely
                np.random.shuffle(self.current_window_order)
            elif self.shuffle_method == 'trials':
                # Shuffle trial order, but keep windows within trials in temporal order
                np.random.shuffle(self.current_trial_order)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'n_trials': len(self.trials_data),
            'n_windows': len(self.window_indices),
            'n_channels': self.n_channels,
            'emg_channels': self.emg_channels,
            'window_size': self.window_size,
            'window_size_seconds': self.window_size / 2000.0,
            'stride': self.stride,
            'stride_seconds': self.stride / 2000.0,
            'delay': self.delay,
            'horizon': self.horizon,
            'n_trajectory_points': self.n_trajectory_points,
            'trajectory_times': self.trajectory_times.tolist(),
            'batch_size': self.batch_size,
            'split': self.split,
            'sampling_rate': 2000.0
        }
        
        if self.trials_data:
            # Get trial duration statistics
            durations = [t['n_samples'] / 2000.0 for t in self.trials_data]
            stats.update({
                'trial_duration_mean': np.mean(durations),
                'trial_duration_std': np.std(durations),
                'trial_duration_min': np.min(durations),
                'trial_duration_max': np.max(durations),
                'total_data_hours': sum(durations) / 3600.0
            })
        
        return stats
    
    def get_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a sample batch for inspection."""
        return self[0]
    
    def validate_trajectories(self, n_samples: int = 5) -> Dict[str, Any]:
        """
        Validate that trajectory sampling is working correctly.
        
        Args:
            n_samples: Number of sample windows to validate
            
        Returns:
            Validation statistics
        """
        validation_stats = {
            'future_data_available': 0,
            'valid_trajectories': 0,
            'trajectory_ranges': []
        }
        
        sample_indices = np.random.choice(len(self.window_indices), 
                                        min(n_samples, len(self.window_indices)), 
                                        replace=False)
        
        for idx in sample_indices:
            window_info = self.window_indices[idx]
            trial = self.trials_data[window_info['trial_idx']]
            
            # Check if all trajectory samples are within bounds
            all_valid = all(s < trial['n_samples'] for s in window_info['traj_samples'])
            if all_valid:
                validation_stats['valid_trajectories'] += 1
                
                # Get trajectory values for range analysis
                traj_values = [trial['angle'][s] for s in window_info['traj_samples']]
                validation_stats['trajectory_ranges'].append((min(traj_values), max(traj_values)))
            
            validation_stats['future_data_available'] += 1
        
        validation_stats['validation_success_rate'] = (
            validation_stats['valid_trajectories'] / validation_stats['future_data_available']
        )
        
        return validation_stats


def test_trajectory_generator():
    """Test the IDMSTrajectoryDataGenerator."""
    print("Testing IDMSTrajectoryDataGenerator...")
    print("=" * 50)
    
    try:
        # Create generator
        gen = IDMSTrajectoryDataGenerator(
            dataset_path='data/idms_ready_dataset.h5',
            subjects=['subject_001'],  # Test with one subject first
            window_size=500,   # 0.25s
            stride=50,         # 0.025s
            delay=0.2,         # 0.2s delay
            horizon=0.5,       # 0.5s horizon  
            n_trajectory_points=10,
            batch_size=8,
            split='train'
        )
        
        # Get statistics
        stats = gen.get_dataset_stats()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Test batch generation
        print(f"\nTesting batch generation...")
        X, y = gen[0]
        print(f"EMG input shape: {X.shape}")
        print(f"Trajectory output shape: {y.shape}")
        print(f"EMG range: [{X.min():.6f}, {X.max():.6f}]")
        print(f"Trajectory range: [{y.min():.6f}, {y.max():.6f}]")
        
        # Validate trajectories
        print(f"\nValidating trajectory sampling...")
        validation = gen.validate_trajectories()
        print(f"Validation success rate: {validation['validation_success_rate']:.2%}")
        
        print("\n✅ IDMSTrajectoryDataGenerator test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_trajectory_generator()