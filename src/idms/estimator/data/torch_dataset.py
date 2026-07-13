"""
PyTorch Data Adapter for IDMS Trajectory Prediction

Wraps the existing IDMSTrajectoryDataGenerator to provide PyTorch-compatible 
data loading while reusing all preprocessing and splitting logic.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os

# Add the parent directory to path to import existing data generator
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from idms.data.generator import IDMSTrajectoryDataGenerator


class PyTorchIDMSDataset(Dataset):
    """
    PyTorch Dataset wrapper for IDMSTrajectoryDataGenerator
    
    This maintains all the existing data processing logic while providing
    PyTorch-compatible data loading.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 subjects=None,
                 trials=None,
                 window_size: int = 1000,
                 stride: int = 50,
                 delay: float = 0.05,
                 horizon: float = 0.25,
                 n_trajectory_points: int = 10,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 emg_preproc=None,
                 split: str = 'train',
                 test_ratio: float = 0.05,
                 val_ratio_from_trainval: float = 0.2,
                 seed: int = 42):
        """
        Initialize PyTorch dataset using existing IDMS data generator
        
        Args: Same as IDMSTrajectoryDataGenerator
        """
        super().__init__()
        
        # Create the existing data generator
        self.data_generator = IDMSTrajectoryDataGenerator(
            dataset_path=dataset_path,
            subjects=subjects,
            trials=trials,
            window_size=window_size,
            stride=stride,
            delay=delay,
            horizon=horizon,
            n_trajectory_points=n_trajectory_points,
            batch_size=1,  # We'll handle batching through DataLoader
            shuffle=False,  # We'll handle shuffling through DataLoader
            emg_preproc=emg_preproc,
            split=split,
            test_ratio=test_ratio,
            val_ratio_from_trainval=val_ratio_from_trainval,
            seed=seed
        )
        
        # Pre-generate all data for efficient PyTorch loading
        self._preload_data()
        
    def _preload_data(self):
        """Preload all data from the generator for efficient access"""
        print(f"Preloading data from generator with {len(self.data_generator)} batches...")
        
        all_X = []
        all_y = []
        
        loaded_count = 0
        for batch_idx in range(len(self.data_generator)):
            try:
                X_batch, y_batch = self.data_generator[batch_idx]
                
                # Debug: Check batch contents
                if batch_idx < 5:  # Debug first few batches
                    print(f"  Debug batch {batch_idx}: X_batch shape={np.array(X_batch).shape if len(X_batch) > 0 else 'empty'}, y_batch shape={np.array(y_batch).shape if len(y_batch) > 0 else 'empty'}")
                
                # Check if batch is empty
                if len(X_batch) == 0 or len(y_batch) == 0:
                    if batch_idx < 10:  # Only show first few warnings
                        print(f"  Warning: Empty batch at index {batch_idx}, skipping...")
                    continue
                
                # Since batch_size=1 in generator, each batch should have 1 sample
                if len(X_batch) != 1 or len(y_batch) != 1:
                    print(f"  Warning: Unexpected batch size at index {batch_idx}: X={len(X_batch)}, y={len(y_batch)}")
                    continue
                
                all_X.append(X_batch[0])
                all_y.append(y_batch[0])
                loaded_count += 1
                
                if loaded_count % 1000 == 0:
                    print(f"  Loaded {loaded_count} samples (batch {batch_idx+1}/{len(self.data_generator)})")
                    
            except Exception as e:
                if batch_idx < 10:  # Only show first few errors
                    print(f"  Error loading batch {batch_idx}: {e}")
                continue
        
        # Check if we loaded any data
        if len(all_X) == 0 or len(all_y) == 0:
            raise ValueError(f"No valid data loaded from generator! Split: {self.data_generator.split}")
        
        # Convert to numpy arrays
        self.X_data = np.array(all_X)  # (n_samples, window_size, n_channels)
        self.y_data = np.array(all_y)  # (n_samples, n_trajectory_points)
        
        print(f"Successfully preloaded {len(self.X_data)} samples")
        print(f"  X shape: {self.X_data.shape}")
        print(f"  y shape: {self.y_data.shape}")
        
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            X: (1, n_channels, window_size) - EMG window in PyTorch format
            y: (n_trajectory_points,) - trajectory targets
        """
        # Get EMG window and transpose to PyTorch format
        X = self.X_data[idx]  # (window_size, n_channels)
        X = X.transpose(1, 0)  # (n_channels, window_size)
        X = X[np.newaxis, ...]  # (1, n_channels, window_size) - add channel dim for Conv2D
        
        # Get trajectory targets
        y = self.y_data[idx]  # (n_trajectory_points,)
        
        # Convert to PyTorch tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        return X_tensor, y_tensor


class PyTorchIDMSDataModule:
    """
    Data module for managing train/val/test splits with PyTorch DataLoaders
    """
    
    def __init__(self,
                 dataset_path: str,
                 subjects=None,
                 trials=None,
                 window_size: int = 1000,
                 stride: int = 50,
                 delay: float = 0.05,
                 horizon: float = 0.25,
                 n_trajectory_points: int = 10,
                 batch_size: int = 32,
                 emg_preproc=None,
                 test_ratio: float = 0.05,
                 val_ratio_from_trainval: float = 0.2,
                 seed: int = 42,
                 num_workers: int = 0):
        """
        Initialize data module
        
        Args:
            Same as PyTorchIDMSDataset plus:
            num_workers: Number of worker processes for data loading
        """
        self.dataset_path = dataset_path
        self.subjects = subjects
        self.trials = trials
        self.window_size = window_size
        self.stride = stride
        self.delay = delay
        self.horizon = horizon
        self.n_trajectory_points = n_trajectory_points
        self.batch_size = batch_size
        self.emg_preproc = emg_preproc
        self.test_ratio = test_ratio
        self.val_ratio_from_trainval = val_ratio_from_trainval
        self.seed = seed
        self.num_workers = num_workers
        
    def create_datasets(self):
        """Create train, validation, and test datasets"""
        
        # Create datasets for each split
        self.train_dataset = PyTorchIDMSDataset(
            dataset_path=self.dataset_path,
            subjects=self.subjects,
            trials=self.trials,
            window_size=self.window_size,
            stride=self.stride,
            delay=self.delay,
            horizon=self.horizon,
            n_trajectory_points=self.n_trajectory_points,
            emg_preproc=self.emg_preproc,
            split='train',
            test_ratio=self.test_ratio,
            val_ratio_from_trainval=self.val_ratio_from_trainval,
            seed=self.seed
        )
        
        self.val_dataset = PyTorchIDMSDataset(
            dataset_path=self.dataset_path,
            subjects=self.subjects,
            trials=self.trials,
            window_size=self.window_size,
            stride=self.stride,
            delay=self.delay,
            horizon=self.horizon,
            n_trajectory_points=self.n_trajectory_points,
            emg_preproc=self.emg_preproc,
            split='val',
            test_ratio=self.test_ratio,
            val_ratio_from_trainval=self.val_ratio_from_trainval,
            seed=self.seed
        )
        
        self.test_dataset = PyTorchIDMSDataset(
            dataset_path=self.dataset_path,
            subjects=self.subjects,
            trials=self.trials,
            window_size=self.window_size,
            stride=self.stride,
            delay=self.delay,
            horizon=self.horizon,
            n_trajectory_points=self.n_trajectory_points,
            emg_preproc=self.emg_preproc,
            split='test',
            test_ratio=self.test_ratio,
            val_ratio_from_trainval=self.val_ratio_from_trainval,
            seed=self.seed
        )
        
        print(f"Created datasets:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples") 
        print(f"  Test: {len(self.test_dataset)} samples")
        
    def create_dataloaders(self):
        """Create PyTorch DataLoaders"""
        
        if not hasattr(self, 'train_dataset'):
            self.create_datasets()
            
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Created DataLoaders:")
        print(f"  Train: {len(self.train_loader)} batches")
        print(f"  Val: {len(self.val_loader)} batches")
        print(f"  Test: {len(self.test_loader)} batches")
        
        return self.train_loader, self.val_loader, self.test_loader


def create_idms_dataloaders(
    dataset_path: str = "data/idms_ready_dataset.h5",
    window_size: int = 1000,
    horizon: float = 0.25,
    batch_size: int = 32,
    emg_preproc: str = None,
    subjects=None,
    trials=None
):
    """
    Convenience function to create IDMS data loaders
    
    Args:
        dataset_path: Path to HDF5 dataset
        window_size: EMG window size (samples)
        horizon: Trajectory prediction horizon (seconds)  
        batch_size: Training batch size
        emg_preproc: EMG preprocessing method
        subjects: List of subjects to include
        trials: List of trials to include
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    data_module = PyTorchIDMSDataModule(
        dataset_path=dataset_path,
        subjects=subjects,
        trials=trials,
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size,
        emg_preproc=emg_preproc,
        stride=50,  # Standard stride
        delay=0.05,  # Standard delay
        n_trajectory_points=10,  # Standard trajectory points
        test_ratio=0.05,  # 5% test set
        val_ratio_from_trainval=0.2,  # 20% of remaining for validation
        seed=42,
        num_workers=4 if torch.cuda.is_available() else 0
    )
    
    return data_module.create_dataloaders()


# Example usage and testing
