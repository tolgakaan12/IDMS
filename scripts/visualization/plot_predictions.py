#!/usr/bin/env python3
"""
Visualize PyTorch model predictions as instantaneous velocities across full trials.

This script validates R² calculations by:
1. Loading PyTorch TCANet-IDMS model
2. Using first trajectory output as instantaneous velocity prediction
3. Comparing with ground truth instantaneous velocities
4. Plotting predictions vs actual across entire trials
5. Computing detailed metrics for validation

Adapted from visualize_velocity_predictions.py for PyTorch models.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_gen.idms_trajectory_datagenerator import IDMSTrajectoryDataGenerator
from pytorch_models.pytorch_data_adapter import PyTorchIDMSDataModule

# Import PyTorch components
from pytorch_models.tcanet_idms import create_tcanet_idms_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PyTorchVelocityPredictor:
    """Visualize PyTorch model predictions as instantaneous velocities."""

    def __init__(self, experiment_dir: str, dataset_path: str):
        """
        Initialize the PyTorch velocity predictor.

        Args:
            experiment_dir: Path to PyTorch experiment directory
            dataset_path: Path to the HDF5 dataset
        """
        self.experiment_dir = Path(experiment_dir)
        self.dataset_path = dataset_path

        # Load model configuration
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load data splits to get test trials
        splits_path = self.experiment_dir / "data_splits.json"
        self.test_trials = []
        if splits_path.exists():
            with open(splits_path, "r") as f:
                splits_data = json.load(f)
                self.test_trials = splits_data.get("data_splits", {}).get(
                    "test_trials", []
                )

        logger.info(f"Loading PyTorch model from {experiment_dir}")
        logger.info(f"  Subjects: {self.config.get('subjects', 'all')}")
        logger.info(f"  Window size: {self.config['window_size']}")
        logger.info(f"  Horizon: {self.config['horizon']}s")
        if self.test_trials:
            logger.info(f"  Test trials available: {len(self.test_trials)}")

        # Load model
        self.model = self._load_model()

        # Create data generator for getting trial information
        # Load all trials by setting test_ratio very high
        self.data_gen = IDMSTrajectoryDataGenerator(
            dataset_path=dataset_path,
            subjects=self.config.get("subjects"),
            trials=self.config.get("trials"),
            window_size=self.config["window_size"],
            stride=50,  # For comprehensive coverage
            delay=self.config["delay"],
            horizon=self.config["horizon"],
            n_trajectory_points=self.config["n_trajectory_points"],
            batch_size=1,
            shuffle=False,
            split="test",  # Use test split
            test_ratio=1.0,  # Get all trials
            val_ratio_from_trainval=0.0,  # No validation split needed
            seed=42,
        )

        logger.info(
            f"Loaded {len(self.data_gen)} test windows across {len(self.data_gen.trials_data)} trials"
        )

        # Create mapping from trial names to indices
        self.trial_name_to_idx = {}
        for idx, trial in enumerate(self.data_gen.trials_data):
            trial_name = f"{trial['subject']}/{trial['trial']}"
            self.trial_name_to_idx[trial_name] = idx

    def _load_model(self):
        """Load the trained PyTorch model."""

        # Create model architecture
        model = create_tcanet_idms_model(
            window_size=self.config["window_size"],
            n_channels=len(self.config["emg_channels"]),
            trajectory_points=self.config["n_trajectory_points"],
            trajectory_horizon=self.config["horizon"],
            trajectory_delay=self.config["delay"],
        )

        # Load trained weights
        model_path = self.experiment_dir / "best_model.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Handle compiled models (with _orig_mod. prefix)
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                logger.info("Detected compiled model, removing _orig_mod. prefixes")
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("_orig_mod."):
                        new_key = key[10:]  # Remove '_orig_mod.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict

            model.load_state_dict(state_dict)
            logger.info(
                f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
            )
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        logger.info(
            f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        return model

    def get_test_trial_indices(self) -> List[int]:
        """Get indices of test trials that are available in the data generator."""
        test_indices = []
        available_trials = []
        missing_trials = []

        for trial_name in self.test_trials:
            if trial_name in self.trial_name_to_idx:
                trial_idx = self.trial_name_to_idx[trial_name]
                # Check if this trial actually has windows in the data generator
                has_windows = any(
                    entry["trial_idx"] == trial_idx
                    for entry in self.data_gen.trial_windows
                )
                if has_windows:
                    test_indices.append(trial_idx)
                    available_trials.append(trial_name)
                else:
                    missing_trials.append(f"{trial_name} (no windows)")
            else:
                missing_trials.append(f"{trial_name} (not in data)")

        logger.info(f"Found {len(test_indices)} test trials with windows available")
        logger.info(
            f"Available: {available_trials[:5]}{'...' if len(available_trials) > 5 else ''}"
        )
        if missing_trials:
            logger.info(
                f"Missing {len(missing_trials)} test trials: {missing_trials[:3]}{'...' if len(missing_trials) > 3 else ''}"
            )

        return test_indices

    def predict_trial_velocities(
        self, trial_idx: int, trajectory_point: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict velocities across an entire trial.

        Args:
            trial_idx: Index of trial in data generator
            trajectory_point: Which trajectory point to use (0-9, default: 0)

        Returns:
            time_points, predicted_velocities, actual_velocities_mocap, actual_velocities_trajectory
        """

        trial = self.data_gen.trials_data[trial_idx]
        logger.info(f"Processing trial: {trial['subject']}/{trial['trial']}")

        # Get all windows for this trial
        trial_windows = []
        for entry in self.data_gen.trial_windows:
            if entry["trial_idx"] == trial_idx:
                # Safely get window indices, checking bounds
                for w_idx in entry["window_indices"]:
                    if w_idx < len(self.data_gen.window_indices):
                        trial_windows.append(self.data_gen.window_indices[w_idx])
                    else:
                        logger.debug(
                            f"Window index {w_idx} out of range (max: {len(self.data_gen.window_indices) - 1})"
                        )

        if not trial_windows:
            logger.warning(f"No windows found for trial {trial_idx}")
            return np.array([]), np.array([]), np.array([]), np.array([])

        logger.info(f"Found {len(trial_windows)} windows for this trial")

        # Sort windows by EMG start time
        trial_windows.sort(key=lambda w: w["emg_start"])

        predicted_velocities = []
        actual_velocities_mocap = []
        actual_velocities_trajectory = []
        time_points = []

        # Calculate trajectory times and get the specified point
        trajectory_times = np.linspace(
            self.config["delay"],
            self.config["delay"] + self.config["horizon"],
            self.config["n_trajectory_points"],
        )

        if trajectory_point >= len(trajectory_times):
            raise ValueError(
                f"Trajectory point {trajectory_point} out of range. Max: {len(trajectory_times) - 1}"
            )

        dt = trajectory_times[
            trajectory_point
        ]  # Time from current to specified prediction point
        logger.info(f"Using trajectory point {trajectory_point} at time {dt:.3f}s")

        with torch.no_grad():
            for window_info in trial_windows:
                try:
                    # Get EMG window
                    emg_start, emg_end = (
                        window_info["emg_start"],
                        window_info["emg_end"],
                    )
                    emg_window = trial["emg"][
                        :, emg_start:emg_end
                    ].T  # (window_size, n_channels)

                    # Convert to PyTorch format
                    X = torch.from_numpy(
                        emg_window.T[np.newaxis, np.newaxis, ...]
                    ).float()  # (1, 1, n_channels, window_size)

                    # Get prediction
                    pred_traj = self.model(X)[0].numpy()  # (n_trajectory_points,)

                    # Convert specified trajectory point to velocity by dividing by time interval
                    pred_velocity = pred_traj[trajectory_point] / dt

                    # Get actual velocities at prediction time
                    pred_time_idx = emg_end + int(dt * 2000)  # delay in samples
                    if pred_time_idx < len(trial["velocity"]) and pred_time_idx < len(
                        trial["angle"]
                    ):
                        # Method 1: Direct mocap velocity
                        actual_velocity_mocap = trial["velocity"][pred_time_idx]

                        # Method 2: Calculate velocity from trajectory points (same as prediction method)
                        angle_start = trial["angle"][
                            emg_end
                        ]  # Angle at end of EMG window
                        angle_future = trial["angle"][
                            pred_time_idx
                        ]  # Angle at prediction time
                        angle_diff = angle_future - angle_start  # Angular difference
                        actual_velocity_trajectory = angle_diff / dt  # Angular velocity

                        time_point = pred_time_idx / 2000.0  # Convert to seconds

                        predicted_velocities.append(pred_velocity)
                        actual_velocities_mocap.append(actual_velocity_mocap)
                        actual_velocities_trajectory.append(actual_velocity_trajectory)
                        time_points.append(time_point)

                except Exception as e:
                    logger.debug(f"Error processing window: {e}")
                    continue

        return (
            np.array(time_points),
            np.array(predicted_velocities),
            np.array(actual_velocities_mocap),
            np.array(actual_velocities_trajectory),
        )

    def compute_detailed_metrics(
        self, pred_velocities: np.ndarray, actual_velocities: np.ndarray
    ) -> Dict[str, float]:
        """Compute detailed metrics for validation."""

        if len(pred_velocities) == 0 or len(actual_velocities) == 0:
            return {}

        # Remove any NaN or infinite values
        mask = np.isfinite(pred_velocities) & np.isfinite(actual_velocities)
        pred_clean = pred_velocities[mask]
        actual_clean = actual_velocities[mask]

        if len(pred_clean) == 0:
            return {}

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(actual_clean, pred_clean))
        mae = mean_absolute_error(actual_clean, pred_clean)
        r2 = r2_score(actual_clean, pred_clean)

        # Additional metrics
        correlation = (
            np.corrcoef(actual_clean, pred_clean)[0, 1]
            if len(actual_clean) > 1
            else 0.0
        )
        max_error = np.max(np.abs(actual_clean - pred_clean))

        # Manual R² calculation for verification
        ss_res = np.sum((actual_clean - pred_clean) ** 2)
        ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
        r2_manual = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "n_points": len(pred_clean),
            "rmse": rmse,
            "mae": mae,
            "r2_sklearn": r2,
            "r2_manual": r2_manual,
            "correlation": correlation,
            "max_error": max_error,
            "pred_mean": np.mean(pred_clean),
            "pred_std": np.std(pred_clean),
            "actual_mean": np.mean(actual_clean),
            "actual_std": np.std(actual_clean),
        }

    def plot_trial_comparison(
        self,
        trial_idx: int,
        save_path: str = None,
        trajectory_point: int = 0,
        include_mocap: bool = False,
    ):
        """Plot velocity predictions vs actual for a single trial."""

        (
            time_points,
            pred_velocities,
            actual_velocities_mocap,
            actual_velocities_trajectory,
        ) = self.predict_trial_velocities(trial_idx, trajectory_point)

        if len(time_points) == 0:
            logger.warning(f"No valid predictions for trial {trial_idx}")
            return

        trial = self.data_gen.trials_data[trial_idx]
        trial_name = f"{trial['subject']}/{trial['trial']}"

        # Compute metrics for trajectory velocity (always)
        metrics_trajectory = self.compute_detailed_metrics(
            pred_velocities, actual_velocities_trajectory
        )

        # Compute mocap metrics only if requested
        metrics_mocap = None
        gt_comparison = None
        if include_mocap:
            metrics_mocap = self.compute_detailed_metrics(
                pred_velocities, actual_velocities_mocap
            )
            gt_comparison = self.compute_detailed_metrics(
                actual_velocities_mocap, actual_velocities_trajectory
            )

        # Create plot with appropriate number of rows
        if include_mocap:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Time series plot
        if include_mocap:
            ax1.plot(
                time_points,
                actual_velocities_mocap,
                "g-",
                linewidth=2,
                label="Mocap Velocity",
                alpha=0.8,
            )
        ax1.plot(
            time_points,
            actual_velocities_trajectory,
            "b-",
            linewidth=2,
            label="Trajectory Velocity",
            alpha=0.8,
        )
        ax1.plot(
            time_points,
            pred_velocities,
            "r--",
            linewidth=2,
            label="PyTorch Prediction",
            alpha=0.8,
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity (rad/s)")
        if include_mocap:
            ax1.set_title(
                f"Velocity Predictions vs Ground Truth Methods - {trial_name}"
            )
        else:
            ax1.set_title(f"Velocity Predictions vs Trajectory Method - {trial_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add metrics text
        if include_mocap and metrics_mocap and metrics_trajectory:
            metrics_text = f"vs Mocap: R²={metrics_mocap['r2_sklearn']:.4f}, RMSE={metrics_mocap['rmse']:.4f} vs Trajectory: R²={metrics_trajectory['r2_sklearn']:.4f}, RMSE={metrics_trajectory['rmse']:.4f} GT Comparison: R²={gt_comparison['r2_sklearn']:.4f} ({gt_comparison['n_points']} points)"
        elif metrics_trajectory:
            metrics_text = f"vs Trajectory: R²={metrics_trajectory['r2_sklearn']:.4f}, RMSE={metrics_trajectory['rmse']:.4f} ({metrics_trajectory['n_points']} points)"

        if metrics_trajectory:  # Always show if we have trajectory metrics
            ax1.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        if include_mocap:
            # Scatter plot: Prediction vs Mocap
            ax2.scatter(
                actual_velocities_mocap, pred_velocities, alpha=0.6, s=20, c="green"
            )
            min_val = min(np.min(actual_velocities_mocap), np.min(pred_velocities))
            max_val = max(np.max(actual_velocities_mocap), np.max(pred_velocities))
            ax2.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                linewidth=2,
                label="Perfect Prediction",
            )
            ax2.set_xlabel("Mocap Velocity (rad/s)")
            ax2.set_ylabel("Predicted Velocity (rad/s)")
            ax2.set_title(
                f"Prediction vs Mocap Velocity (R² = {metrics_mocap['r2_sklearn']:.4f})"
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axis("equal")

            # Scatter plot: Prediction vs Trajectory (use ax3 when mocap is included)
            ax3.scatter(
                actual_velocities_trajectory, pred_velocities, alpha=0.6, s=20, c="blue"
            )
            min_val = min(np.min(actual_velocities_trajectory), np.min(pred_velocities))
            max_val = max(np.max(actual_velocities_trajectory), np.max(pred_velocities))
            ax3.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                linewidth=2,
                label="Perfect Prediction",
            )
            ax3.set_xlabel("Trajectory Velocity (rad/s)")
            ax3.set_ylabel("Predicted Velocity (rad/s)")
            ax3.set_title(
                f"Prediction vs Trajectory Velocity (R² = {metrics_trajectory['r2_sklearn']:.4f})"
            )
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axis("equal")
        else:
            # Only trajectory scatter plot (use ax2 when mocap is not included)
            ax2.scatter(
                actual_velocities_trajectory, pred_velocities, alpha=0.6, s=20, c="blue"
            )
            min_val = min(np.min(actual_velocities_trajectory), np.min(pred_velocities))
            max_val = max(np.max(actual_velocities_trajectory), np.max(pred_velocities))
            ax2.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                linewidth=2,
                label="Perfect Prediction",
            )
            ax2.set_xlabel("Trajectory Velocity (rad/s)")
            ax2.set_ylabel("Predicted Velocity (rad/s)")
            ax2.set_title(
                f"Prediction vs Trajectory Velocity (R² = {metrics_trajectory['r2_sklearn']:.4f})"
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axis("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.show()

        # Print detailed metrics
        print(f"Detailed Metrics for {trial_name}:")
        if include_mocap and metrics_mocap:
            print(f"  vs Mocap Velocity:")
            print(f"    R² (sklearn): {metrics_mocap['r2_sklearn']:.6f}")
            print(f"    RMSE: {metrics_mocap['rmse']:.6f} rad/s")
            print(f"    MAE: {metrics_mocap['mae']:.6f} rad/s")
        if metrics_trajectory:
            print(f"  vs Trajectory Velocity:")
            print(f"    R² (sklearn): {metrics_trajectory['r2_sklearn']:.6f}")
            print(f"    RMSE: {metrics_trajectory['rmse']:.6f} rad/s")
            print(f"    MAE: {metrics_trajectory['mae']:.6f} rad/s")
        if include_mocap and gt_comparison:
            print(f"  Ground Truth Comparison (Mocap vs Trajectory):")
            print(f"    R²: {gt_comparison['r2_sklearn']:.6f}")
            print(f"    RMSE: {gt_comparison['rmse']:.6f} rad/s")
            print(f"    Points: {gt_comparison['n_points']}")

        return {
            "mocap": metrics_mocap,
            "trajectory": metrics_trajectory,
            "gt_comparison": gt_comparison,
        }

    def evaluate_all_trials(
        self,
        max_trials: int = 5,
        use_test_trials: bool = True,
        trajectory_point: int = 0,
        include_mocap: bool = False,
    ):
        """Evaluate velocity predictions across multiple trials."""

        # Get trial indices to use
        if use_test_trials and self.test_trials:
            test_indices = self.get_test_trial_indices()
            trial_indices = (
                test_indices[:max_trials]
                if test_indices
                else list(range(min(len(self.data_gen.trials_data), max_trials)))
            )
            logger.info(f"Using {len(trial_indices)} test trials from data_splits.json")
        else:
            trial_indices = list(range(min(len(self.data_gen.trials_data), max_trials)))
            logger.info(f"Using first {len(trial_indices)} available trials")

        logger.info(
            f"Evaluating velocity predictions across {len(trial_indices)} trials..."
        )

        all_metrics = []

        # Create plots directory in experiment folder
        plots_dir = self.experiment_dir / "velocity_validation_plots"
        plots_dir.mkdir(exist_ok=True)

        for i, trial_idx in enumerate(trial_indices):
            trial = self.data_gen.trials_data[trial_idx]
            trial_name = f"{trial['subject']}/{trial['trial']}"
            logger.info(f"Processing trial {i + 1}/{len(trial_indices)}: {trial_name}")

            # Save plot with descriptive name
            plot_filename = (
                f"velocity_validation_{trial['subject']}_{trial['trial']}.png"
            )
            save_path = plots_dir / plot_filename

            metrics = self.plot_trial_comparison(
                trial_idx,
                save_path=str(save_path),
                trajectory_point=trajectory_point,
                include_mocap=include_mocap,
            )
            if metrics:
                metrics["trial_idx"] = trial_idx
                metrics["trial_name"] = trial_name
                all_metrics.append(metrics)

        # Overall summary
        if all_metrics:
            print(f"\n{'=' * 60}")
            print(f"OVERALL VELOCITY PREDICTION SUMMARY")
            print(f"{'=' * 60}")

            # Extract metrics
            traj_r2_scores = [
                m["trajectory"]["r2_sklearn"]
                for m in all_metrics
                if m.get("trajectory")
            ]
            traj_rmse_scores = [
                m["trajectory"]["rmse"] for m in all_metrics if m.get("trajectory")
            ]
            total_points = sum(
                [
                    m["trajectory"]["n_points"]
                    for m in all_metrics
                    if m.get("trajectory")
                ]
            )

            print(f"Trials evaluated: {len(all_metrics)}")
            print(f"Total prediction points: {total_points}")
            print(f"Plots saved to: {plots_dir}")
            print(f"Average Performance:")

            if include_mocap:
                mocap_r2_scores = [
                    m["mocap"]["r2_sklearn"] for m in all_metrics if m.get("mocap")
                ]
                mocap_rmse_scores = [
                    m["mocap"]["rmse"] for m in all_metrics if m.get("mocap")
                ]
                print(f"  vs Mocap Velocity:")
                print(f"    R²: {np.mean(mocap_r2_scores):.6f}")
                print(f"    RMSE: {np.mean(mocap_rmse_scores):.6f} rad/s")

            print(f"  vs Trajectory Velocity:")
            print(f"    R²: {np.mean(traj_r2_scores):.6f}")
            print(f"    RMSE: {np.mean(traj_rmse_scores):.6f} rad/s")

            if include_mocap:
                print(f"Per-trial R² values (Mocap | Trajectory):")
                for m in all_metrics:
                    mocap_r2 = m["mocap"]["r2_sklearn"] if m.get("mocap") else 0
                    traj_r2 = (
                        m["trajectory"]["r2_sklearn"] if m.get("trajectory") else 0
                    )
                    print(f"  {m['trial_name']}: {mocap_r2:.4f} | {traj_r2:.4f}")
            else:
                print(f"Per-trial R² values (Trajectory):")
                for m in all_metrics:
                    traj_r2 = (
                        m["trajectory"]["r2_sklearn"] if m.get("trajectory") else 0
                    )
                    print(f"  {m['trial_name']}: {traj_r2:.4f}")

        return all_metrics

    def test_all_trajectory_points(
        self, trial_idx: int = 0, compare_velocity_methods: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """Test all 10 trajectory points as velocity predictors."""

        # Calculate trajectory times based on config
        trajectory_times = np.linspace(
            self.config["delay"],
            self.config["delay"] + self.config["horizon"],
            self.config["n_trajectory_points"],
        )

        trial = self.data_gen.trials_data[trial_idx]
        logger.info(
            f"Testing all trajectory points for trial: {trial['subject']}/{trial['trial']}"
        )
        logger.info(f"Trajectory times: {trajectory_times}")

        # Get all windows for this trial
        trial_windows = []
        for entry in self.data_gen.trial_windows:
            if entry["trial_idx"] == trial_idx:
                # Safely get window indices, checking bounds
                for w_idx in entry["window_indices"]:
                    if w_idx < len(self.data_gen.window_indices):
                        trial_windows.append(self.data_gen.window_indices[w_idx])
                    else:
                        logger.debug(
                            f"Window index {w_idx} out of range (max: {len(self.data_gen.window_indices) - 1})"
                        )

        if not trial_windows:
            logger.warning(f"No windows found for trial {trial_idx}")
            return {}

        # Sort windows by EMG start time
        trial_windows.sort(key=lambda w: w["emg_start"])
        logger.info(f"Found {len(trial_windows)} windows for this trial")

        point_metrics = {}

        for point_idx in range(self.config["n_trajectory_points"]):
            predicted_velocities = []
            actual_velocities_mocap = []
            actual_velocities_trajectory = []

            dt = trajectory_times[
                point_idx
            ]  # Time from current to this prediction point

            with torch.no_grad():
                for window_info in trial_windows:
                    try:
                        # Get EMG window
                        emg_start, emg_end = (
                            window_info["emg_start"],
                            window_info["emg_end"],
                        )
                        emg_window = trial["emg"][
                            :, emg_start:emg_end
                        ].T  # (window_size, n_channels)

                        # Convert to PyTorch format
                        X = torch.from_numpy(
                            emg_window.T[np.newaxis, np.newaxis, ...]
                        ).float()

                        # Get prediction
                        pred_traj = self.model(X)[0].numpy()

                        # Convert trajectory point to velocity
                        pred_velocity = pred_traj[point_idx] / dt

                        # Get actual velocity at prediction time
                        pred_time_idx = emg_end + int(
                            dt * 2000
                        )  # trajectory time in samples
                        if pred_time_idx < len(
                            trial["velocity"]
                        ) and pred_time_idx < len(trial["angle"]):
                            # Method 1: Direct mocap velocity
                            actual_velocity_mocap = trial["velocity"][pred_time_idx]

                            # Method 2: Calculate velocity from trajectory points (same as prediction method)
                            angle_start = trial["angle"][
                                emg_end
                            ]  # Angle at end of EMG window
                            angle_future = trial["angle"][
                                pred_time_idx
                            ]  # Angle at prediction time
                            angle_diff = (
                                angle_future - angle_start
                            )  # Angular difference
                            actual_velocity_trajectory = (
                                angle_diff / dt
                            )  # Angular velocity

                            predicted_velocities.append(pred_velocity)
                            actual_velocities_mocap.append(actual_velocity_mocap)
                            actual_velocities_trajectory.append(
                                actual_velocity_trajectory
                            )

                    except Exception as e:
                        logger.debug(
                            f"Error processing window for point {point_idx}: {e}"
                        )
                        continue

            # Compute metrics for this trajectory point
            if len(predicted_velocities) > 0:
                # Compare with mocap velocity
                metrics_mocap = self.compute_detailed_metrics(
                    np.array(predicted_velocities), np.array(actual_velocities_mocap)
                )

                # Compare with trajectory-calculated velocity
                metrics_trajectory = self.compute_detailed_metrics(
                    np.array(predicted_velocities),
                    np.array(actual_velocities_trajectory),
                )

                if metrics_mocap and metrics_trajectory:
                    combined_metrics = {
                        "trajectory_time": dt,
                        "r2_vs_mocap": metrics_mocap["r2_sklearn"],
                        "rmse_vs_mocap": metrics_mocap["rmse"],
                        "r2_vs_trajectory": metrics_trajectory["r2_sklearn"],
                        "rmse_vs_trajectory": metrics_trajectory["rmse"],
                        "n_points": metrics_mocap["n_points"],
                        "mocap_mean": np.mean(actual_velocities_mocap),
                        "trajectory_mean": np.mean(actual_velocities_trajectory),
                        "pred_mean": np.mean(predicted_velocities),
                    }

                    # Also compare the two ground truth methods
                    if compare_velocity_methods:
                        gt_comparison = self.compute_detailed_metrics(
                            np.array(actual_velocities_mocap),
                            np.array(actual_velocities_trajectory),
                        )
                        if gt_comparison:
                            combined_metrics["gt_methods_r2"] = gt_comparison[
                                "r2_sklearn"
                            ]
                            combined_metrics["gt_methods_rmse"] = gt_comparison["rmse"]

                    point_metrics[point_idx] = combined_metrics

                    if compare_velocity_methods:
                        print(
                            f"Point {point_idx} (t={dt:.3f}s): R²[mocap]={combined_metrics['r2_vs_mocap']:.4f}, R²[traj]={combined_metrics['r2_vs_trajectory']:.4f}, GT_comparison_R²={combined_metrics.get('gt_methods_r2', 0):.4f}"
                        )
                    else:
                        print(
                            f"Point {point_idx} (t={dt:.3f}s): R²[mocap]={combined_metrics['r2_vs_mocap']:.4f}, R²[traj]={combined_metrics['r2_vs_trajectory']:.4f}"
                        )

        # Summary
        if point_metrics:
            print(f"\nSummary for {trial['subject']}/{trial['trial']}:")
            r2_mocap_scores = [m["r2_vs_mocap"] for m in point_metrics.values()]
            r2_traj_scores = [m["r2_vs_trajectory"] for m in point_metrics.values()]

            best_mocap = max(
                point_metrics.keys(), key=lambda k: point_metrics[k]["r2_vs_mocap"]
            )
            best_traj = max(
                point_metrics.keys(), key=lambda k: point_metrics[k]["r2_vs_trajectory"]
            )

            print(
                f"Best vs mocap: point {best_mocap} (t={point_metrics[best_mocap]['trajectory_time']:.3f}s) R²={point_metrics[best_mocap]['r2_vs_mocap']:.4f}"
            )
            print(
                f"Best vs trajectory: point {best_traj} (t={point_metrics[best_traj]['trajectory_time']:.3f}s) R²={point_metrics[best_traj]['r2_vs_trajectory']:.4f}"
            )
            print(f"Average R² vs mocap: {np.mean(r2_mocap_scores):.4f}")
            print(f"Average R² vs trajectory: {np.mean(r2_traj_scores):.4f}")

            if compare_velocity_methods:
                gt_r2_scores = [
                    m.get("gt_methods_r2", 0) for m in point_metrics.values()
                ]
                print(
                    f"Ground truth methods comparison (avg R²): {np.mean(gt_r2_scores):.4f}"
                )

        return point_metrics

    def plot_trajectory_points_r2_table(
        self, trial_idx: int = 0, save_path: str = None, include_mocap: bool = False
    ):
        """Create a table plot showing R² values for each trajectory point as velocity predictor."""

        # Get metrics for all trajectory points
        point_metrics = self.test_all_trajectory_points(
            trial_idx, compare_velocity_methods=include_mocap
        )

        if not point_metrics:
            logger.warning(
                f"No trajectory point metrics available for trial {trial_idx}"
            )
            return

        trial = self.data_gen.trials_data[trial_idx]
        trial_name = f"{trial['subject']}/{trial['trial']}"

        # Prepare data for table
        trajectory_times = [
            point_metrics[i]["trajectory_time"] for i in sorted(point_metrics.keys())
        ]
        r2_trajectory = [
            point_metrics[i]["r2_vs_trajectory"] for i in sorted(point_metrics.keys())
        ]
        rmse_trajectory = [
            point_metrics[i]["rmse_vs_trajectory"] for i in sorted(point_metrics.keys())
        ]
        n_points = [point_metrics[i]["n_points"] for i in sorted(point_metrics.keys())]

        if include_mocap:
            r2_mocap = [
                point_metrics[i]["r2_vs_mocap"] for i in sorted(point_metrics.keys())
            ]
            rmse_mocap = [
                point_metrics[i]["rmse_vs_mocap"] for i in sorted(point_metrics.keys())
            ]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot R² values
        x_pos = np.arange(len(trajectory_times))

        if include_mocap:
            width = 0.35
            bars1 = ax1.bar(
                x_pos - width / 2,
                r2_mocap,
                width,
                label="vs Mocap Velocity",
                color="green",
                alpha=0.7,
            )
            bars2 = ax1.bar(
                x_pos + width / 2,
                r2_trajectory,
                width,
                label="vs Trajectory Velocity",
                color="blue",
                alpha=0.7,
            )

            # Add value labels on bars
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                ax1.text(
                    bar1.get_x() + bar1.get_width() / 2.0,
                    height1 + 0.005,
                    f"{height1:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
                ax1.text(
                    bar2.get_x() + bar2.get_width() / 2.0,
                    height2 + 0.005,
                    f"{height2:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            bars2 = ax1.bar(
                x_pos,
                r2_trajectory,
                label="vs Trajectory Velocity",
                color="blue",
                alpha=0.7,
            )

            # Add value labels on bars
            for i, bar2 in enumerate(bars2):
                height2 = bar2.get_height()
                ax1.text(
                    bar2.get_x() + bar2.get_width() / 2.0,
                    height2 + 0.005,
                    f"{height2:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax1.set_xlabel("Trajectory Point")
        ax1.set_ylabel("R² Score")
        ax1.set_title(
            f"R² Scores for Each Trajectory Point as Velocity Predictor - {trial_name}"
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(
            [f"P{i}\n({t:.3f}s)" for i, t in enumerate(trajectory_times)]
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Plot RMSE values
        if include_mocap:
            bars3 = ax2.bar(
                x_pos - width / 2,
                rmse_mocap,
                width,
                label="vs Mocap Velocity",
                color="green",
                alpha=0.7,
            )
            bars4 = ax2.bar(
                x_pos + width / 2,
                rmse_trajectory,
                width,
                label="vs Trajectory Velocity",
                color="blue",
                alpha=0.7,
            )

            # Add value labels on bars
            for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
                height3 = bar3.get_height()
                height4 = bar4.get_height()
                ax2.text(
                    bar3.get_x() + bar3.get_width() / 2.0,
                    height3 + max(rmse_mocap + rmse_trajectory) * 0.01,
                    f"{height3:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
                ax2.text(
                    bar4.get_x() + bar4.get_width() / 2.0,
                    height4 + max(rmse_mocap + rmse_trajectory) * 0.01,
                    f"{height4:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            bars4 = ax2.bar(
                x_pos,
                rmse_trajectory,
                label="vs Trajectory Velocity",
                color="blue",
                alpha=0.7,
            )

            # Add value labels on bars
            for i, bar4 in enumerate(bars4):
                height4 = bar4.get_height()
                ax2.text(
                    bar4.get_x() + bar4.get_width() / 2.0,
                    height4 + max(rmse_trajectory) * 0.01,
                    f"{height4:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax2.set_xlabel("Trajectory Point")
        ax2.set_ylabel("RMSE (rad/s)")
        ax2.set_title(
            f"RMSE for Each Trajectory Point as Velocity Predictor - {trial_name}"
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(
            [f"P{i}\n({t:.3f}s)" for i, t in enumerate(trajectory_times)]
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add summary text
        best_traj_idx = np.argmax(r2_trajectory)

        if include_mocap:
            best_mocap_idx = np.argmax(r2_mocap)
            summary_text = f"""Summary:
Best vs Mocap: Point {best_mocap_idx} (t={trajectory_times[best_mocap_idx]:.3f}s) R²={r2_mocap[best_mocap_idx]:.4f}
Best vs Trajectory: Point {best_traj_idx} (t={trajectory_times[best_traj_idx]:.3f}s) R²={r2_trajectory[best_traj_idx]:.4f}
Avg R² vs Mocap: {np.mean(r2_mocap):.4f}
Avg R² vs Trajectory: {np.mean(r2_trajectory):.4f}
Data points: {n_points[0]}"""
        else:
            summary_text = f"""Summary:
Best Trajectory: Point {best_traj_idx} (t={trajectory_times[best_traj_idx]:.3f}s) R²={r2_trajectory[best_traj_idx]:.4f}
Avg R² vs Trajectory: {np.mean(r2_trajectory):.4f}
Data points: {n_points[0]}"""

        fig.text(
            0.02,
            0.02,
            summary_text,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for summary text

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"R² table plot saved to {save_path}")

        plt.show()

        return point_metrics


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize PyTorch velocity predictions"
    )
    parser.add_argument("experiment_dir", help="Path to PyTorch experiment directory")
    parser.add_argument(
        "--dataset", default="data/idms_ready_dataset.h5", help="Path to dataset file"
    )
    parser.add_argument(
        "--trial_idx", type=int, help="Specific trial index to visualize"
    )
    parser.add_argument(
        "--max_trials", type=int, default=3, help="Maximum number of trials to evaluate"
    )
    parser.add_argument(
        "--test_all_points",
        action="store_true",
        help="Test all 10 trajectory points as velocity predictors",
    )
    parser.add_argument(
        "--plot_r2_table",
        action="store_true",
        help="Create R² table plot for all trajectory points",
    )
    parser.add_argument(
        "--point",
        type=int,
        default=0,
        help="Which trajectory point to use for velocity prediction (0-9, default: 0)",
    )
    parser.add_argument(
        "--mocap",
        action="store_true",
        help="Include mocap velocities in plots and analysis (default: trajectory only)",
    )

    args = parser.parse_args()

    try:
        # Create predictor
        predictor = PyTorchVelocityPredictor(args.experiment_dir, args.dataset)

        if args.test_all_points:
            # Test all trajectory points
            trial_idx = args.trial_idx if args.trial_idx is not None else 0
            logger.info(f"Testing all trajectory points for trial {trial_idx}")
            predictor.test_all_trajectory_points(trial_idx)
        elif args.plot_r2_table:
            # Create R² table plot using a test trial
            if args.trial_idx is not None:
                trial_idx = args.trial_idx
            else:
                # Use first available test trial if none specified
                test_indices = predictor.get_test_trial_indices()
                trial_idx = test_indices[0] if test_indices else 0

            trial = predictor.data_gen.trials_data[trial_idx]
            logger.info(
                f"Creating R² table plot for test trial {trial_idx}: {trial['subject']}/{trial['trial']}"
            )

            # Save in plots directory
            plots_dir = predictor.experiment_dir / "velocity_validation_plots"
            plots_dir.mkdir(exist_ok=True)
            plot_filename = f"r2_table_{trial['subject']}_{trial['trial']}.png"
            save_path = plots_dir / plot_filename

            predictor.plot_trajectory_points_r2_table(
                trial_idx, save_path=str(save_path), include_mocap=args.mocap
            )
        elif args.trial_idx is not None:
            # Single trial
            logger.info(
                f"Analyzing single trial: {args.trial_idx} using trajectory point {args.point} (mocap: {args.mocap})"
            )
            predictor.plot_trial_comparison(
                args.trial_idx, trajectory_point=args.point, include_mocap=args.mocap
            )
        else:
            # Multiple trials
            logger.info(
                f"Analyzing up to {args.max_trials} trials using trajectory point {args.point} (mocap: {args.mocap})"
            )
            predictor.evaluate_all_trials(
                args.max_trials, trajectory_point=args.point, include_mocap=args.mocap
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
