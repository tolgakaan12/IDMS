#!/usr/bin/env python3
"""
Train Elbow Trajectory Predictor - Method 2 Implementation
=========================================================

Training script for elbow trajectory prediction using:
- IDMSTrajectoryDataGenerator (real future data)
- ElbowTrajEstimator (trajectory parameter expansion)
- Method 2 approach (smooth trajectory fitting)

Date: 2025-08-24
"""

import os
import sys

sys.path.append(".")

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import custom modules
from idms.data.generator import IDMSTrajectoryDataGenerator
from idms.uncertainty.architectures import (
    elbow_trajectory_dual_head_model,
    elbow_trajectory_single_head_model,
)
from idms.uncertainty.model import create_elbow_uncertainty_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU configuration
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)],
    )
    logger.info(f"GPU memory limited to 8GB")


class ElbowTrajectoryTrainer:
    """Training manager for elbow trajectory prediction models."""

    def __init__(self, config):
        """Initialize trainer with configuration."""
        self.config = config
        self.model = None
        self.train_gen = None
        self.val_gen = None
        self.history = None

        # Create output directory
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized. Output dir: {self.output_dir}")

    def setup_data_generators(self):
        """Set up training and validation data generators."""
        logger.info("Setting up data generators...")

        # Training generator
        self.train_gen = IDMSTrajectoryDataGenerator(
            dataset_path=self.config["dataset_path"],
            subjects=self.config.get("train_subjects"),
            window_size=self.config["window_size"],
            stride=self.config["stride"],
            delay=self.config["delay"],
            horizon=self.config["horizon"],
            n_trajectory_points=self.config["n_trajectory_points"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            shuffle_method=self.config["shuffle_method"],
            normalize_emg=self.config.get("normalize_emg", True),
            emg_preproc=self.config.get("emg_preproc", None),
            split="train",
            test_ratio=self.config["test_ratio"],
            val_ratio_from_trainval=self.config["val_ratio_from_trainval"],
            seed=self.config["seed"],
        )

        # Validation generator
        self.val_gen = IDMSTrajectoryDataGenerator(
            dataset_path=self.config["dataset_path"],
            subjects=self.config.get("val_subjects"),
            window_size=self.config["window_size"],
            stride=self.config["stride"] * 2,  # Less dense sampling for validation
            delay=self.config["delay"],
            horizon=self.config["horizon"],
            n_trajectory_points=self.config["n_trajectory_points"],
            batch_size=self.config["batch_size"],
            shuffle=False,  # No shuffling for validation
            shuffle_method="windows",
            normalize_emg=self.config.get("normalize_emg", True),
            emg_preproc=self.config.get("emg_preproc", None),
            split="val",
            test_ratio=self.config["test_ratio"],
            val_ratio_from_trainval=self.config["val_ratio_from_trainval"],
            seed=self.config["seed"],
        )

        # Print data statistics
        train_stats = self.train_gen.get_dataset_stats()
        val_stats = self.val_gen.get_dataset_stats()

        logger.info(
            f"Training data: {train_stats['n_windows']} windows from {train_stats['n_trials']} trials"
        )
        logger.info(
            f"Validation data: {val_stats['n_windows']} windows from {val_stats['n_trials']} trials"
        )
        logger.info(f"EMG window: {train_stats['window_size_seconds']:.3f}s")
        logger.info(
            f"Trajectory: {train_stats['n_trajectory_points']} points over {train_stats['horizon']:.3f}s, with {train_stats['delay']}s delay"
        )

    def build_model(self):
        """Build the trajectory prediction model."""
        logger.info("Building model...")

        input_shape = (self.config["window_size"], len(self.config["emg_channels"]))

        if self.config["model_type"] == "single_head":
            self.model = elbow_trajectory_single_head_model(
                input_shape=input_shape,
                trajectory_samples=self.config["n_trajectory_points"],
                trajectory_horizon=self.config["horizon"],
                **self.config["model_params"],
            )
        elif self.config["model_type"] == "dual_head":
            self.model = elbow_trajectory_dual_head_model(
                input_shape=input_shape,
                trajectory_samples=self.config["n_trajectory_points"],
                trajectory_horizon=self.config["horizon"],
                **self.config["model_params"],
            )
        elif self.config["model_type"] == "uncertainty":
            self.model = create_elbow_uncertainty_model(
                input_shape=input_shape,
                trajectory_samples=self.config["n_trajectory_points"],
                trajectory_horizon=self.config["horizon"],
                **self.config["model_params"],
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")

        logger.info(f"Model built: {self.model.count_params():,} parameters")

        # Save model architecture
        with open(self.output_dir / "model_summary.txt", "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

    def setup_callbacks(self):
        """Set up training callbacks."""
        callbacks = []

        # Model checkpointing
        checkpoint_path = self.output_dir / "best_model.h5"
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                verbose=1,
            )
        )

        # Early stopping
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=self.config["early_stopping_patience"],
                restore_best_weights=True,
                mode="min",
                verbose=1,
            )
        )

        # Learning rate reduction
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.7,
                patience=self.config["lr_reduction_patience"],
                min_lr=1e-6,
                mode="min",
                verbose=1,
            )
        )

        # TensorBoard logging
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        callbacks.append(
            TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                profile_batch=0,  # Disable profiling to save memory
            )
        )

        return callbacks

    def train(self):
        """Train the model."""
        logger.info("Starting training...")

        callbacks = self.setup_callbacks()

        # Train the model
        self.history = self.model.fit(
            self.train_gen,
            epochs=self.config["epochs"],
            validation_data=self.val_gen,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Training completed!")

    def save_results(self):
        """Save training results and create visualizations."""
        logger.info("Saving results...")

        # Save training history
        history_path = self.output_dir / "training_history.npz"
        np.savez(str(history_path), **self.history.history)

        # Save configuration
        import json

        config_path = self.output_dir / "config.json"

        # Make config JSON serializable
        config_to_save = {}
        for key, value in self.config.items():
            if isinstance(value, (Path, np.integer, np.floating)):
                config_to_save[key] = str(value)
            elif (
                isinstance(value, (list, tuple))
                and len(value) > 0
                and isinstance(value[0], (np.integer, np.floating))
            ):
                config_to_save[key] = [float(x) for x in value]
            else:
                config_to_save[key] = value

        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        # Plot training history
        self.plot_training_history()

        # Test model on sample data
        self.evaluate_sample()

    def plot_training_history(self):
        """Plot and save training history."""
        history = self.history.history
        epochs = range(1, len(history["loss"]) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training History", fontsize=16)

        # Loss
        axes[0, 0].plot(epochs, history["loss"], "b-", label="Training Loss")
        axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Validation Loss")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("MSE Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE
        mae_key = "mae" if "mae" in history else "mean_absolute_error"
        val_mae_key = "val_mae" if "val_mae" in history else "val_mean_absolute_error"

        if mae_key in history:
            axes[0, 1].plot(epochs, history[mae_key], "b-", label="Training MAE")
            axes[0, 1].plot(epochs, history[val_mae_key], "r-", label="Validation MAE")
            axes[0, 1].set_title("Mean Absolute Error")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("MAE")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Learning rate (if available)
        if "lr" in history:
            axes[1, 0].plot(epochs, history["lr"], "g-")
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True, alpha=0.3)

        # Model statistics
        final_train_loss = history["loss"][-1]
        final_val_loss = history["val_loss"][-1]
        best_val_loss = min(history["val_loss"])
        best_epoch = history["val_loss"].index(best_val_loss) + 1

        stats_text = f"""Final Training Loss: {final_train_loss:.6f}
Final Validation Loss: {final_val_loss:.6f}
Best Validation Loss: {best_val_loss:.6f} (Epoch {best_epoch})
Total Parameters: {self.model.count_params():,}
Total Epochs: {len(epochs)}"""

        axes[1, 1].text(
            0.1,
            0.5,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )
        axes[1, 1].set_axis_off()

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_history.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        logger.info(
            f"Training plot saved to {self.output_dir / 'training_history.png'}"
        )

    def evaluate_sample(self):
        """Evaluate model on sample data and create visualization."""
        logger.info("Evaluating model on sample data...")

        # Get a batch of validation data
        X_sample, y_true = self.val_gen[0]
        y_pred = self.model.predict(X_sample, verbose=0)

        # Handle dual-head model output
        if isinstance(y_pred, list):
            y_pred = y_pred[0]  # Use angle predictions for visualization

        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))

        # Create trajectory comparison plot
        n_samples = min(4, len(X_sample))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Time points for plotting
        time_points = np.linspace(
            self.config["delay"],
            self.config["delay"] + self.config["horizon"],
            self.config["n_trajectory_points"],
        )

        for i in range(n_samples):
            axes[i].plot(
                time_points,
                y_true[i],
                "b-",
                linewidth=2,
                label="Ground Truth",
                marker="o",
            )
            axes[i].plot(
                time_points,
                y_pred[i],
                "r--",
                linewidth=2,
                label="Prediction",
                marker="s",
            )
            axes[i].set_title(f"Sample {i + 1}")
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Elbow Angle (rad)")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # Calculate sample-specific error
            sample_mae = np.mean(np.abs(y_true[i] - y_pred[i]))
            axes[i].text(
                0.02,
                0.98,
                f"MAE: {sample_mae:.4f}",
                transform=axes[i].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.suptitle(
            f"Trajectory Predictions (MSE: {mse:.6f}, MAE: {mae:.6f})", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "trajectory_predictions.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Sample evaluation: MSE={mse:.6f}, MAE={mae:.6f}")
        logger.info(
            f"Prediction plot saved to {self.output_dir / 'trajectory_predictions.png'}"
        )


def find_next_experiment_number(base_dir="Models/ElbowTrajectory"):
    """Find the next available experiment number."""
    import os

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    experiment_dirs = [
        d
        for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith("experiment_")
    ]
    if not experiment_dirs:
        return 1

    # Extract numbers and find the highest
    numbers = []
    for d in experiment_dirs:
        try:
            num = int(d.name.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    return max(numbers) + 1 if numbers else 1


def main():
    """Main training function."""
    # Auto-increment experiment number
    experiment_num = find_next_experiment_number()

    # Training configuration
    config = {
        # Data parameters
        "dataset_path": "data/idms_ready_dataset.h5",
        "train_subjects": None,  # None = all subjects for now
        "val_subjects": None,
        "emg_channels": ["biceps", "triceps", "bra", "ecu"],
        # Window parameters
        "window_size": 1000,  # 0.5s EMG window (500ms at 2kHz)
        "stride": 25,  # 0.0125s between windows (dense sampling)
        "delay": 0.05,  # 0.05s delay before trajectory (50ms)
        "horizon": 0.5,  # 0.5s trajectory horizon (500ms)
        "n_trajectory_points": 10,  # 10 trajectory samples
        # Training parameters
        "batch_size": 512,
        "epochs": 50,  # Short test run
        "shuffle_method": "windows",  # 'windows' or 'trials'
        "test_ratio": 0.05,  # 5% for test set
        "val_ratio_from_trainval": 0.2,  # 20% of remaining 95% = 19% overall for validation
        "seed": 42,
        # Model parameters
        "model_type": "single_head",  # 'single_head' or 'dual_head'
        "model_params": {
            "depth_mul_in": (3, 3, 2),
            "krnl_in": ((1, 15), (1, 3), (1, 3)),
            "pad": "same",
            "dil": ((1, 1), (1, 15), (1, 45)),
            "dense": (512, 128, 128),
            "acts": ("silu", "silu", "silu"),
            "drp": 0.2,
            "l2_reg": 0.001,
            "learning_rate": 0.002,
            "weight_decay": 0.01,
            "initial_eig": -3.0,
            "trainable_eig": True,
        },
        # Callback parameters
        "early_stopping_patience": 15,
        "lr_reduction_patience": 8,
        # Output - auto-incremented experiment number
        "output_dir": f"Models/ElbowTrajectory/experiment_{experiment_num:03d}",
    }

    # Create trainer and run training
    trainer = ElbowTrajectoryTrainer(config)

    try:
        trainer.setup_data_generators()
        trainer.build_model()
        trainer.train()
        trainer.save_results()

        logger.info(
            f"Training completed successfully! Results saved to: {trainer.output_dir}"
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
