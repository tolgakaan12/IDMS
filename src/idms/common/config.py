"""Canonical data/trajectory configuration — the single source of truth.

These constants were previously scattered (and sometimes inconsistent) across the
data generator, the torch adapter, and the training scripts. The values here are
the ones used for the thesis results (see the reproduced test R2 = 0.7982814312).

Notable footgun this fixes: several call sites defaulted `horizon` to 0.5s, but the
thesis used a 0.25s horizon (10 points over [0.05, 0.30]s); and `test_ratio`
defaulted to 0.2 in the generator but 0.05 in the adapter. Both are unified here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Windowing + split configuration for the elbow trajectory dataset."""

    # EMG windowing
    sampling_rate_hz: int = 2000
    window_size: int = 1000          # 0.5 s at 2 kHz
    stride: int = 50                 # 0.025 s between windows
    emg_channels: List[str] = field(default_factory=lambda: ["biceps", "triceps", "bra", "ecu"])

    # Trajectory target
    delay: float = 0.05              # s before the prediction horizon starts
    horizon: float = 0.25            # s of predicted trajectory (thesis value; NOT 0.5)
    n_trajectory_points: int = 10    # uniform samples over [delay, delay + horizon]

    # Splits (deterministic)
    test_ratio: float = 0.05
    val_ratio_from_trainval: float = 0.2
    seed: int = 42

    # Training
    batch_size: int = 32
