"""Shared regression metrics for trajectory prediction.

Single source of truth for R²/RMSE/MAE/correlation, which were previously
re-implemented (identically) across the trainer, baseline, and comparison
scripts. All metrics are computed over the flattened trajectory points, exactly
as the thesis reports them, so values are unchanged.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """R²/RMSE/MSE/MAE/correlation over all flattened trajectory points.

    Args:
        y_true, y_pred: arrays of matching shape, e.g. (N, T) trajectory points.
    Returns:
        dict with keys: r2, rmse, mse, mae, correlation.
    """
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    mse = float(mean_squared_error(yt, yp))
    return {
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "mae": float(mean_absolute_error(yt, yp)),
        "correlation": float(np.corrcoef(yp, yt)[0, 1]),
    }


def r2(y_true, y_pred) -> float:
    """Just the flattened R² (convenience for call sites that only need it)."""
    return float(r2_score(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)))


def per_point_metrics(y_true, y_pred) -> Dict[str, np.ndarray]:
    """Per-trajectory-point R² and RMSE. Inputs shaped (N, T)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n_points = yt.shape[1]
    return {
        "r2_per_point": np.array([r2_score(yt[:, t], yp[:, t]) for t in range(n_points)]),
        "rmse_per_point": np.array(
            [np.sqrt(mean_squared_error(yt[:, t], yp[:, t])) for t in range(n_points)]
        ),
    }
