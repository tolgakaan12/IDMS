"""PyTorch port of the heteroscedastic uncertainty losses (Contribution 3).

Replaces the TensorFlow `losses.py`. The main loss is the heteroscedastic Gaussian
negative log-likelihood (Kendall & Gal, 2017):

    L = 0.5 * exp(-s) * (y - mu)^2 + 0.5 * s        with  s = log(sigma^2)

which auto-attenuates the error term for high-variance predictions while the
0.5*s term prevents collapse to zero variance. Predictions are packed as
`y_pred[..., 0] = mean`, `y_pred[..., 1] = log-variance`, matching the model output.
"""
import torch

LOG_VAR_MIN, LOG_VAR_MAX = -10.0, 3.0  # numerical-stability clamp on log(sigma^2)


def aleatoric_trajectory_loss(y_true, y_pred, loss_weight: float = 1.0,
                              var_regularization: float = 0.0):
    """Heteroscedastic Gaussian NLL over trajectory points.

    Args:
        y_true: (batch, n_times) ground-truth trajectories.
        y_pred: (batch, n_times, 2) with [..., 0]=mean, [..., 1]=log-variance.
        loss_weight: overall scale.
        var_regularization: optional L2 penalty on log-variance.
    Returns:
        scalar loss.
    """
    y_mean = y_pred[..., 0]
    y_log_var = torch.clamp(y_pred[..., 1], LOG_VAR_MIN, LOG_VAR_MAX)

    precision = torch.exp(-y_log_var)              # 1 / sigma^2
    squared_error = (y_true - y_mean) ** 2
    total = 0.5 * precision * squared_error + 0.5 * y_log_var  # (batch, n_times)

    if var_regularization > 0.0:
        total = total + var_regularization * torch.mean(y_log_var ** 2)
    return torch.mean(total) * loss_weight


def uncertainty_mse_loss(y_true, y_pred, uncertainty_weight: float = 1.0,
                         target_log_var: float = -2.0):
    """MSE on the mean + a regulariser pulling log-variance toward a target."""
    y_mean = y_pred[..., 0]
    y_log_var = torch.clamp(y_pred[..., 1], LOG_VAR_MIN, LOG_VAR_MAX)
    mse = torch.mean((y_true - y_mean) ** 2)
    reg = torch.mean((y_log_var - target_log_var) ** 2)
    return mse + uncertainty_weight * reg


def calibrated_uncertainty_loss(y_true, y_pred, temperature: float = 1.0):
    """Temperature-scaled NLL (temperature > 1 inflates uncertainty)."""
    import math
    y_mean = y_pred[..., 0]
    y_log_var = y_pred[..., 1] + math.log(temperature)
    scaled = torch.stack([y_mean, y_log_var], dim=-1)
    return aleatoric_trajectory_loss(y_true, scaled)


@torch.no_grad()
def uncertainty_stats(y_pred) -> dict:
    """Summary of predicted uncertainty (replaces the TF training callback)."""
    log_var = torch.clamp(y_pred[..., 1], LOG_VAR_MIN, LOG_VAR_MAX)
    sigma = torch.exp(0.5 * log_var)
    return {
        "mean_sigma": float(sigma.mean()),
        "min_sigma": float(sigma.min()),
        "max_sigma": float(sigma.max()),
    }
