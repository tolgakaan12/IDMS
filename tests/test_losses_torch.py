"""Tests for the PyTorch heteroscedastic uncertainty losses (Contribution 3 port)."""
import torch

from idms.uncertainty.losses_torch import (
    aleatoric_trajectory_loss,
    uncertainty_mse_loss,
    uncertainty_stats,
)


def _pred(mean, log_var):
    return torch.stack([mean, log_var], dim=-1)


def test_matches_manual_formula():
    torch.manual_seed(1)
    y_true = torch.randn(8, 10)
    y_mean = y_true + 0.1 * torch.randn(8, 10)
    log_var = torch.randn(8, 10) - 2.0
    got = aleatoric_trajectory_loss(y_true, _pred(y_mean, log_var))
    s = torch.clamp(log_var, -10.0, 3.0)
    manual = torch.mean(0.5 * torch.exp(-s) * (y_true - y_mean) ** 2 + 0.5 * s)
    assert torch.allclose(got, manual, atol=1e-6)


def test_optimal_log_var_has_zero_gradient():
    """NLL is minimised when log(sigma^2) == log(error^2): gradient there is 0."""
    y_true = torch.zeros(4, 10)
    y_mean = torch.full((4, 10), 0.5)            # constant error -> error^2 = 0.25
    s = torch.log((y_true - y_mean) ** 2).clone().requires_grad_(True)
    aleatoric_trajectory_loss(y_true, _pred(y_mean, s)).backward()
    assert s.grad.abs().max() < 1e-6


def test_perfect_prediction_reduces_to_uncertainty_term():
    """Zero error -> loss = 0.5 * mean(log_var)."""
    y = torch.zeros(4, 10)
    s = torch.full((4, 10), -3.0)
    loss = aleatoric_trajectory_loss(y, _pred(y, s))
    assert torch.allclose(loss, torch.tensor(0.5 * -3.0), atol=1e-6)


def test_extreme_log_var_is_clamped_and_finite():
    y_true = torch.zeros(2, 3)
    y_mean = torch.ones(2, 3)
    for extreme in (-1000.0, 1000.0):
        loss = aleatoric_trajectory_loss(y_true, _pred(y_mean, torch.full((2, 3), extreme)))
        assert torch.isfinite(loss)


def test_var_regularization_increases_loss():
    y = torch.randn(4, 10)
    s = torch.full((4, 10), 2.0)
    base = aleatoric_trajectory_loss(y, _pred(y.clone(), s), var_regularization=0.0)
    reg = aleatoric_trajectory_loss(y, _pred(y.clone(), s), var_regularization=0.1)
    assert reg > base


def test_uncertainty_stats_sensible():
    s = torch.full((4, 10), -2.0)  # sigma = exp(-1) ~ 0.368
    stats = uncertainty_stats(_pred(torch.zeros(4, 10), s))
    assert abs(stats["mean_sigma"] - torch.exp(torch.tensor(-1.0)).item()) < 1e-4
    assert stats["min_sigma"] <= stats["mean_sigma"] <= stats["max_sigma"]
