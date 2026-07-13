"""Correctness tests for the PyTorch Jacobian trajectory layer (Contribution 3 port).

The load-bearing test is `test_jacobian_matches_autograd`: the *analytical* Jacobian
must equal `torch.autograd`'s Jacobian of the trajectory (autograd is exact ground
truth). We also check the trajectory itself matches the C1 estimator's torch
trajectory layer, and that uncertainty propagation produces sensible values.
"""
import numpy as np
import torch

from idms.uncertainty.jacobian_torch import JacobianTrajEstimator


def _layer(eig=-3.0):
    return JacobianTrajEstimator(eig=eig, time_samples=np.linspace(0.05, 0.30, 10))


def test_shapes():
    layer = _layer()
    params = torch.randn(8, 3)
    traj, jac = layer(params)
    assert traj.shape == (8, 10)
    assert jac.shape == (8, 10, 3)
    assert torch.isfinite(traj).all() and torch.isfinite(jac).all()


def test_jacobian_matches_autograd():
    """Analytical Jacobian == autograd Jacobian of the trajectory (to ~1e-5)."""
    for eig in (-1.5, -3.0, -3.567):
        layer = _layer(eig)
        p = torch.tensor([0.4, -0.2, 0.8], dtype=torch.float32)

        def traj_fn(x):  # (3,) -> (n_times,)
            return layer.trajectory(x.unsqueeze(0)).squeeze(0)

        jac_autograd = torch.autograd.functional.jacobian(traj_fn, p)  # (n_times, 3)
        jac_analytical = layer.jacobians(1)[0]                          # (n_times, 3)
        assert torch.allclose(jac_autograd, jac_analytical, atol=1e-5), (
            f"eig={eig}: max diff {(jac_autograd - jac_analytical).abs().max().item():.2e}"
        )


def test_trajectory_matches_c1_estimator():
    """The torch trajectory equals the C1 estimator's PyTorchTrajEstimator output."""
    from idms.estimator.models.tcanet import PyTorchTrajEstimator

    ts = np.linspace(0.05, 0.30, 10)
    params = torch.randn(6, 3, 1)
    mine, _ = JacobianTrajEstimator(eig=-3.0, time_samples=ts)(params)
    c1 = PyTorchTrajEstimator(eig=-3.0, time_samples=ts)(params)
    assert torch.allclose(mine, c1, atol=1e-5)


def test_propagate_uncertainty_sensible():
    layer = _layer()
    params = torch.randn(5, 3)
    _, jac = layer(params)
    log_var = torch.full((5, 3), -4.0)  # small parameter variances
    traj_log_var = layer.propagate_uncertainty(log_var, jac)
    traj_var = torch.exp(traj_log_var)
    assert traj_log_var.shape == (5, 10)
    assert torch.isfinite(traj_var).all()
    assert (traj_var > 0).all()


def test_zero_param_variance_gives_zero_traj_variance():
    """No parameter uncertainty -> (near) zero trajectory variance."""
    layer = _layer()
    params = torch.randn(3, 3)
    _, jac = layer(params)
    log_var = torch.full((3, 3), -30.0)  # ~0 variance
    traj_var = torch.exp(layer.propagate_uncertainty(log_var, jac))
    assert (traj_var < 1e-6).all()
