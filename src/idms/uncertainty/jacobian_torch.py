"""PyTorch port of the analytical [vd, c1, a0] Jacobian trajectory layer (Contribution 3).

Uses the SAME integrated trajectory equation as the C1 estimator's
`PyTorchTrajEstimator` (idms.estimator.models.tcanet):

    c2 = a0 - lambda*c1
    c3 = c2 - lambda*c1                      (= a0 - 2*lambda*c1)
    y(t) = vd*t + [ c3 + (lambda*c1 + lambda*c2*t - c2) * exp(lambda*t) ] / lambda**2

and adds the analytical Jacobians d y / d [vd, c1, a0] (thesis Appendix A.3):

    dy/dvd = t
    dy/dc1 = (-2/lambda) + ((2/lambda) - t) * exp(lambda*t)
    dy/da0 = (1/lambda**2) + ((t/lambda) - (1/lambda**2)) * exp(lambda*t)

so parameter variances propagate to trajectory-space variance via the diagonal
first-order rule  sigma^2_traj(t) = sum_i (dy/dp_i)^2 * sigma^2_i.

This replaces the TensorFlow `jacobian_layers.JacobianTrajEstimator`. The analytical
Jacobian is unit-tested against torch.autograd (see tests/test_jacobian_torch.py).
"""
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Eigenvalue stability clip, matching PyTorchTrajEstimator and the TF layer.
_EIG_MIN, _EIG_MAX = -4.0, -0.5


def _default_time_samples() -> np.ndarray:
    """Thesis default: 10 points over [delay, delay+horizon] = [0.05, 0.30] s."""
    return np.linspace(0.05, 0.30, 10)


class JacobianTrajEstimator(nn.Module):
    """Critically-damped trajectory + analytical Jacobians w.r.t. [vd, c1, a0].

    Args:
        eig: eigenvalue lambda (clipped to [-4, -0.5] for stability).
        time_samples: 1-D sequence of times t (seconds). Defaults to the thesis grid.
    """

    def __init__(self, eig: float = -3.0, time_samples: Optional[Sequence[float]] = None):
        super().__init__()
        if time_samples is None:
            time_samples = _default_time_samples()
        self.eig = float(eig)
        T = torch.as_tensor(np.asarray(time_samples), dtype=torch.float32).view(1, -1)  # (1, n_times)
        self.register_buffer("T", T)

    @property
    def lam(self) -> float:
        return float(np.clip(self.eig, _EIG_MIN, _EIG_MAX))

    @property
    def n_times(self) -> int:
        return self.T.shape[1]

    @staticmethod
    def _as_params(params: torch.Tensor) -> torch.Tensor:
        """Accept (batch, 3) or (batch, 3, 1) -> (batch, 3)."""
        if params.dim() == 3:
            params = params.reshape(params.shape[0], 3)
        return params

    def trajectory(self, params: torch.Tensor) -> torch.Tensor:
        """params (batch, 3) [vd, c1, a0] -> trajectory (batch, n_times)."""
        p = self._as_params(params)
        vd, c1, a0 = p[:, 0:1], p[:, 1:2], p[:, 2:3]  # each (batch, 1)
        lam = self.lam
        lam_sq = lam * lam
        T = self.T  # (1, n_times)
        exp_lt = torch.exp(lam * T)  # (1, n_times)
        c2 = a0 - lam * c1
        c3 = c2 - lam * c1
        numerator = c3 + (lam * c1 + lam * c2 * T - c2) * exp_lt  # (batch, n_times)
        return vd * T + numerator / lam_sq  # (batch, n_times)

    def jacobians(self, batch_size: int) -> torch.Tensor:
        """Analytical Jacobians -> (batch, n_times, 3) for [vd, c1, a0]."""
        lam = self.lam
        lam_sq = lam * lam
        T = self.T  # (1, n_times)
        exp_lt = torch.exp(lam * T)
        dy_dvd = T
        dy_dc1 = (-2.0 / lam) + ((2.0 / lam) - T) * exp_lt
        dy_da0 = (1.0 / lam_sq) + ((T / lam) - (1.0 / lam_sq)) * exp_lt
        jac = torch.stack([dy_dvd, dy_dc1, dy_da0], dim=-1)  # (1, n_times, 3)
        return jac.expand(batch_size, -1, -1)

    def forward(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (trajectory (batch, n_times), jacobians (batch, n_times, 3))."""
        p = self._as_params(params)
        return self.trajectory(p), self.jacobians(p.shape[0])

    @staticmethod
    def propagate_uncertainty(param_log_var: torch.Tensor, jacobians: torch.Tensor) -> torch.Tensor:
        """First-order diagonal propagation.

        Args:
            param_log_var: (batch, 3) log-variances of [vd, c1, a0].
            jacobians:     (batch, n_times, 3).
        Returns:
            traj_log_var:  (batch, n_times), log of sigma^2_traj = sum_i J_i^2 * sigma^2_i.
        """
        param_var = torch.exp(param_log_var)  # (batch, 3)
        traj_var = torch.sum(jacobians**2 * param_var[:, None, :], dim=-1)  # (batch, n_times)
        return torch.log(torch.clamp(traj_var, min=1e-8))
