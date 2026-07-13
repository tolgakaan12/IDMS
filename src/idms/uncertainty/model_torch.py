"""PyTorch aleatoric + epistemic uncertainty model (Contribution 3 port).

Reuses the C1 TCANet backbone (idms.estimator.models.tcanet.TCANet_IDMS) and adds a
second head so the network predicts, for the 3 trajectory parameters [vd, c1, a0],
both a **mean** and a **log-variance**. The mean parameters go through the analytical
`JacobianTrajEstimator` (idms.uncertainty.jacobian_torch) to give the mean trajectory
and Jacobians; the parameter log-variances are propagated to trajectory-space via
sigma^2_traj = sum_i J_i^2 * sigma^2_i (aleatoric). MC-dropout at inference gives the
epistemic term.

This replaces the TensorFlow `elbow_uncertainty_model.py`, and fixes its bug: it uses
the correct [vd, c1, a0] Jacobian rather than the old [theta_inf, A, B] one.
"""
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from idms.estimator.models.tcanet import TCANet_IDMS
from idms.uncertainty.jacobian_torch import JacobianTrajEstimator


class AleatoricUncertaintyModel(nn.Module):
    """EMG -> TCANet encoder -> dual heads (mean, log-var) -> trajectory + uncertainty.

    Forward output: (batch, n_times, 2) with [..., 0] = mean trajectory,
    [..., 1] = trajectory log-variance (aleatoric). Pairs with
    idms.uncertainty.losses_torch.aleatoric_trajectory_loss.
    """

    def __init__(self, backbone: Optional[TCANet_IDMS] = None, eig: float = -3.0,
                 time_samples: Optional[Sequence[float]] = None, logvar_init: float = -3.0,
                 **backbone_kwargs):
        super().__init__()
        if backbone is None:
            backbone = TCANet_IDMS(**backbone_kwargs)

        # Reuse the encoder + the shared FC stack from the C1 backbone.
        self.ms_emg_net = backbone.ms_emg_net
        self.tcn_block = backbone.tcn_block
        self.transformer = backbone.transformer
        self.drop = backbone.drop
        self.shared = backbone.trajectory_params[:-1]   # everything up to the final Linear(->3)
        self.mean_head = backbone.trajectory_params[-1]  # reuse final Linear as the mean head

        # New log-variance head, initialised small (thesis: log-var head init -3 to avoid overconfidence).
        in_features = self.mean_head.in_features
        self.log_var_head = nn.Linear(in_features, 3)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, logvar_init)

        if time_samples is None:
            time_samples = backbone.trajectory_estimator.time_samples
        self.traj = JacobianTrajEstimator(eig=eig, time_samples=time_samples)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ms_emg_net(x)
        x = self.tcn_block(x)
        sa = self.transformer(x)
        x = self.drop(sa + x)
        return self.shared(x)  # (batch, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features(x)
        mean_params = self.mean_head(feat)        # (batch, 3)
        log_var_params = self.log_var_head(feat)  # (batch, 3)
        mean_traj, jac = self.traj(mean_params)   # (batch, n_times), (batch, n_times, 3)
        traj_log_var = self.traj.propagate_uncertainty(log_var_params, jac)  # (batch, n_times)
        return torch.stack([mean_traj, traj_log_var], dim=-1)  # (batch, n_times, 2)

    @torch.no_grad()
    def mc_dropout_predict(self, x: torch.Tensor, n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """MC-dropout inference: keep dropout active, sample the network n_samples times.

        Returns pred mean and the epistemic / aleatoric / total predictive variances,
        each shaped (batch, n_times).
        """
        was_training = self.training
        self.train()  # dropout ON
        means, aleatoric_vars = [], []
        for _ in range(n_samples):
            out = self.forward(x)
            means.append(out[..., 0])
            aleatoric_vars.append(torch.exp(out[..., 1]))
        if not was_training:
            self.eval()
        means = torch.stack(means)              # (S, batch, n_times)
        aleatoric_vars = torch.stack(aleatoric_vars)
        epistemic_var = means.var(dim=0, unbiased=True)   # variance across dropout samples
        aleatoric_var = aleatoric_vars.mean(dim=0)
        return {
            "mean": means.mean(dim=0),
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
            "total_var": epistemic_var + aleatoric_var,
        }
