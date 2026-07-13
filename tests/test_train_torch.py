"""Tests for the PyTorch uncertainty trainer (Contribution 3 port)."""
import torch
from torch.utils.data import TensorDataset, DataLoader

from idms.uncertainty.model_torch import AleatoricUncertaintyModel
from idms.uncertainty.losses_torch import aleatoric_trajectory_loss
from idms.uncertainty.train_torch import fit_uncertainty_model


def test_model_can_learn_overfit():
    """Optimizer steps on a fixed batch reduce the NLL (the model can fit)."""
    import math

    torch.manual_seed(0)
    model = AleatoricUncertaintyModel().eval()  # no dropout -> deterministic overfit
    x = torch.randn(4, 1, 4, 1000)
    y = torch.randn(4, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = aleatoric_trajectory_loss(y, model(x)).item()
    for _ in range(30):
        opt.zero_grad()
        aleatoric_trajectory_loss(y, model(x)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # NLL stability
        opt.step()
    last = aleatoric_trajectory_loss(y, model(x)).item()
    assert math.isfinite(last) and last < first


def test_fit_loop_runs_and_returns_history():
    torch.manual_seed(0)
    x = torch.randn(8, 1, 4, 1000)
    y = torch.randn(8, 10)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = AleatoricUncertaintyModel()
    hist = fit_uncertainty_model(model, loader, loader, epochs=3, patience=5, device=torch.device("cpu"))
    assert len(hist["train_loss"]) >= 1
    assert len(hist["val_loss"]) >= 1
    assert all(isinstance(v, float) for v in hist["train_loss"])
