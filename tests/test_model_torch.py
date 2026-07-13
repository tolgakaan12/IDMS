"""Tests for the PyTorch aleatoric+epistemic uncertainty model (Contribution 3 port)."""
import sys

import torch

from idms.uncertainty.model_torch import AleatoricUncertaintyModel
from idms.uncertainty.losses_torch import aleatoric_trajectory_loss


def test_no_tensorflow_pulled():
    import idms.uncertainty.model_torch  # noqa: F401
    assert "tensorflow" not in sys.modules


def test_forward_shape_and_finite():
    model = AleatoricUncertaintyModel().eval()
    out = model(torch.randn(2, 1, 4, 1000))
    assert out.shape == (2, 10, 2)  # (batch, n_times, [mean, log_var])
    assert torch.isfinite(out).all()


def test_loss_backprops_to_both_heads():
    torch.manual_seed(0)
    model = AleatoricUncertaintyModel()
    out = model(torch.randn(2, 1, 4, 1000))
    aleatoric_trajectory_loss(torch.randn(2, 10), out).backward()
    assert model.mean_head.weight.grad is not None
    assert model.log_var_head.weight.grad is not None
    assert torch.isfinite(model.log_var_head.weight.grad).all()


def test_log_var_head_initialised_small():
    model = AleatoricUncertaintyModel(logvar_init=-3.0)
    assert torch.allclose(model.log_var_head.bias.detach(), torch.full((3,), -3.0))


def test_mc_dropout_gives_epistemic_and_aleatoric():
    torch.manual_seed(0)
    model = AleatoricUncertaintyModel()
    res = model.mc_dropout_predict(torch.randn(2, 1, 4, 1000), n_samples=20)
    for key in ("mean", "epistemic_var", "aleatoric_var", "total_var"):
        assert res[key].shape == (2, 10)
        assert torch.isfinite(res[key]).all()
    assert (res["epistemic_var"] > 0).any()      # dropout produces variance across samples
    assert (res["aleatoric_var"] > 0).all()
    assert torch.allclose(res["total_var"], res["epistemic_var"] + res["aleatoric_var"])
