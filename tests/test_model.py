"""Tests for the TCANet intent-estimator model (Contribution 1).

Locks the forward-pass contract and parameter count that the replication-verified
checkpoint (test R2 = 0.7982814312) depends on.
"""
import torch
from idms.estimator.models.tcanet import create_tcanet_idms_model


def test_forward_shape():
    model = create_tcanet_idms_model()
    x = torch.randn(4, 1, 4, 1000)  # (batch, 1, channels, time)
    y = model(x)
    assert y.shape == (4, 10)  # 10 trajectory points


def test_param_count_locked():
    # 557,347 params is the exact architecture that reproduces R2=0.798.
    n = sum(p.numel() for p in create_tcanet_idms_model().parameters())
    assert n == 557347
