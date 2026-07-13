"""Tests for idms.common.metrics (the shared regression metrics)."""
import numpy as np
from sklearn.metrics import r2_score
from idms.common.metrics import regression_metrics, r2, per_point_metrics


def test_matches_sklearn_flattened():
    rng = np.random.RandomState(0)
    y = rng.randn(200, 10)
    p = y + 0.1 * rng.randn(200, 10)
    m = regression_metrics(y, p)
    assert abs(m["r2"] - r2_score(y.reshape(-1), p.reshape(-1))) < 1e-12
    assert m["rmse"] > 0 and m["mae"] > 0


def test_perfect_prediction():
    y = np.random.RandomState(1).randn(50, 10)
    m = regression_metrics(y, y)
    assert abs(m["r2"] - 1.0) < 1e-9
    assert m["rmse"] < 1e-9


def test_r2_convenience_matches():
    y = np.random.RandomState(2).randn(80, 10)
    p = y + 0.2
    assert abs(r2(y, p) - regression_metrics(y, p)["r2"]) < 1e-12


def test_per_point_shape():
    y = np.random.RandomState(3).randn(100, 10)
    p = y + 0.1
    pp = per_point_metrics(y, p)
    assert pp["r2_per_point"].shape == (10,)
    assert pp["rmse_per_point"].shape == (10,)
