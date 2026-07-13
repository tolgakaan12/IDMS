"""Tests for the ARMA-GARCH-t residual model (Contribution 2)."""
import numpy as np
from idms.residuals.arma_garch import EnhancedARMAGARCH


def _synthetic_ar_garch(n=1500, seed=0):
    """AR(1) mean + GARCH(1,1) volatility with Student-t innovations."""
    rng = np.random.RandomState(seed)
    e = rng.standard_t(6, n)
    h = np.ones(n)
    r = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.02 + 0.10 * r[t - 1] ** 2 + 0.85 * h[t - 1]
        r[t] = 0.3 * r[t - 1] + np.sqrt(h[t]) * e[t]
    return r


def test_fit_then_simulate():
    r = _synthetic_ar_garch()
    model = EnhancedARMAGARCH(p=1, q=1)
    model.fit(r)
    sim = model.simulate(500)
    assert len(sim) == 500
    assert np.all(np.isfinite(sim))
