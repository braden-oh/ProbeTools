from __future__ import annotations

import numpy as np
import pytest

from orsini_lp.forward import synthetic_trace
from orsini_lp.inference import BayesianConfig, fit_bayesian


@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0])
def test_fit_bayesian_returns_finite_posteriors_for_synthetic_trace(p_true: float):
    bias = np.linspace(-4.0, 9.0, 44)
    trace = synthetic_trace(
        trace_id=f"synthetic_{p_true}",
        bias_voltage=bias,
        plasma_potential_v=5.0,
        electron_density_m3=4.0e17,
        electron_temperature_ev=2.5,
        p_value=p_true,
        ion_slope_a_per_v=-1.2e-4,
        ion_intercept_a=-3.0e-4,
        noise_std_a=2.0e-5,
        probe_area_m2=1.5e-6,
        probe_radius_m=1.5e-4,
        seed=123,
    )
    result = fit_bayesian(
        trace,
        BayesianConfig(
            nlive=35,
            dlogz=2.5,
            posterior_draws=40,
            max_points=None,
            random_seed=17,
        ),
    )
    summary = result.summary
    truths = {
        "Vp": 5.0,
        "Te": 2.5,
        "ne": 4.0e17,
        "p": p_true,
    }
    for parameter, truth in truths.items():
        assert np.isfinite(summary.loc[parameter, "median"])
    assert result.diagnostics["fit_window_point_count"] >= 6
    assert summary.loc["Vp", "median"] > result.diagnostics["floating_potential_v"]
    assert summary.loc["Te", "median"] > 0.0
    assert summary.loc["ne", "median"] > 0.0
    assert 1.0 <= summary.loc["p", "median"] <= 3.0
