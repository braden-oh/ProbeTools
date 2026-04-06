from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orsini_lp.forward import estimate_floating_potential, synthetic_trace, synthetic_trace_for_model
from orsini_lp.inference import BayesianConfig, compare_physical_models, fit_bayesian


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


def test_compare_physical_models_returns_evidence_and_bayes_factors_for_oml_trace():
    krypton_mass_kg = 83.798 * 1.66053906660e-27
    bias = np.linspace(-10.0, 10.0, 52)
    trace = synthetic_trace_for_model(
        model_name="oml",
        trace_id="synthetic_oml",
        bias_voltage=bias,
        plasma_potential_v=6.0,
        electron_density_m3=3.5e17,
        electron_temperature_ev=2.8,
        p_value=1.4,
        noise_std_a=6.0e-6,
        probe_radius_m=2.0e-5,
        probe_length_m=2.5e-3,
        ion_mass_kg=krypton_mass_kg,
        seed=11,
    )
    priors = {
        "Vp": (4.0, 8.0),
        "Vf": (-1.0, 5.0),
        "ne": (5.0e16, 2.0e18),
        "ni": (5.0e16, 2.0e18),
        "Te": (1.0, 5.0),
        "p": (1.0, 2.5),
        "I1": (-5.0e-5, 5.0e-5),
        "I2": (-2.0e-4, 0.0),
    }
    result = compare_physical_models(
        trace.bias_voltage,
        trace.probe_current,
        probe_radius_m=trace.probe_radius_m,
        probe_length_m=2.5e-3,
        ion_mass_kg=krypton_mass_kg,
        prior_bounds=priors,
        current_std_a=trace.current_std_a,
        trace_id=trace.trace_id,
        config=BayesianConfig(
            nlive=45,
            dlogz=2.5,
            posterior_draws=40,
            max_points=None,
            random_seed=9,
        ),
    )
    assert set(result.model_results) == {"linear", "oml", "transitional"}
    assert {"oml_vs_linear", "transitional_vs_linear"} == set(result.bayes_factor_table.index)
    assert np.isfinite(result.log_evidence_table.loc["linear", "log_evidence"])
    assert np.isfinite(result.log_evidence_table.loc["oml", "log_evidence"])
    assert np.isfinite(result.log_evidence_table.loc["transitional", "log_evidence"])
    assert result.winning_model in {"oml", "transitional", "linear"}
    assert result.figures["iv_overlay"] is not None
    assert result.figures["eedf_overlay"] is not None
    winning_summary = result.model_results[result.winning_model].summary
    assert np.isclose(
        winning_summary.loc["Vf", "median"],
        estimate_floating_potential(trace.bias_voltage, trace.probe_current),
    )
    assert np.isfinite(winning_summary.loc["Ie_sat", "median"])
    assert result.model_results[result.winning_model].current_quantiles["probe_current_a_valid_fraction"].iloc[-1] == 0.0
    for parameter in ("Vf", "Vp", "ne", "Te", "p"):
        assert np.isfinite(winning_summary.loc[parameter, "median"])


def test_scitech_trace_vp_stays_near_observed_knee_instead_of_railing():
    trace_path = Path("SciTech Sample Data/Kr_15sccm_10A1.txt")
    frame = pd.read_csv(trace_path, sep="\t")
    bias = frame["Bias Voltage (V)"].to_numpy(dtype=float)
    current = frame["Probe Current (A)"].to_numpy(dtype=float)
    vf = estimate_floating_potential(bias, current)
    current_span = max(float(np.ptp(current)), 1.0e-9)
    voltage_span = max(float(np.ptp(bias)), 1.0)

    analysis = compare_physical_models(
        bias,
        current,
        probe_radius_m=25.4e-6,
        probe_length_m=2.54e-3,
        ion_mass_kg=83.798 * 1.66053906660e-27,
        prior_bounds={
            "Vp": (vf + 0.5, vf + 10.0),
            "Vf": (vf - 1.0, vf + 1.0),
            "ne": (1.0e15, 1.0e20),
            "ni": (1.0e15, 1.0e20),
            "Te": (0.25, 4.0),
            "p": (1.0, 2.0),
            "I1": (-5.0 * current_span / voltage_span, 5.0 * current_span / voltage_span),
            "I2": (-5.0 * current_span, 5.0 * current_span),
        },
        trace_id=trace_path.stem,
        gas="Kr",
        config=BayesianConfig(
            nlive=10,
            dlogz=4.0,
            posterior_draws=10,
            max_points=40,
            random_seed=1,
        ),
        make_plots=False,
    )

    result = analysis.model_results[analysis.winning_model]
    assert result.summary.loc["Vp", "median"] < vf + 3.0
    assert result.summary.loc["Vp", "median"] <= result.diagnostics["fit_window_upper_v"] + 0.25
    assert result.current_quantiles["probe_current_a_valid_fraction"].iloc[-1] == 0.0
