from __future__ import annotations

import numpy as np

from orsini_lp.legacy import run_legacy
from orsini_lp.schema import TraceData


def test_run_legacy_recovers_reasonable_classical_parameters():
    bias_voltage = np.linspace(-10.0, 20.0, 181)
    vp_true = 12.0
    te_true = 1.8
    ion_current = 1.2e-5 * bias_voltage - 2.4e-4
    electron_retarding = 7.5e-4 * np.exp((bias_voltage - vp_true) / te_true)
    electron_current = np.where(
        bias_voltage <= vp_true,
        electron_retarding,
        7.5e-4 * (1.0 + 0.08 * np.sqrt(np.clip(bias_voltage - vp_true, 0.0, None))),
    )
    probe_current = ion_current + electron_current
    probe_current += np.random.default_rng(9).normal(scale=2.0e-6, size=probe_current.size)

    trace = TraceData(
        trace_id="legacy_synthetic",
        trace_path=None,  # type: ignore[arg-type]
        bias_voltage=bias_voltage,
        probe_current=probe_current,
        gas="Kr",
        probe_geometry="cylindrical",
        probe_radius_m=2.54e-5,
        probe_area_m2=4.0739276835807376e-7,
        flow_sccm=10.0,
        discharge_current_a=10.0,
        current_std_a=None,
        metadata={},
    )

    result = run_legacy(trace)

    assert result.success
    assert result.error is None
    assert result.diagnostic_trace is not None
    assert abs(result.summary.loc["Vp", "median"] - vp_true) < 1.0
    assert abs(result.summary.loc["Te", "median"] - te_true) < 0.5
    assert "legacy_total_model_a" in result.diagnostic_trace.columns
    assert result.metadata["vp_method"] in {"derivative-peak", "global-derivative-maximum"}
