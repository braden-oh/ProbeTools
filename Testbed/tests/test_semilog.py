from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from orsini_lp.semilog import fit_semilog_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fit_semilog_trace_recovers_exponential_slope(tmp_path: Path):
    true_te_ev = 2.2
    bias_voltage = np.linspace(-6.0, 12.0, 120)
    probe_current = -2.0e-3 + 5.0e-5 * np.exp((bias_voltage - 1.0) / true_te_ev)
    frame = pd.DataFrame(
        {
            "Bias Voltage (V)": bias_voltage,
            "Probe Current (A)": probe_current,
        }
    )
    trace_path = tmp_path / "synthetic_semilog_trace.csv"
    frame.to_csv(trace_path, index=False)

    result = fit_semilog_trace(trace_path, lower_current_fraction=0.02, upper_current_fraction=0.70)

    assert result.fit_point_count >= 12
    assert result.fit_lower_v < result.fit_upper_v
    assert result.r_squared > 0.98
    assert abs(result.electron_temperature_ev - true_te_ev) / true_te_ev < 0.2


def test_semilog_cli_writes_expected_outputs(tmp_path: Path):
    output_dir = tmp_path / "semilog"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orsini_lp.semilog",
            str(PROJECT_ROOT / "Kr_10sccm_10A.csv"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "kr_10sccm_10a" in result.stdout
    expected = [
        "semilog_summary.csv",
        "kr_10sccm_10a_semilog_fit.png",
        "kr_10sccm_10a_semilog_points.csv",
    ]
    for filename in expected:
        assert (output_dir / filename).exists()
