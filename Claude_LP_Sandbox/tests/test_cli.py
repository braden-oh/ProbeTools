from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "sample_data" / "local_manifest.csv"


def test_compare_cli_writes_expected_artifacts(tmp_path: Path):
    output_dir = tmp_path / "compare"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orsini_lp.cli",
            "compare",
            str(MANIFEST_PATH),
            "--trace-id",
            "kr_10sccm_10a",
            "--output-dir",
            str(output_dir),
            "--nlive",
            "30",
            "--dlogz",
            "2.5",
            "--max-points",
            "48",
            "--posterior-draws",
            "40",
            "--random-seed",
            "13",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Vp" in result.stdout
    expected = [
        "posterior_samples.csv",
        "bayesian_summary.csv",
        "comparison_summary.csv",
        "iv_fit.png",
        "eedf_fit.png",
        "legacy_diagnostic_trace.csv",
        "legacy_diagnostics.png",
        "metadata.json",
    ]
    for filename in expected:
        assert (output_dir / filename).exists()


def test_batch_cli_writes_batch_summary(tmp_path: Path):
    output_dir = tmp_path / "batch"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "orsini_lp.cli",
            "batch",
            str(MANIFEST_PATH),
            "--trace-id",
            "kr_10sccm_10a",
            "--trace-id",
            "kr_10sccm_10a2",
            "--output-dir",
            str(output_dir),
            "--nlive",
            "25",
            "--dlogz",
            "3.0",
            "--max-points",
            "40",
            "--posterior-draws",
            "30",
            "--random-seed",
            "5",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    batch_summary = output_dir / "batch_summary.csv"
    assert batch_summary.exists()
    text = batch_summary.read_text()
    assert "kr_10sccm_10a" in text
    assert "kr_10sccm_10a2" in text
