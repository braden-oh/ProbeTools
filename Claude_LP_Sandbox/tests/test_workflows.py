from __future__ import annotations

import shutil
from pathlib import Path

from orsini_lp.adapters import cylindrical_probe_metadata
from orsini_lp.inference import BayesianConfig
from orsini_lp.workflows import build_directory_manifest, process_trace_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE = cylindrical_probe_metadata(exposed_length_m=2.540e-3, diameter_m=5.08e-5)


def test_build_directory_manifest_uses_absolute_paths(tmp_path: Path):
    shutil.copy(PROJECT_ROOT / "Kr_10sccm_10A.csv", tmp_path / "Kr_10sccm_10A.csv")

    manifest = build_directory_manifest(tmp_path, **PROBE)

    assert manifest.shape[0] == 1
    assert manifest.iloc[0]["trace_id"] == "kr_10sccm_10a"
    assert Path(manifest.iloc[0]["trace_path"]).is_absolute()


def test_process_trace_directory_writes_results_tree(tmp_path: Path):
    shutil.copy(PROJECT_ROOT / "Kr_10sccm_10A.csv", tmp_path / "Kr_10sccm_10A.csv")

    summary = process_trace_directory(
        tmp_path,
        **PROBE,
        config=BayesianConfig(
            nlive=25,
            dlogz=3.0,
            max_points=40,
            posterior_draws=30,
            random_seed=5,
        ),
        run_legacy_processor=False,
    )

    results_dir = tmp_path / "results"
    assert "kr_10sccm_10a" in summary["trace_id"].tolist()
    assert (results_dir / "auto_manifest.csv").exists()
    assert (results_dir / "batch_summary.csv").exists()
    assert (results_dir / "kr_10sccm_10a" / "bayesian_summary.csv").exists()
    assert (results_dir / "kr_10sccm_10a" / "iv_fit.png").exists()
