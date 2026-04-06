from __future__ import annotations

from pathlib import Path

import pandas as pd

from .adapters import infer_local_manifest_rows
from .compare import compare_trace, write_trace_artifacts
from .inference import BayesianConfig, fit_bayesian
from .io import load_trace
from .legacy import run_legacy


def build_directory_manifest(
    trace_dir: str | Path,
    *,
    probe_geometry: str | None = None,
    probe_radius_m: float | None = None,
    probe_area_m2: float | None = None,
    gas: str | None = None,
    notes: str | None = None,
    use_absolute_paths: bool = True,
) -> pd.DataFrame:
    trace_root = Path(trace_dir).resolve()
    manifest = infer_local_manifest_rows(
        trace_root,
        probe_geometry=probe_geometry,
        probe_radius_m=probe_radius_m,
        probe_area_m2=probe_area_m2,
        gas=gas,
        notes=notes,
    )
    if manifest.empty:
        raise ValueError(f"No supported Langmuir trace files were found in {trace_root}.")

    manifest = manifest.copy()
    if use_absolute_paths:
        manifest["trace_path"] = manifest["trace_path"].map(lambda value: str((trace_root / str(value)).resolve()))
    manifest.attrs["trace_root"] = str(trace_root)
    return manifest


def process_trace_directory(
    trace_dir: str | Path,
    *,
    probe_geometry: str | None = None,
    probe_radius_m: float | None = None,
    probe_area_m2: float | None = None,
    gas: str | None = None,
    notes: str | None = None,
    config: BayesianConfig | None = None,
    results_dir_name: str = "results",
    trace_ids: list[str] | None = None,
    run_legacy_processor: bool = True,
    write_manifest_copy: bool = True,
) -> pd.DataFrame:
    trace_root = Path(trace_dir).resolve()
    results_dir = trace_root / results_dir_name
    manifest = build_directory_manifest(
        trace_root,
        probe_geometry=probe_geometry,
        probe_radius_m=probe_radius_m,
        probe_area_m2=probe_area_m2,
        gas=gas,
        notes=notes,
        use_absolute_paths=True,
    )

    if trace_ids is not None:
        manifest = manifest[manifest["trace_id"].isin(trace_ids)].copy()
    if manifest.empty:
        raise ValueError(f"No traces remain to process in {trace_root} after applying the requested filters.")

    results_dir.mkdir(parents=True, exist_ok=True)
    if write_manifest_copy:
        manifest.to_csv(results_dir / "auto_manifest.csv", index=False)

    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        trace = load_trace(record)
        bayes = fit_bayesian(trace, config=config)
        legacy = run_legacy(trace) if run_legacy_processor else None
        comparison = compare_trace(trace, bayes, legacy)
        write_trace_artifacts(results_dir / trace.trace_id, trace, bayes, legacy, comparison)
        rows.append(_flatten_comparison_row(comparison))

    summary = pd.DataFrame.from_records(rows).sort_values("trace_id").reset_index(drop=True)
    summary.to_csv(results_dir / "batch_summary.csv", index=False)
    return summary


def _flatten_comparison_row(comparison) -> dict[str, object]:
    record: dict[str, object] = {
        "trace_id": comparison.trace_id,
        "legacy_success": comparison.metadata.get("legacy_success"),
        "legacy_error": comparison.metadata.get("legacy_error"),
    }
    for metric, row in comparison.comparison_table.iterrows():
        record[f"{metric}_bayes_median"] = row["bayes_median"]
        record[f"{metric}_legacy_value"] = row["legacy_value"]
        record[f"{metric}_delta"] = row["delta"]
    return record
