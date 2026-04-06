from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inference import BayesianConfig, fit_bayesian
from .io import load_manifest, load_trace
from .legacy import run_legacy
from .plotting import plot_eedf_fit, plot_iv_fit, plot_legacy_diagnostics
from .schema import BayesianResult, ComparisonResult, LegacyResult, TraceData


def compare_trace(
    trace: TraceData,
    bayes: BayesianResult,
    legacy: LegacyResult | None,
) -> ComparisonResult:
    trace_summary = pd.DataFrame(
        {
            "trace_id": [trace.trace_id],
            "trace_path": [str(trace.trace_path)],
            "gas": [trace.gas],
            "probe_geometry": [trace.probe_geometry],
            "probe_radius_m": [trace.probe_radius_m],
            "probe_area_m2": [trace.probe_area_m2],
            "flow_sccm": [trace.flow_sccm],
            "discharge_current_a": [trace.discharge_current_a],
            "point_count": [trace.point_count],
        }
    )

    legacy_summary = legacy.summary if legacy is not None else pd.DataFrame()
    comparison_rows = []

    comparisons = [
        ("Vp", "Vp"),
        ("Te", "Te"),
        ("ne", "ne"),
        ("n_qn", "n"),
    ]
    for bayes_name, legacy_name in comparisons:
        row = {
            "metric": bayes_name,
            "bayes_median": float(bayes.summary.loc[bayes_name, "median"]),
            "bayes_q16": float(bayes.summary.loc[bayes_name, "q16"]),
            "bayes_q84": float(bayes.summary.loc[bayes_name, "q84"]),
            "legacy_value": np.nan,
            "legacy_std": np.nan,
            "delta": np.nan,
        }
        if legacy is not None and legacy.success and legacy_name in legacy_summary.index:
            row["legacy_value"] = float(legacy_summary.loc[legacy_name, "median"])
            row["legacy_std"] = float(legacy_summary.loc[legacy_name, "std"])
            row["delta"] = row["bayes_median"] - row["legacy_value"]
        comparison_rows.append(row)

    comparison_rows.append(
        {
            "metric": "p",
            "bayes_median": float(bayes.summary.loc["p", "median"]),
            "bayes_q16": float(bayes.summary.loc["p", "q16"]),
            "bayes_q84": float(bayes.summary.loc["p", "q84"]),
            "legacy_value": np.nan,
            "legacy_std": np.nan,
            "delta": np.nan,
        }
    )

    return ComparisonResult(
        trace_id=trace.trace_id,
        trace_summary=trace_summary,
        bayesian_summary=bayes.summary,
        legacy_summary=legacy_summary,
        comparison_table=pd.DataFrame.from_records(comparison_rows).set_index("metric"),
        metadata={
            "legacy_success": False if legacy is None else legacy.success,
            "legacy_error": None if legacy is None else legacy.error,
        },
    )


def batch_compare(
    manifest_path: str | Path,
    trace_ids: list[str] | None = None,
    config: BayesianConfig | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    manifest["manifest_path"] = manifest.attrs["manifest_path"]
    if trace_ids is not None:
        manifest = manifest[manifest["trace_id"].isin(trace_ids)].copy()

    rows = []
    output_root = None if output_dir is None else Path(output_dir)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    for record in manifest.to_dict(orient="records"):
        trace = load_trace(record)
        bayes = fit_bayesian(trace, config=config)
        legacy = run_legacy(trace)
        comparison = compare_trace(trace, bayes, legacy)
        rows.append(_flatten_comparison(comparison))

        if output_root is not None:
            write_trace_artifacts(output_root / trace.trace_id, trace, bayes, legacy, comparison)

    summary = pd.DataFrame.from_records(rows)
    if output_root is not None:
        summary.to_csv(output_root / "batch_summary.csv", index=False)
    return summary


def write_trace_artifacts(
    output_dir: str | Path,
    trace: TraceData,
    bayes: BayesianResult,
    legacy: LegacyResult | None,
    comparison: ComparisonResult,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bayes.posterior_samples.to_csv(output_path / "posterior_samples.csv", index=False)
    bayes.summary.to_csv(output_path / "bayesian_summary.csv")
    bayes.model_current_quantiles.to_csv(output_path / "iv_posterior_predictive.csv", index=False)
    bayes.eedf_density_quantiles.to_csv(output_path / "eedf_posterior_predictive.csv", index=False)
    comparison.comparison_table.to_csv(output_path / "comparison_summary.csv")
    comparison.trace_summary.to_csv(output_path / "trace_summary.csv", index=False)

    if legacy is not None and legacy.derived_quantities is not None:
        legacy.derived_quantities.to_csv(output_path / "legacy_derived_quantities.csv")
        legacy.summary.to_csv(output_path / "legacy_summary.csv")
    if legacy is not None and legacy.diagnostic_trace is not None:
        legacy.diagnostic_trace.to_csv(output_path / "legacy_diagnostic_trace.csv", index=False)

    metadata = {
        "trace_id": trace.trace_id,
        "trace_path": str(trace.trace_path),
        "bayesian_log_evidence": bayes.log_evidence,
        "bayesian_diagnostics": bayes.diagnostics,
        "legacy_success": None if legacy is None else legacy.success,
        "legacy_error": None if legacy is None else legacy.error,
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    iv_fig = plot_iv_fit(trace, bayes, legacy)
    iv_fig.savefig(output_path / "iv_fit.png", dpi=160, bbox_inches="tight")
    eedf_fig = plot_eedf_fit(bayes)
    eedf_fig.savefig(output_path / "eedf_fit.png", dpi=160, bbox_inches="tight")
    if legacy is not None and legacy.success and legacy.diagnostic_trace is not None:
        legacy_fig = plot_legacy_diagnostics(trace, legacy, bayes)
        legacy_fig.savefig(output_path / "legacy_diagnostics.png", dpi=160, bbox_inches="tight")
        plt.close(legacy_fig)
    plt.close(iv_fig)
    plt.close(eedf_fig)


def _flatten_comparison(comparison: ComparisonResult) -> dict[str, object]:
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
