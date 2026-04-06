from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MANDATORY_MANIFEST_COLUMNS = [
    "trace_id",
    "trace_path",
    "file_format",
    "voltage_selector",
    "current_selector",
    "gas",
    "probe_geometry",
    "probe_radius_m",
    "probe_area_m2",
    "flow_sccm",
    "discharge_current_a",
]

OPTIONAL_MANIFEST_COLUMNS = [
    "current_std_a",
    "delimiter",
    "skiprows",
    "notes",
]

CENTRAL_68 = (15.865, 84.135)
CENTRAL_95_5 = (2.25, 97.75)


@dataclass(slots=True)
class TraceData:
    trace_id: str
    trace_path: Path
    bias_voltage: np.ndarray
    probe_current: np.ndarray
    gas: str
    probe_geometry: str
    probe_radius_m: float
    probe_area_m2: float
    flow_sccm: float | None
    discharge_current_a: float | None
    current_std_a: float | np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return int(self.bias_voltage.size)


@dataclass(slots=True)
class BayesianResult:
    trace_id: str
    posterior_samples: pd.DataFrame
    summary: pd.DataFrame
    log_evidence: float
    model_current_samples: np.ndarray
    model_current_quantiles: pd.DataFrame
    energy_grid_ev: np.ndarray
    eedf_density_samples: np.ndarray
    eedf_density_quantiles: pd.DataFrame
    config: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LegacyResult:
    trace_id: str
    summary: pd.DataFrame
    derived_quantities: pd.DataFrame | None
    diagnostic_trace: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass(slots=True)
class ComparisonResult:
    trace_id: str
    trace_summary: pd.DataFrame
    bayesian_summary: pd.DataFrame
    legacy_summary: pd.DataFrame
    comparison_table: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def summarize_samples(
    samples: np.ndarray,
    parameter_names: list[str],
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for index, name in enumerate(parameter_names):
        values = np.asarray(samples[:, index], dtype=float)
        q16, q84 = np.percentile(values, CENTRAL_68)
        q2, q97 = np.percentile(values, CENTRAL_95_5)
        records.append(
            {
                "parameter": name,
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "q16": float(q16),
                "q84": float(q84),
                "q2.25": float(q2),
                "q97.75": float(q97),
            }
        )
    return pd.DataFrame.from_records(records).set_index("parameter")


def summarize_vector_quantiles(
    vector_samples: np.ndarray,
    coordinate: np.ndarray,
    coordinate_name: str,
    value_name: str,
) -> pd.DataFrame:
    median = np.full(vector_samples.shape[1], np.nan, dtype=float)
    q16 = np.full(vector_samples.shape[1], np.nan, dtype=float)
    q84 = np.full(vector_samples.shape[1], np.nan, dtype=float)
    q2 = np.full(vector_samples.shape[1], np.nan, dtype=float)
    q97 = np.full(vector_samples.shape[1], np.nan, dtype=float)
    valid_fraction = np.mean(np.isfinite(vector_samples), axis=0)

    for index in range(vector_samples.shape[1]):
        column = vector_samples[:, index]
        finite = np.isfinite(column)
        if not np.any(finite):
            continue
        finite_values = column[finite]
        median[index] = float(np.median(finite_values))
        q16[index], q84[index] = np.percentile(finite_values, CENTRAL_68)
        q2[index], q97[index] = np.percentile(finite_values, CENTRAL_95_5)

    return pd.DataFrame(
        {
            coordinate_name: coordinate,
            f"{value_name}_median": median,
            f"{value_name}_q16": q16,
            f"{value_name}_q84": q84,
            f"{value_name}_q2.25": q2,
            f"{value_name}_q97.75": q97,
            f"{value_name}_valid_fraction": valid_fraction,
        }
    )


def single_value_summary(name: str, value: float, uncertainty: float | None = None) -> pd.DataFrame:
    record = {
        "parameter": name,
        "median": float(value),
        "mean": float(value),
        "std": float(uncertainty or 0.0),
        "q16": float(value - (uncertainty or 0.0)),
        "q84": float(value + (uncertainty or 0.0)),
        "q2.25": float(value - 2.0 * (uncertainty or 0.0)),
        "q97.75": float(value + 2.0 * (uncertainty or 0.0)),
    }
    return pd.DataFrame.from_records([record]).set_index("parameter")
