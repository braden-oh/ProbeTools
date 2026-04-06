from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .io import _guess_delimiter


def selector_for_column(column_name: str) -> str:
    return f"column:{column_name}"


def selector_for_dataset(dataset_name: str, index: int) -> str:
    return f"dataset:{dataset_name}:{index}"


def parse_discharge_filename(path: Path) -> dict[str, float | str | None]:
    name = path.stem
    result: dict[str, float | str | None] = {
        "gas": None,
        "flow_sccm": np.nan,
        "discharge_current_a": np.nan,
    }

    gas_match = re.match(r"^(Kr|Xe|Ar|Zn)_", name, flags=re.IGNORECASE)
    if gas_match:
        result["gas"] = gas_match.group(1).title()

    flow_match = re.search(r"(\d+(?:[pP\.]\d+)?)sccm", name, flags=re.IGNORECASE)
    if flow_match:
        result["flow_sccm"] = _decode_numeric_token(flow_match.group(1))

    current_match = re.search(r"_(\d+(?:[pP\.]\d+)?)A", name, flags=re.IGNORECASE)
    if current_match:
        result["discharge_current_a"] = _decode_numeric_token(current_match.group(1))

    return result


def cylindrical_probe_metadata(
    exposed_length_m: float,
    diameter_m: float,
) -> dict[str, float | str]:
    if exposed_length_m <= 0.0 or diameter_m <= 0.0:
        raise ValueError("Cylindrical probe exposed_length_m and diameter_m must both be positive.")
    radius_m = diameter_m / 2.0
    area_m2 = float(np.pi * diameter_m * (exposed_length_m + diameter_m / 4.0))
    return {
        "probe_geometry": "cylindrical",
        "probe_radius_m": radius_m,
        "probe_area_m2": area_m2,
    }


def infer_local_manifest_rows(
    root: str | Path,
    *,
    probe_geometry: str | None = None,
    probe_radius_m: float | None = None,
    probe_area_m2: float | None = None,
    gas: str | None = None,
    notes: str | None = None,
) -> pd.DataFrame:
    probe_geometry_value, probe_radius_value, probe_area_value = _validate_explicit_probe_metadata(
        probe_geometry=probe_geometry,
        probe_radius_m=probe_radius_m,
        probe_area_m2=probe_area_m2,
    )
    root_path = Path(root)
    rows: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    row_notes = notes or f"Auto-generated adapter row using caller-supplied {probe_geometry_value} probe metadata."

    for suffix in ("*.csv", "*.txt"):
        for path in sorted(root_path.glob(suffix)):
            if path in seen_paths or not _looks_like_langmuir_trace(path):
                continue
            seen_paths.add(path)
            metadata = parse_discharge_filename(path)
            rows.append(
                {
                    "trace_id": path.stem.lower(),
                    "trace_path": path.name,
                    "file_format": "delimited",
                    "delimiter": _guess_delimiter(path),
                    "voltage_selector": selector_for_column("Bias Voltage (V)"),
                    "current_selector": selector_for_column("Probe Current (A)"),
                    "gas": gas if gas is not None else metadata["gas"] or "",
                    "probe_geometry": probe_geometry_value,
                    "probe_radius_m": probe_radius_value,
                    "probe_area_m2": probe_area_value,
                    "flow_sccm": metadata["flow_sccm"],
                    "discharge_current_a": metadata["discharge_current_a"],
                    "current_std_a": np.nan,
                    "notes": row_notes,
                }
            )

    hdf5_path = root_path / "LP_33-34deg_150V.hdf5"
    if hdf5_path.exists():
        rows.append(
            {
                "trace_id": "lp_33_34deg_150v",
                "trace_path": hdf5_path.name,
                "file_format": "hdf5_matrix",
                "delimiter": np.nan,
                "voltage_selector": selector_for_dataset("Data", 0),
                "current_selector": selector_for_dataset("Data", 1),
                "gas": gas or "",
                "probe_geometry": probe_geometry_value,
                "probe_radius_m": probe_radius_value,
                "probe_area_m2": probe_area_value,
                "flow_sccm": np.nan,
                "discharge_current_a": np.nan,
                "current_std_a": np.nan,
                "notes": row_notes,
            }
        )

    return pd.DataFrame.from_records(rows)


def _decode_numeric_token(token: str) -> float:
    return float(token.replace("p", ".").replace("P", "."))


def _looks_like_langmuir_trace(path: Path) -> bool:
    try:
        frame = pd.read_csv(path, sep=_guess_delimiter(path), nrows=1)
    except Exception:
        return False
    expected_columns = {"Bias Voltage (V)", "Probe Current (A)"}
    return expected_columns.issubset(set(frame.columns))


def _validate_explicit_probe_metadata(
    *,
    probe_geometry: str | None,
    probe_radius_m: float | None,
    probe_area_m2: float | None,
) -> tuple[str, float, float]:
    missing = [
        name
        for name, value in (
            ("probe_geometry", probe_geometry),
            ("probe_radius_m", probe_radius_m),
            ("probe_area_m2", probe_area_m2),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Auto-manifest generation requires explicit probe metadata. "
            f"Missing: {', '.join(missing)}."
        )

    geometry = str(probe_geometry).strip().lower()
    if not geometry:
        raise ValueError("probe_geometry must be a non-empty string.")

    radius_m = float(probe_radius_m)
    area_m2 = float(probe_area_m2)
    if radius_m <= 0.0:
        raise ValueError("probe_radius_m must be positive.")
    if area_m2 <= 0.0:
        raise ValueError("probe_area_m2 must be positive.")
    return geometry, radius_m, area_m2
