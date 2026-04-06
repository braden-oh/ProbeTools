from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .schema import MANDATORY_MANIFEST_COLUMNS, OPTIONAL_MANIFEST_COLUMNS, TraceData


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    path = Path(manifest_path)
    manifest = pd.read_csv(path)

    missing = [column for column in MANDATORY_MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    for column in OPTIONAL_MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = np.nan

    manifest["trace_path"] = manifest["trace_path"].astype(str)
    manifest.attrs["manifest_path"] = str(path.resolve())
    return manifest


def load_trace(manifest_row: pd.Series | dict[str, Any]) -> TraceData:
    row = manifest_row if isinstance(manifest_row, dict) else manifest_row.to_dict()
    manifest_dir = Path(row.get("manifest_path", ".")).resolve().parent if row.get("manifest_path") else None
    trace_path = Path(str(row["trace_path"]))
    if not trace_path.is_absolute() and manifest_dir is not None:
        trace_path = manifest_dir / trace_path
    trace_path = trace_path.resolve()

    file_format = str(row["file_format"]).strip().lower()
    bias_voltage, probe_current = _read_trace_data(
        trace_path=trace_path,
        file_format=file_format,
        voltage_selector=str(row["voltage_selector"]),
        current_selector=str(row["current_selector"]),
        delimiter=row.get("delimiter"),
        skiprows=row.get("skiprows"),
    )

    order = np.argsort(bias_voltage)
    bias_voltage = np.asarray(bias_voltage, dtype=float)[order]
    probe_current = np.asarray(probe_current, dtype=float)[order]

    current_std = row.get("current_std_a")
    if pd.isna(current_std):
        current_std_value: float | np.ndarray | None = None
    else:
        current_std_value = float(current_std)

    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"trace_id", "trace_path", "file_format", "voltage_selector", "current_selector"}
    }
    metadata["trace_path"] = str(trace_path)

    return TraceData(
        trace_id=str(row["trace_id"]),
        trace_path=trace_path,
        bias_voltage=bias_voltage,
        probe_current=probe_current,
        gas="" if pd.isna(row["gas"]) else str(row["gas"]),
        probe_geometry=str(row["probe_geometry"]),
        probe_radius_m=float(row["probe_radius_m"]),
        probe_area_m2=float(row["probe_area_m2"]),
        flow_sccm=None if pd.isna(row["flow_sccm"]) else float(row["flow_sccm"]),
        discharge_current_a=None if pd.isna(row["discharge_current_a"]) else float(row["discharge_current_a"]),
        current_std_a=current_std_value,
        metadata=metadata,
    )


def _read_trace_data(
    trace_path: Path,
    file_format: str,
    voltage_selector: str,
    current_selector: str,
    delimiter: object | None,
    skiprows: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    if file_format == "delimited":
        read_kwargs: dict[str, Any] = {}
        if delimiter is not None and not pd.isna(delimiter):
            read_kwargs["sep"] = str(delimiter)
        else:
            read_kwargs["sep"] = _guess_delimiter(trace_path)
        if skiprows is not None and not pd.isna(skiprows):
            read_kwargs["skiprows"] = int(skiprows)
        frame = pd.read_csv(trace_path, **read_kwargs)
        voltage = _select_from_frame(frame, voltage_selector)
        current = _select_from_frame(frame, current_selector)
        return voltage, current

    if file_format == "hdf5_matrix":
        with h5py.File(trace_path, "r") as handle:
            voltage = _select_from_hdf5(handle, voltage_selector)
            current = _select_from_hdf5(handle, current_selector)
        return voltage, current

    raise ValueError(f"Unsupported file_format {file_format!r} for trace {trace_path}")


def _guess_delimiter(trace_path: Path) -> str:
    with trace_path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()
    if "\t" in header:
        return "\t"
    if ";" in header:
        return ";"
    return ","


def _select_from_frame(frame: pd.DataFrame, selector: str) -> np.ndarray:
    selector_type, selector_value = selector.split(":", 1)
    if selector_type != "column":
        raise ValueError(f"CSV selector must be column-based, got {selector!r}")
    return frame[selector_value].to_numpy(dtype=float)


def _select_from_hdf5(handle: h5py.File, selector: str) -> np.ndarray:
    parts = selector.split(":")
    if len(parts) != 3 or parts[0] != "dataset":
        raise ValueError(f"HDF5 selector must have the form dataset:<name>:<index>, got {selector!r}")
    dataset_name = parts[1]
    column_index = int(parts[2])
    data = np.asarray(handle[dataset_name])
    if data.ndim != 2:
        raise ValueError(f"HDF5 dataset {dataset_name!r} must be 2-D, got shape {data.shape}")
    return data[:, column_index].astype(float)
