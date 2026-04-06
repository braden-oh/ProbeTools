"""Data I/O for Langmuir probe traces.

Handles loading and normalizing I-V characteristic data from various file formats.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


def load_trace(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a Langmuir probe I-V trace from file.

    Supports tab-delimited .txt, CSV, and attempts .hdf5 with key 'Ie'.
    Automatically sorts by bias voltage and ensures correct sign convention
    (current should increase monotonically with voltage for electron saturation).

    Parameters
    ----------
    filepath : str
        Path to trace file

    Returns
    -------
    bias_voltage : ndarray
        Bias voltages [V], sorted in ascending order
    probe_current : ndarray
        Probe currents [A], corresponding to bias_voltage

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If expected columns are missing or file format not recognized
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path.suffix.lower() == ".hdf5":
        return _load_hdf5(filepath)
    elif path.suffix.lower() in [".txt", ".csv"]:
        return _load_text(filepath)
    else:
        # Try as text first
        try:
            return _load_text(filepath)
        except Exception as e:
            raise ValueError(f"Could not parse file {filepath}. Supported: .txt, .csv, .hdf5") from e


def _load_text(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load trace from tab-delimited or comma-separated .txt or .csv file.

    Expects columns: 'Bias Voltage (V)' and 'Probe Current (A)'
    (or 'Voltage' and 'Current', case-insensitive)

    Tries tab separator first, then falls back to comma separator.
    """
    # Try tab separator first (most common for LP data)
    try:
        df = pd.read_csv(filepath, sep="\t", skipinitialspace=True)
    except Exception:
        # Fall back to comma separator
        try:
            df = pd.read_csv(filepath, sep=",", skipinitialspace=True)
        except Exception as e:
            raise ValueError(f"Could not parse file {filepath} with tab or comma separators") from e

    # Normalize column names (case-insensitive search)
    cols_lower = {col.lower().strip(): col for col in df.columns}

    voltage_col = None
    current_col = None

    # Search for voltage column
    for key in ["bias voltage (v)", "bias voltage", "voltage (v)", "voltage"]:
        if key in cols_lower:
            voltage_col = cols_lower[key]
            break

    # Search for current column
    for key in ["probe current (a)", "probe current", "current (a)", "current"]:
        if key in cols_lower:
            current_col = cols_lower[key]
            break

    if voltage_col is None or current_col is None:
        raise ValueError(
            f"Could not find voltage and current columns in {filepath}. "
            f"Found columns: {list(df.columns)}"
        )

    V = df[voltage_col].to_numpy(dtype=float)
    I = df[current_col].to_numpy(dtype=float)

    return _normalize_trace(V, I)


def _load_hdf5(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load trace from HDF5 file with key 'Ie' (electron current array)."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 support. Install with: pip install h5py")

    with h5py.File(filepath, "r") as f:
        if "Ie" not in f:
            raise ValueError(f"HDF5 file must contain 'Ie' dataset. Found: {list(f.keys())}")
        Ie = f["Ie"][:]

    # For HDF5, we assume it's electron current. Reconstruct V from index or use default sweep
    # This is a simplified implementation; adapt as needed for your data
    V = np.linspace(-50, 50, len(Ie))  # Default assumption
    return _normalize_trace(V, Ie)


def _normalize_trace(
    bias_voltage: np.ndarray, probe_current: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize voltage and current arrays.

    - Convert to float
    - Remove NaN/inf values
    - Sort by ascending voltage
    - Ensure current sign convention (positive in electron saturation region)

    Parameters
    ----------
    bias_voltage : array-like
        Raw bias voltages
    probe_current : array-like
        Raw probe currents

    Returns
    -------
    V : ndarray
        Sorted, finite bias voltages [V]
    I : ndarray
        Corresponding probe currents [A], with corrected sign convention
    """
    V = np.asarray(bias_voltage, dtype=float)
    I = np.asarray(probe_current, dtype=float)

    # Remove non-finite values
    mask = np.isfinite(V) & np.isfinite(I)
    V = V[mask]
    I = I[mask]

    # Sort by voltage
    order = np.argsort(V)
    V = V[order]
    I = I[order]

    # Correct sign convention: current should increase from negative (ion sat) to positive (electron sat)
    # Check correlation: if I decreases with V, flip sign
    if len(V) > 1:
        dV = V[-1] - V[0]
        dI = I[-1] - I[0]
        if dV != 0.0 and (dI / dV) < 0.0:
            I = -I

    return V, I
