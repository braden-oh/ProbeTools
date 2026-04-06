from __future__ import annotations

import os
from pathlib import Path

_mplconfigdir = Path(os.environ.get("MPLCONFIGDIR", Path.cwd() / ".mplconfig"))
_mplconfigdir.mkdir(parents=True, exist_ok=True)
_xdg_cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.cwd() / ".cache"))
_xdg_cache_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache_home))
os.environ.setdefault("MPLBACKEND", "Agg")

from .compare import batch_compare, compare_trace
from .inference import BayesianConfig, BayesianResult, fit_bayesian
from .io import load_manifest, load_trace
from .legacy import LegacyResult, run_legacy
from .schema import ComparisonResult, TraceData
from .adapters import cylindrical_probe_metadata
from .workflows import build_directory_manifest, process_trace_directory

__all__ = [
    "BayesianConfig",
    "BayesianResult",
    "ComparisonResult",
    "LegacyResult",
    "TraceData",
    "batch_compare",
    "build_directory_manifest",
    "compare_trace",
    "cylindrical_probe_metadata",
    "fit_bayesian",
    "load_manifest",
    "load_trace",
    "process_trace_directory",
    "run_legacy",
]
