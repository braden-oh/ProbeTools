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
from .forward import cylindrical_probe_area
from .inference import (
    BayesianConfig,
    BayesianResult,
    compare_physical_models,
    fit_bayesian,
    fit_bayesian_model,
)
from .io import load_manifest, load_trace
from .legacy import LegacyResult, run_legacy
from .schema import BayesianModelResult, ComparisonResult, ModelComparisonAnalysis, ProbeGeometry, TraceData
from .adapters import cylindrical_probe_metadata
from .workflows import build_directory_manifest, process_trace_directory

__all__ = [
    "BayesianConfig",
    "BayesianModelResult",
    "BayesianResult",
    "ComparisonResult",
    "LegacyResult",
    "ModelComparisonAnalysis",
    "ProbeGeometry",
    "TraceData",
    "batch_compare",
    "build_directory_manifest",
    "compare_physical_models",
    "compare_trace",
    "cylindrical_probe_area",
    "cylindrical_probe_metadata",
    "fit_bayesian",
    "fit_bayesian_model",
    "load_manifest",
    "load_trace",
    "process_trace_directory",
    "run_legacy",
]
