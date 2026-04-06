from __future__ import annotations

from .lobbia import run_lobbia
from .schema import LegacyResult, TraceData


def run_legacy(trace: TraceData) -> LegacyResult:
    return run_lobbia(trace)
