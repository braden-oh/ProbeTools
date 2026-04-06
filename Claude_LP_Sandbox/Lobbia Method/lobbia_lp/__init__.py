"""Lobbia LP: Langmuir probe analysis library implementing Lobbia & Beal (2017).

This package provides tools for analyzing single Langmuir probe I-V characteristics
in electric propulsion plasma environments.

Main entry points:
  - analyze(): Run full 11-step analysis on an I-V trace
  - load_trace(): Load trace data from file

Example:
  >>> from lobbia_lp import analyze, load_trace
  >>> V, I = load_trace('trace.txt')
  >>> result = analyze(V, I, probe_radius_m=1e-4, probe_area_m2=1e-6,
  ...                  probe_length_m=0.01, gas='xe')
  >>> print(result.to_dataframe())
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from .core import analyze, LPResult
from .io import load_trace

__all__ = [
    "analyze",
    "load_trace",
    "LPResult",
]
