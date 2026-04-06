from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from .inference import _select_dense_sweep_mask
from .io import _guess_delimiter


EXPECTED_TRACE_COLUMNS = ("Bias Voltage (V)", "Probe Current (A)")


@dataclass(slots=True)
class SemilogFitResult:
    trace_id: str
    trace_path: Path
    bias_voltage_v: np.ndarray
    probe_current_a: np.ndarray
    shifted_current_a: np.ndarray
    log_shifted_current: np.ndarray
    dense_sweep_mask: np.ndarray
    fit_mask: np.ndarray
    current_shift_a: float
    slope_inv_v: float
    slope_stderr_inv_v: float
    intercept_log_a: float
    r_squared: float
    electron_temperature_ev: float
    electron_temperature_stderr_ev: float
    fit_lower_v: float
    fit_upper_v: float
    fit_point_count: int
    lower_current_fraction: float
    upper_current_fraction: float


def fit_semilog_trace(
    trace_path: str | Path,
    lower_current_fraction: float = 0.02,
    upper_current_fraction: float = 0.70,
    min_points: int = 12,
    max_points: int = 80,
) -> SemilogFitResult:
    if not (0.0 <= lower_current_fraction < upper_current_fraction <= 1.0):
        raise ValueError("Current-fraction bounds must satisfy 0 <= lower < upper <= 1.")
    if min_points < 3:
        raise ValueError("min_points must be at least 3.")
    if max_points < min_points:
        raise ValueError("max_points must be greater than or equal to min_points.")

    path = Path(trace_path).resolve()
    bias_voltage, probe_current = load_two_column_trace(path)
    dense_sweep_mask = _select_dense_sweep_mask(bias_voltage)

    dense_bias = bias_voltage[dense_sweep_mask]
    dense_current = probe_current[dense_sweep_mask]
    shifted_dense_current, current_shift = shift_current_positive(dense_current)
    dense_log_current = np.log(shifted_dense_current)

    dense_fit_mask = select_semilog_fit_window(
        bias_voltage_v=dense_bias,
        shifted_current_a=shifted_dense_current,
        lower_current_fraction=lower_current_fraction,
        upper_current_fraction=upper_current_fraction,
        min_points=min_points,
        max_points=max_points,
    )
    regression = _fit_line(dense_bias[dense_fit_mask], dense_log_current[dense_fit_mask])

    fit_mask = np.zeros_like(bias_voltage, dtype=bool)
    dense_indices = np.flatnonzero(dense_sweep_mask)
    fit_mask[dense_indices[dense_fit_mask]] = True

    shifted_current = np.full_like(probe_current, np.nan, dtype=float)
    shifted_current[dense_sweep_mask] = shifted_dense_current
    log_shifted_current = np.full_like(probe_current, np.nan, dtype=float)
    log_shifted_current[dense_sweep_mask] = dense_log_current

    te_ev = 1.0 / regression.slope
    te_stderr_ev = abs(regression.stderr / (regression.slope**2)) if np.isfinite(regression.stderr) else np.nan

    return SemilogFitResult(
        trace_id=path.stem.lower(),
        trace_path=path,
        bias_voltage_v=bias_voltage,
        probe_current_a=probe_current,
        shifted_current_a=shifted_current,
        log_shifted_current=log_shifted_current,
        dense_sweep_mask=dense_sweep_mask,
        fit_mask=fit_mask,
        current_shift_a=current_shift,
        slope_inv_v=regression.slope,
        slope_stderr_inv_v=float(regression.stderr) if np.isfinite(regression.stderr) else np.nan,
        intercept_log_a=regression.intercept,
        r_squared=regression.rvalue**2,
        electron_temperature_ev=te_ev,
        electron_temperature_stderr_ev=te_stderr_ev,
        fit_lower_v=float(np.min(bias_voltage[fit_mask])),
        fit_upper_v=float(np.max(bias_voltage[fit_mask])),
        fit_point_count=int(np.count_nonzero(fit_mask)),
        lower_current_fraction=lower_current_fraction,
        upper_current_fraction=upper_current_fraction,
    )


def load_two_column_trace(trace_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(trace_path)
    frame = pd.read_csv(path, sep=_guess_delimiter(path))
    if set(EXPECTED_TRACE_COLUMNS).issubset(frame.columns):
        bias_voltage = frame[EXPECTED_TRACE_COLUMNS[0]].to_numpy(dtype=float)
        probe_current = frame[EXPECTED_TRACE_COLUMNS[1]].to_numpy(dtype=float)
    elif frame.shape[1] >= 2:
        bias_voltage = frame.iloc[:, 0].to_numpy(dtype=float)
        probe_current = frame.iloc[:, 1].to_numpy(dtype=float)
    else:
        raise ValueError(f"Expected at least two columns in {path}")

    order = np.argsort(bias_voltage)
    return bias_voltage[order], probe_current[order]


def shift_current_positive(probe_current_a: np.ndarray, pad_fraction: float = 1.0e-6) -> tuple[np.ndarray, float]:
    current = np.asarray(probe_current_a, dtype=float)
    current_span = max(float(np.ptp(current)), 1.0e-12)
    current_shift = max(0.0, -float(np.min(current))) + pad_fraction * current_span
    return current + current_shift, current_shift


def select_semilog_fit_window(
    bias_voltage_v: np.ndarray,
    shifted_current_a: np.ndarray,
    lower_current_fraction: float,
    upper_current_fraction: float,
    min_points: int,
    max_points: int,
) -> np.ndarray:
    bias_voltage = np.asarray(bias_voltage_v, dtype=float)
    shifted_current = np.asarray(shifted_current_a, dtype=float)
    if bias_voltage.size != shifted_current.size:
        raise ValueError("bias_voltage_v and shifted_current_a must have the same length.")

    current_fraction = _current_fraction(shifted_current)
    log_shifted_current = np.log(shifted_current)
    window_limit = min(max_points, bias_voltage.size)
    best_window: tuple[float, float, int, int, int] | None = None

    for start in range(0, bias_voltage.size - min_points + 1):
        stop_upper = min(bias_voltage.size, start + window_limit)
        for stop in range(start + min_points, stop_upper + 1):
            window_fraction = current_fraction[start:stop]
            if float(np.min(window_fraction)) < lower_current_fraction or float(np.max(window_fraction)) > upper_current_fraction:
                continue
            try:
                regression = _fit_line(bias_voltage[start:stop], log_shifted_current[start:stop])
            except ValueError:
                continue
            score = (
                regression.rvalue**2,
                float(bias_voltage[stop - 1] - bias_voltage[start]),
                stop - start,
                start,
                stop,
            )
            if best_window is None or score > best_window:
                best_window = score

    if best_window is None:
        fallback_mask = (current_fraction >= lower_current_fraction) & (current_fraction <= upper_current_fraction)
        if int(np.count_nonzero(fallback_mask)) < min_points:
            raise ValueError("Could not identify a semilog fit window with the requested settings.")
        fallback_regression = _fit_line(bias_voltage[fallback_mask], log_shifted_current[fallback_mask])
        if fallback_regression.slope <= 0.0:
            raise ValueError("Fallback semilog fit produced a non-positive slope.")
        return fallback_mask

    fit_mask = np.zeros_like(bias_voltage, dtype=bool)
    fit_mask[best_window[3] : best_window[4]] = True
    return fit_mask


def semilog_result_frame(result: SemilogFitResult) -> pd.DataFrame:
    shifted_current = np.asarray(result.shifted_current_a, dtype=float)
    finite_shifted = np.isfinite(shifted_current)
    current_fraction = np.full_like(shifted_current, np.nan, dtype=float)
    if np.any(finite_shifted):
        current_fraction[finite_shifted] = _current_fraction(shifted_current[finite_shifted])
    return pd.DataFrame(
        {
            "bias_voltage_v": result.bias_voltage_v,
            "probe_current_a": result.probe_current_a,
            "shifted_current_a": result.shifted_current_a,
            "log_shifted_current": result.log_shifted_current,
            "dense_sweep_mask": result.dense_sweep_mask.astype(int),
            "semilog_fit_mask": result.fit_mask.astype(int),
            "current_fraction": current_fraction,
        }
    )


def plot_semilog_fit(
    result: SemilogFitResult,
    bayes_te_ev: float | None = None,
):
    fig, (ax_iv, ax_semilog) = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    ax_iv.plot(result.bias_voltage_v, result.probe_current_a, "k.", ms=3, label="Measured trace")
    ax_iv.plot(
        result.bias_voltage_v[result.fit_mask],
        result.probe_current_a[result.fit_mask],
        ".",
        color="tab:blue",
        ms=5,
        label="Semilog fit window",
    )
    ax_iv.axvspan(result.fit_lower_v, result.fit_upper_v, color="tab:blue", alpha=0.08)
    ax_iv.set_ylabel("Probe Current [A]")
    ax_iv.set_title(f"Semilog Te Check: {result.trace_id}")
    ax_iv.grid(True, alpha=0.25)
    ax_iv.legend(loc="best")

    dense_mask = result.dense_sweep_mask
    ax_semilog.semilogy(
        result.bias_voltage_v[dense_mask],
        result.shifted_current_a[dense_mask],
        ".",
        color="0.45",
        ms=3,
        label="Shifted current",
    )
    ax_semilog.semilogy(
        result.bias_voltage_v[result.fit_mask],
        result.shifted_current_a[result.fit_mask],
        ".",
        color="tab:blue",
        ms=5,
        label="Fitted points",
    )
    fit_bias = result.bias_voltage_v[result.fit_mask]
    fit_curve = np.exp(result.intercept_log_a + result.slope_inv_v * fit_bias)
    ax_semilog.semilogy(fit_bias, fit_curve, color="tab:red", lw=2, label="Semilog line fit")
    ax_semilog.grid(True, alpha=0.25, which="both")
    ax_semilog.set_xlabel("Bias Voltage [V]")
    ax_semilog.set_ylabel("Shifted Current [A]")

    note_lines = [
        f"Shift = {result.current_shift_a:.3e} A",
        f"Te = {result.electron_temperature_ev:.3f} eV",
        f"R^2 = {result.r_squared:.4f}",
        f"Window = [{result.fit_lower_v:.2f}, {result.fit_upper_v:.2f}] V",
    ]
    if bayes_te_ev is not None and np.isfinite(bayes_te_ev):
        note_lines.append(f"Bayesian Te = {bayes_te_ev:.3f} eV")
    ax_semilog.text(
        0.02,
        0.98,
        "\n".join(note_lines),
        transform=ax_semilog.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax_semilog.legend(loc="lower right")
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quick semilog electron-temperature cross-check for Langmuir traces. "
            "This uses a constant current shift, so treat it as a diagnostic rather than a full replacement "
            "for ion-current subtraction."
        )
    )
    parser.add_argument("paths", nargs="+", help="Trace files to analyze.")
    parser.add_argument("--output-dir", default="semilog_te_check_output", help="Directory for plots and CSV outputs.")
    parser.add_argument(
        "--bayes-summary",
        default=None,
        help="Optional batch_summary.csv from the Bayesian workflow to merge into the semilog summary.",
    )
    parser.add_argument("--lower-frac", type=float, default=0.02, help="Lower shifted-current fraction for candidate windows.")
    parser.add_argument("--upper-frac", type=float, default=0.70, help="Upper shifted-current fraction for candidate windows.")
    parser.add_argument("--min-points", type=int, default=12, help="Minimum number of points allowed in the fitted segment.")
    parser.add_argument("--max-points", type=int, default=80, help="Maximum number of points allowed in the fitted segment.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bayes_lookup = None
    if args.bayes_summary:
        bayes_frame = pd.read_csv(args.bayes_summary)
        if "trace_id" not in bayes_frame.columns:
            raise ValueError("Bayesian summary must include a trace_id column.")
        bayes_lookup = bayes_frame.set_index("trace_id")

    records: list[dict[str, float | int | str]] = []
    for path_str in args.paths:
        result = fit_semilog_trace(
            trace_path=path_str,
            lower_current_fraction=args.lower_frac,
            upper_current_fraction=args.upper_frac,
            min_points=args.min_points,
            max_points=args.max_points,
        )

        bayes_te_ev = None
        bayes_vp_v = None
        if bayes_lookup is not None and result.trace_id in bayes_lookup.index:
            bayes_row = bayes_lookup.loc[result.trace_id]
            bayes_te_ev = float(bayes_row["Te_bayes_median"]) if "Te_bayes_median" in bayes_row.index else None
            bayes_vp_v = float(bayes_row["Vp_bayes_median"]) if "Vp_bayes_median" in bayes_row.index else None

        plot_path = output_dir / f"{result.trace_id}_semilog_fit.png"
        point_path = output_dir / f"{result.trace_id}_semilog_points.csv"
        semilog_result_frame(result).to_csv(point_path, index=False)
        figure = plot_semilog_fit(result, bayes_te_ev=bayes_te_ev)
        figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)

        record: dict[str, float | int | str] = {
            "trace_id": result.trace_id,
            "trace_path": str(result.trace_path),
            "current_shift_a": result.current_shift_a,
            "semilog_te_ev": result.electron_temperature_ev,
            "semilog_te_stderr_ev": result.electron_temperature_stderr_ev,
            "semilog_slope_inv_v": result.slope_inv_v,
            "semilog_slope_stderr_inv_v": result.slope_stderr_inv_v,
            "semilog_r_squared": result.r_squared,
            "fit_lower_v": result.fit_lower_v,
            "fit_upper_v": result.fit_upper_v,
            "fit_point_count": result.fit_point_count,
            "lower_current_fraction": result.lower_current_fraction,
            "upper_current_fraction": result.upper_current_fraction,
            "plot_path": str(plot_path),
            "points_path": str(point_path),
        }
        if bayes_te_ev is not None:
            record["bayes_te_ev"] = bayes_te_ev
            record["te_delta_semilog_minus_bayes"] = result.electron_temperature_ev - bayes_te_ev
        if bayes_vp_v is not None:
            record["bayes_vp_v"] = bayes_vp_v
        records.append(record)

    summary = pd.DataFrame.from_records(records).sort_values("trace_id")
    summary_path = output_dir / "semilog_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote summary to {summary_path}")
    return 0


def _current_fraction(shifted_current_a: np.ndarray) -> np.ndarray:
    shifted_current = np.asarray(shifted_current_a, dtype=float)
    current_span = max(float(np.ptp(shifted_current)), 1.0e-12)
    return (shifted_current - float(np.min(shifted_current))) / current_span


def _fit_line(x: np.ndarray, y: np.ndarray):
    regression = linregress(x, y)
    if not np.isfinite(regression.slope) or regression.slope <= 0.0:
        raise ValueError("Semilog fit requires a positive finite slope.")
    return regression


if __name__ == "__main__":
    raise SystemExit(main())
