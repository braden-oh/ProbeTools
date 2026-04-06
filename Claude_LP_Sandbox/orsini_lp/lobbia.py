from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .schema import LegacyResult, TraceData, single_value_summary


QE = 1.602176634e-19
ME = 9.1093837015e-31
EPS0 = 8.8541878128e-12
GAS_MASSES_AMU = {
    "xe": 131.293,
    "kr": 83.798,
    "ar": 39.948,
    "n": 14.007,
    "zn": 65.38,
}
AMU_TO_KG = 1.66053906660e-27


@dataclass(slots=True)
class _LineFit:
    slope: float
    intercept: float
    slope_stderr: float
    intercept_stderr: float
    r_squared: float


@dataclass(slots=True)
class _VpEstimate:
    value: float
    std: float
    index: int
    method: str
    smoothed_electron_current: np.ndarray
    derivative: np.ndarray


@dataclass(slots=True)
class _TeEstimate:
    value: float
    std: float
    line_fit: _LineFit
    window_mask: np.ndarray
    method: str
    upper_bound_v: float


def run_lobbia(trace: TraceData, max_iterations: int = 8) -> LegacyResult:
    gas_key = (trace.gas or "").strip().lower()
    if gas_key not in GAS_MASSES_AMU:
        return LegacyResult(
            trace_id=trace.trace_id,
            summary=pd.DataFrame(),
            derived_quantities=None,
            diagnostic_trace=None,
            metadata={"skipped": True},
            success=False,
            error=f"Unsupported gas species {trace.gas!r} for the Lobbia legacy solver.",
        )

    try:
        bias_voltage, probe_current, sign_flipped = _normalize_trace(trace.bias_voltage, trace.probe_current)
        floating_potential, floating_std = _estimate_floating_potential(bias_voltage, probe_current)
        ion_mass_kg = GAS_MASSES_AMU[gas_key] * AMU_TO_KG

        ion_line = _fit_preliminary_ion_line(bias_voltage, probe_current, floating_potential)
        if ion_line is None:
            raise ValueError("Not enough points below the floating potential for the preliminary ion-current fit.")

        preliminary_ion_current = np.minimum(ion_line.slope * bias_voltage + ion_line.intercept, 0.0)
        electron_current = np.clip(probe_current - preliminary_ion_current, 0.0, None)

        vp_estimate = _estimate_plasma_potential(
            bias_voltage,
            electron_current,
            floating_potential,
        )
        plasma_potential = vp_estimate.value
        plasma_potential_std = vp_estimate.std

        semilog_fit = _estimate_electron_temperature(
            bias_voltage=bias_voltage,
            electron_current=electron_current,
            floating_potential=floating_potential,
            plasma_potential=plasma_potential,
            ion_mass_kg=ion_mass_kg,
        )
        electron_temperature = semilog_fit.value
        electron_temperature_std = semilog_fit.std

        electron_sat_current = float(np.interp(plasma_potential, bias_voltage, electron_current))
        current_noise = _estimate_current_noise(
            electron_current=electron_current,
            bias_voltage=bias_voltage,
            semilog_fit=semilog_fit,
        )
        electron_sat_current_std = max(
            current_noise,
            abs(semilog_fit.line_fit.slope) * plasma_potential_std * max(electron_sat_current, 1.0e-12),
        )

        electron_density = _electron_density(
            electron_sat_current,
            electron_temperature,
            trace.probe_area_m2,
        )
        electron_density_std = abs(electron_density) * np.sqrt(
            (_safe_fraction(electron_sat_current_std, electron_sat_current)) ** 2
            + (0.5 * _safe_fraction(electron_temperature_std, electron_temperature)) ** 2
        )

        debye_length_m = _debye_length(electron_density, electron_temperature)
        rp_over_lambda = float(trace.probe_radius_m / debye_length_m) if debye_length_m > 0.0 else np.nan

        ion_fit = _fit_corrected_ion_current(
            bias_voltage=bias_voltage,
            probe_current=probe_current,
            floating_potential=floating_potential,
            plasma_potential=plasma_potential,
            electron_temperature=electron_temperature,
            electron_temperature_std=electron_temperature_std,
            electron_density=electron_density,
            probe_area_m2=trace.probe_area_m2,
            probe_radius_m=trace.probe_radius_m,
            probe_geometry=trace.probe_geometry,
            ion_mass_kg=ion_mass_kg,
        )

        sheath_regime = ion_fit["sheath_regime"]
        ion_density = float(ion_fit["ion_density_m3"])
        ion_density_std = float(ion_fit["ion_density_std_m3"])
        corrected_ion_current = np.asarray(ion_fit["ion_current_model_a"], dtype=float)
        ion_sat_current = float(corrected_ion_current[0])
        ion_sat_current_std = float(ion_fit["ion_sat_current_std_a"])

        quasineutral_density = _nanmean([electron_density, ion_density])
        quasineutral_density_std = _nanstd([electron_density, ion_density])

        legacy_model_mask = bias_voltage <= plasma_potential
        electron_fit_current = np.full_like(bias_voltage, np.nan, dtype=float)
        electron_fit_current[legacy_model_mask] = np.exp(
            semilog_fit.line_fit.slope * bias_voltage[legacy_model_mask] + semilog_fit.line_fit.intercept
        )
        legacy_total_model = preliminary_ion_current + electron_fit_current

        derived = pd.DataFrame(
            {
                "Value": [
                    floating_potential,
                    plasma_potential,
                    electron_temperature,
                    quasineutral_density,
                    electron_density,
                    ion_density,
                    electron_sat_current,
                    ion_sat_current,
                ],
                "Uncertainty": [
                    floating_std,
                    plasma_potential_std,
                    electron_temperature_std,
                    quasineutral_density_std,
                    electron_density_std,
                    ion_density_std,
                    electron_sat_current_std,
                    ion_sat_current_std,
                ],
                "Unit": ["V", "V", "eV", "m-3", "m-3", "m-3", "A", "A"],
            },
            index=["Vf", "Vp", "Te", "n", "ne", "ni", "Ie_sat", "Ii_sat"],
        )

        summary_parts = []
        for parameter_name, row in derived.iterrows():
            summary_parts.append(
                single_value_summary(
                    name=parameter_name,
                    value=float(row["Value"]),
                    uncertainty=float(row["Uncertainty"]),
                )
            )
        summary = pd.concat(summary_parts)

        diagnostic_trace = pd.DataFrame(
            {
                "bias_voltage_v": bias_voltage,
                "probe_current_a": probe_current,
                "ion_current_a": preliminary_ion_current,
                "corrected_ion_current_a": corrected_ion_current,
                "electron_current_a": electron_current,
                "electron_current_smooth_a": vp_estimate.smoothed_electron_current,
                "dIe_dV_a_per_v": vp_estimate.derivative,
                "ln_electron_current": _safe_log(electron_current),
                "semilog_fit_ln_current": np.where(
                    semilog_fit.window_mask,
                    semilog_fit.line_fit.slope * bias_voltage + semilog_fit.line_fit.intercept,
                    np.nan,
                ),
                "semilog_fit_current_a": electron_fit_current,
                "legacy_total_model_a": legacy_total_model,
                "semilog_window": semilog_fit.window_mask.astype(int),
                "legacy_model_valid": legacy_model_mask.astype(int),
            }
        )

        metadata: dict[str, Any] = {
            "gas": trace.gas,
            "probe_geometry": trace.probe_geometry,
            "floating_potential_v": floating_potential,
            "floating_potential_std_v": floating_std,
            "vp_method": vp_estimate.method,
            "te_method": semilog_fit.method,
            "sheath_regime": sheath_regime,
            "rp_over_lambda_d": rp_over_lambda,
            "debye_length_m": debye_length_m,
            "iteration_count": 1,
            "converged": True,
            "sign_flipped": sign_flipped,
            "ion_fit_r_squared": ion_fit["ion_fit_r_squared"],
            "semilog_r_squared": semilog_fit.line_fit.r_squared,
            "semilog_upper_bound_v": semilog_fit.upper_bound_v,
            "ion_baseline_method": "preliminary-linear-fit",
        }

        return LegacyResult(
            trace_id=trace.trace_id,
            summary=summary,
            derived_quantities=derived,
            diagnostic_trace=diagnostic_trace,
            metadata=metadata,
            success=True,
        )
    except Exception as exc:  # pragma: no cover - exercised in real-data runs
        return LegacyResult(
            trace_id=trace.trace_id,
            summary=pd.DataFrame(),
            derived_quantities=None,
            diagnostic_trace=None,
            metadata={"skipped": True},
            success=False,
            error=f"Lobbia legacy solver failed: {exc}",
        )


def _normalize_trace(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    voltage = np.asarray(bias_voltage, dtype=float)
    current = np.asarray(probe_current, dtype=float)
    finite = np.isfinite(voltage) & np.isfinite(current)
    voltage = voltage[finite]
    current = current[finite]
    order = np.argsort(voltage)
    voltage = voltage[order]
    current = current[order]

    sign_flipped = False
    if np.corrcoef(voltage, current)[0, 1] < 0.0:
        current = -current
        sign_flipped = True
    return voltage, current, sign_flipped


def _estimate_floating_potential(bias_voltage: np.ndarray, probe_current: np.ndarray) -> tuple[float, float]:
    sign_changes = np.flatnonzero(np.diff(np.signbit(probe_current)))
    if sign_changes.size:
        index = int(sign_changes[np.argmin(np.abs(probe_current[sign_changes]) + np.abs(probe_current[sign_changes + 1]))])
        x0 = bias_voltage[index]
        x1 = bias_voltage[index + 1]
        y0 = probe_current[index]
        y1 = probe_current[index + 1]
        slope = (y1 - y0) / max(x1 - x0, 1.0e-12)
        floating_potential = x0 - y0 / slope if slope != 0.0 else x0
        floating_std = abs(x1 - x0) / 2.0
        return float(floating_potential), float(max(floating_std, _median_step(bias_voltage)))

    nearest = int(np.argmin(np.abs(probe_current)))
    floating_std = _median_step(bias_voltage)
    return float(bias_voltage[nearest]), float(floating_std)


def _fit_preliminary_ion_line(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    floating_potential: float,
) -> _LineFit | None:
    ion_mask = bias_voltage <= floating_potential
    if int(np.count_nonzero(ion_mask)) < 3:
        return None
    return _line_fit(bias_voltage[ion_mask], probe_current[ion_mask])


def _estimate_plasma_potential(
    bias_voltage: np.ndarray,
    electron_current: np.ndarray,
    floating_potential: float,
) -> _VpEstimate:
    dv = _median_step(bias_voltage)
    smoothed = _smooth_positive_series(electron_current)
    derivative = np.gradient(smoothed, bias_voltage)

    search_mask = (bias_voltage >= floating_potential) & np.isfinite(derivative)
    search_indices = np.flatnonzero(search_mask)
    if search_indices.size < 5:
        index = int(np.nanargmax(derivative))
        return _VpEstimate(
            value=float(bias_voltage[index]),
            std=float(max(2.0 * dv, 0.1)),
            index=index,
            method="global-derivative-maximum",
            smoothed_electron_current=smoothed,
            derivative=derivative,
        )

    search_derivative = derivative[search_mask]
    prominence = max(5.0 * np.nanstd(search_derivative[: max(5, search_derivative.size // 8)]), 0.05 * np.nanmax(search_derivative))
    peaks, properties = find_peaks(search_derivative, prominence=prominence)
    if peaks.size:
        peak_scores = search_derivative[peaks]
        best_local = int(peaks[np.argmax(peak_scores)])
        best_index = int(search_indices[best_local])
        widths = peak_widths(search_derivative, [best_local], rel_height=0.5)[0]
        vp_std = max(float(widths[0] * dv / 2.355), dv)
        method = "derivative-peak"
    else:
        best_local = int(np.nanargmax(search_derivative))
        best_index = int(search_indices[best_local])
        vp_std = max(2.0 * dv, 0.1)
        method = "global-derivative-maximum"

    return _VpEstimate(
        value=float(bias_voltage[best_index]),
        std=float(vp_std),
        index=best_index,
        method=method,
        smoothed_electron_current=smoothed,
        derivative=derivative,
    )


def _estimate_electron_temperature(
    *,
    bias_voltage: np.ndarray,
    electron_current: np.ndarray,
    floating_potential: float,
    plasma_potential: float,
    ion_mass_kg: float,
) -> _TeEstimate:
    dv = _median_step(bias_voltage)
    noise_floor = max(np.nanstd(electron_current[bias_voltage <= floating_potential]), 1.0e-12)

    initial_mask = (
        (bias_voltage >= floating_potential)
        & (bias_voltage <= plasma_potential)
        & (electron_current > max(3.0 * noise_floor, 1.0e-12))
    )
    if int(np.count_nonzero(initial_mask)) < 8:
        potential_temperature = _potential_method_temperature(plasma_potential, floating_potential, ion_mass_kg)
        fit = _LineFit(
            slope=1.0 / max(potential_temperature, 1.0e-6),
            intercept=float(np.log(np.max(electron_current))),
            slope_stderr=0.0,
            intercept_stderr=0.0,
            r_squared=np.nan,
        )
        return _TeEstimate(
            value=float(potential_temperature),
            std=float(max(dv, 0.1 * potential_temperature)),
            line_fit=fit,
            window_mask=initial_mask,
            method="potential-difference-fallback",
            upper_bound_v=float(plasma_potential),
        )

    initial_fit, initial_window = _best_semilog_window(
        bias_voltage[initial_mask],
        _safe_log(electron_current[initial_mask]),
    )
    te_initial = 1.0 / initial_fit.slope
    refined_upper = plasma_potential - 2.0 * te_initial
    refined_mask = initial_mask.copy()
    method = "semilog-window"
    if refined_upper > floating_potential + 4.0 * dv:
        refined_mask = initial_mask & (bias_voltage <= refined_upper)

    if int(np.count_nonzero(refined_mask)) >= 8:
        fit, window_local = _best_semilog_window(
            bias_voltage[refined_mask],
            _safe_log(electron_current[refined_mask]),
        )
        selected_points = np.flatnonzero(refined_mask)[window_local]
    else:
        fit = initial_fit
        selected_points = np.flatnonzero(initial_mask)[initial_window]
        refined_upper = plasma_potential
        method = "semilog-window-unrefined"

    temperature = 1.0 / fit.slope
    temperature_std = abs(fit.slope_stderr / max(fit.slope**2, 1.0e-18))
    window_mask = np.zeros_like(bias_voltage, dtype=bool)
    window_mask[selected_points] = True
    return _TeEstimate(
        value=float(temperature),
        std=float(max(temperature_std, 0.05)),
        line_fit=fit,
        window_mask=window_mask,
        method=method,
        upper_bound_v=float(refined_upper),
    )


def _best_semilog_window(
    bias_voltage: np.ndarray,
    ln_electron_current: np.ndarray,
    min_points: int = 8,
) -> tuple[_LineFit, np.ndarray]:
    point_count = bias_voltage.size
    if point_count < min_points:
        return _line_fit(bias_voltage, ln_electron_current), np.ones(point_count, dtype=bool)

    best_fit = None
    best_window = None
    best_score = -np.inf

    for start in range(0, point_count - min_points + 1):
        for stop in range(start + min_points, point_count + 1):
            fit = _line_fit(bias_voltage[start:stop], ln_electron_current[start:stop])
            if not np.isfinite(fit.slope) or fit.slope <= 0.0:
                continue
            span = bias_voltage[stop - 1] - bias_voltage[start]
            score = fit.r_squared * span * (stop - start)
            if score > best_score:
                best_fit = fit
                best_score = score
                best_window = (start, stop)

    if best_fit is None or best_window is None:
        return _line_fit(bias_voltage, ln_electron_current), np.ones(point_count, dtype=bool)

    window_mask = np.zeros(point_count, dtype=bool)
    window_mask[best_window[0] : best_window[1]] = True
    return best_fit, window_mask


def _fit_corrected_ion_current(
    *,
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    floating_potential: float,
    plasma_potential: float,
    electron_temperature: float,
    electron_temperature_std: float,
    electron_density: float,
    probe_area_m2: float,
    probe_radius_m: float,
    probe_geometry: str,
    ion_mass_kg: float,
) -> dict[str, Any]:
    debye_length_m = _debye_length(electron_density, electron_temperature)
    ratio = probe_radius_m / debye_length_m if debye_length_m > 0.0 else np.nan
    geometry = (probe_geometry or "cylindrical").strip().lower()

    if ratio >= 50.0:
        sheath_regime = "thin-sheath"
        sheath_area = _child_langmuir_area(
            bias_voltage=bias_voltage,
            plasma_potential=plasma_potential,
            electron_temperature=electron_temperature,
            debye_length_m=debye_length_m,
            probe_area_m2=probe_area_m2,
            probe_radius_m=probe_radius_m,
            probe_geometry=geometry,
        )
        ion_sat_current = float(np.interp(np.min(bias_voltage), bias_voltage, probe_current))
        ion_density = _thin_sheath_ion_density(
            ion_sat_current,
            electron_temperature,
            float(sheath_area[0]),
            ion_mass_kg,
        )
        ion_density_std = abs(ion_density) * 0.1
        ion_current_model = -np.exp(-0.5) * QE * ion_density * np.sqrt(QE * electron_temperature / ion_mass_kg) * sheath_area
        return {
            "sheath_regime": sheath_regime,
            "ion_density_m3": ion_density,
            "ion_density_std_m3": ion_density_std,
            "ion_current_model_a": ion_current_model,
            "ion_sat_current_std_a": max(np.nanstd(probe_current[bias_voltage <= floating_potential]), 1.0e-12),
            "ion_fit_r_squared": np.nan,
        }

    if ratio <= 3.0:
        sheath_regime = "oml"
        if geometry == "cylindrical":
            a_param = 2.0 / np.sqrt(np.pi)
            b_param = 0.5
        else:
            a_param = 1.0
            b_param = 1.0
    else:
        sheath_regime = "transitional"
        a_param, b_param = _transitional_sheath_parameters(geometry, ratio)

    ion_mask = bias_voltage <= floating_potential
    transformed_current = np.power(np.clip(-probe_current[ion_mask], 1.0e-18, None), 1.0 / b_param)
    fit = _line_fit(bias_voltage[ion_mask], transformed_current)
    slope_term = max(-fit.slope, 1.0e-18)

    ion_density = (
        1.0
        / (a_param * probe_area_m2)
        * np.sqrt(2.0 * np.pi * ion_mass_kg)
        / (QE ** 1.5)
        * electron_temperature ** (b_param - 0.5)
        * slope_term**b_param
    )
    ion_density_std = abs(ion_density) * np.sqrt(
        (b_param * _safe_fraction(fit.slope_stderr, fit.slope)) ** 2
        + ((b_param - 0.5) * _safe_fraction(electron_temperature_std, electron_temperature)) ** 2
    )

    delta = np.clip((plasma_potential - bias_voltage) / max(electron_temperature, 1.0e-6), 0.0, None)
    ion_current_model = (
        -a_param
        * QE
        * ion_density
        * probe_area_m2
        * np.sqrt(QE * electron_temperature / (2.0 * np.pi * ion_mass_kg))
        * np.power(delta, b_param)
    )
    ion_sat_current_std = max(np.nanstd(probe_current[ion_mask] - np.interp(bias_voltage[ion_mask], bias_voltage, ion_current_model)), 1.0e-12)
    return {
        "sheath_regime": sheath_regime,
        "ion_density_m3": ion_density,
        "ion_density_std_m3": ion_density_std,
        "ion_current_model_a": ion_current_model,
        "ion_sat_current_std_a": ion_sat_current_std,
        "ion_fit_r_squared": fit.r_squared,
    }


def _transitional_sheath_parameters(probe_geometry: str, ratio: float) -> tuple[float, float]:
    geometry = probe_geometry.strip().lower()
    if geometry == "planar":
        return 3.47 * ratio**-0.749, 0.806 * ratio**-0.0692
    if geometry == "spherical":
        a_param = 1.58 + (-0.056 + 0.816 * ratio) ** -0.744
        b_param = -0.933 + (0.0148 + 0.119 * ratio) ** -0.125
        return a_param, b_param
    a_param = 1.18 - 0.00080 * ratio**1.35
    b_param = 0.0684 + (0.722 + 0.928 * ratio) ** -0.729
    return a_param, b_param


def _child_langmuir_area(
    *,
    bias_voltage: np.ndarray,
    plasma_potential: float,
    electron_temperature: float,
    debye_length_m: float,
    probe_area_m2: float,
    probe_radius_m: float,
    probe_geometry: str,
) -> np.ndarray:
    delta = np.clip(plasma_potential - bias_voltage, 0.0, None)
    sheath_thickness = debye_length_m * np.sqrt(2.0) / 3.0 * np.power(
        np.clip(2.0 * delta / max(electron_temperature, 1.0e-6), 0.0, None),
        0.75,
    )
    geometry = probe_geometry.strip().lower()
    if geometry == "planar":
        return np.full_like(bias_voltage, probe_area_m2, dtype=float)
    if geometry == "spherical":
        return probe_area_m2 * np.power(1.0 + sheath_thickness / probe_radius_m, 2.0)
    return probe_area_m2 * (1.0 + sheath_thickness / probe_radius_m)


def _thin_sheath_ion_density(
    ion_sat_current: float,
    electron_temperature: float,
    sheath_area_m2: float,
    ion_mass_kg: float,
) -> float:
    return -np.exp(0.5) * ion_sat_current / (QE * sheath_area_m2) * np.sqrt(ion_mass_kg / (QE * electron_temperature))


def _electron_density(
    electron_sat_current: float,
    electron_temperature: float,
    probe_area_m2: float,
) -> float:
    denominator = QE * probe_area_m2 * np.sqrt(QE * electron_temperature / (2.0 * np.pi * ME))
    return electron_sat_current / max(denominator, 1.0e-30)


def _debye_length(electron_density: float, electron_temperature: float) -> float:
    if electron_density <= 0.0 or electron_temperature <= 0.0:
        return np.nan
    return float(np.sqrt(EPS0 * electron_temperature / (electron_density * QE)))


def _potential_method_temperature(plasma_potential: float, floating_potential: float, ion_mass_kg: float) -> float:
    ratio_term = np.log(np.sqrt(ion_mass_kg / (2.0 * np.pi * ME)))
    return max((plasma_potential - floating_potential) / max(ratio_term, 1.0e-6), 0.1)


def _line_fit(x: np.ndarray, y: np.ndarray) -> _LineFit:
    result = stats.linregress(x, y)
    r_squared = result.rvalue**2 if np.isfinite(result.rvalue) else np.nan
    intercept_stderr = float(result.intercept_stderr) if getattr(result, "intercept_stderr", None) is not None else 0.0
    return _LineFit(
        slope=float(result.slope),
        intercept=float(result.intercept),
        slope_stderr=float(result.stderr or 0.0),
        intercept_stderr=intercept_stderr,
        r_squared=float(r_squared),
    )


def _smooth_positive_series(values: np.ndarray) -> np.ndarray:
    positive = np.clip(np.asarray(values, dtype=float), 0.0, None)
    point_count = positive.size
    if point_count < 7:
        return positive
    window = min(21, point_count if point_count % 2 == 1 else point_count - 1)
    if window < 5:
        return positive
    return np.clip(savgol_filter(positive, window_length=window, polyorder=3, mode="interp"), 0.0, None)


def _estimate_current_noise(
    *,
    electron_current: np.ndarray,
    bias_voltage: np.ndarray,
    semilog_fit: _TeEstimate,
) -> float:
    semilog_window_mask = semilog_fit.window_mask
    if int(np.count_nonzero(semilog_window_mask)) >= 3:
        signal = electron_current[semilog_window_mask]
        model = np.exp(
            semilog_fit.line_fit.slope * bias_voltage[semilog_window_mask] + semilog_fit.line_fit.intercept
        )
        return float(max(np.nanstd(signal - model), 1.0e-12))
    return float(max(np.nanstd(electron_current), 1.0e-12))


def _safe_log(values: np.ndarray) -> np.ndarray:
    output = np.full_like(np.asarray(values, dtype=float), np.nan, dtype=float)
    mask = np.asarray(values) > 0.0
    output[mask] = np.log(np.asarray(values, dtype=float)[mask])
    return output


def _median_step(values: np.ndarray) -> float:
    diffs = np.diff(np.asarray(values, dtype=float))
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    return numerator / denominator


def _nanmean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(finite))


def _nanstd(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite, ddof=1))
