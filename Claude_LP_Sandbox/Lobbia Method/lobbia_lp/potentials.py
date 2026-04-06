"""Floating and plasma potential detection (Lobbia Steps 2, 5, 6).

Implements methods for identifying floating potential Vf and plasma potential Vp
from I-V characteristics using multiple independent algorithms with weighted averaging.
"""

from typing import Tuple, List
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit


def find_floating_potential(
    bias_voltage: np.ndarray, probe_current: np.ndarray
) -> Tuple[float, float]:
    """Locate floating potential by sign change in probe current (Lobbia Step 2).

    Vf is the voltage where probe current crosses zero. Computed via linear
    interpolation between the two nearest points.

    Parameters
    ----------
    bias_voltage : ndarray
        Bias voltages [V]
    probe_current : ndarray
        Probe currents [A]

    Returns
    -------
    vf : float
        Floating potential [V]
    vf_uncertainty : float
        Estimated uncertainty [V] (half the voltage step or 0.1 V, whichever larger)
    """
    I = np.asarray(probe_current, dtype=float)

    # Find sign changes
    sign_changes = np.flatnonzero(np.diff(np.sign(I)))

    if len(sign_changes) == 0:
        # No sign change; use nearest-to-zero current point
        idx = np.nanargmin(np.abs(I))
        vf = float(bias_voltage[idx])
        dv = _median_voltage_step(bias_voltage)
        vf_unc = max(dv, 0.1)
        return vf, vf_unc

    # Use first sign change (most reliable)
    idx = int(sign_changes[0])
    V_left, V_right = bias_voltage[idx], bias_voltage[idx + 1]
    I_left, I_right = probe_current[idx], probe_current[idx + 1]

    # Linear interpolation to zero crossing
    if abs(I_right - I_left) > 1e-15:
        vf = V_left - I_left * (V_right - V_left) / (I_right - I_left)
    else:
        vf = 0.5 * (V_left + V_right)

    # Uncertainty: half the voltage step between these two points
    dv_step = abs(V_right - V_left)
    vf_unc = max(dv_step / 2.0, _median_voltage_step(bias_voltage))

    return float(vf), float(vf_unc)


def find_plasma_potential(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    floating_potential: float,
    ion_current: np.ndarray = None,
    electron_temperature: float = None,
    ion_mass_kg: float = None,
) -> Tuple[float, float, dict]:
    """Locate plasma potential using multiple knee-finding algorithms (Lobbia Step 5).

    Implements 5 independent methods following LP_Process.m:
    1. 2nd derivative zero crossing
    2. 1st derivative maximum
    3. Kneedle algorithm (normalized distance from diagonal)
    4. Bisector/segmented fit method
    5. Theoretical potential method (fallback)

    Takes weighted average of successful methods with outlier removal.

    Parameters
    ----------
    bias_voltage : ndarray
        Bias voltages [V]
    probe_current : ndarray
        Probe currents [A]
    floating_potential : float
        Floating potential [V], used to separate electron and ion regions
    ion_current : ndarray, optional
        Ion current [A]. If provided, used to compute electron current as I_e = I - I_i.
    electron_temperature : float, optional
        Electron temperature [eV], used for fallback Vp estimate
    ion_mass_kg : float, optional
        Ion mass [kg], used for fallback formula

    Returns
    -------
    vp : float
        Plasma potential [V]
    vp_uncertainty : float
        Estimated uncertainty [V]
    diagnostics : dict
        Diagnostic info with method details
    """
    V = np.asarray(bias_voltage, dtype=float)
    I = np.asarray(probe_current, dtype=float)

    # Compute electron current in the region above floating potential
    if ion_current is not None:
        Ie = I - ion_current
    else:
        # Rough estimate: constant ion current equal to minimum value below Vf
        ion_idx = V <= floating_potential
        if np.any(ion_idx):
            Ii_sat = I[ion_idx][0]
        else:
            Ii_sat = 0.0
        Ie = I - Ii_sat

    # Clip to positive (safety)
    Ie = np.maximum(Ie, 0.0)

    # Define search region (Vf to 30V above Vf, matching MATLAB)
    vp_search_upper = floating_potential + 30.0
    search_mask = (V >= floating_potential) & (V <= vp_search_upper)
    search_indices = np.flatnonzero(search_mask)

    if len(search_indices) < 5:
        # Not enough data points; fall back to simple approach
        return _fallback_plasma_potential(
            V, Ie, floating_potential, electron_temperature, ion_mass_kg
        )

    V_search = V[search_indices]
    Ie_search = Ie[search_indices]

    # Smooth electron current
    Ie_smooth = _smooth_positive_series(Ie_search, polyorder=3)

    # Compute derivatives
    dIe_dV = np.gradient(Ie_smooth, V_search)
    d2Ie_dV2 = np.gradient(dIe_dV, V_search)

    # Weights for each method (matching MATLAB: w = [1/2  3/2  1  1  2])
    weights = np.array([0.5, 1.5, 1.0, 1.0, 2.0])

    # Try all 5 methods
    vp_candidates = []
    method_names = []

    # Method 1: 2nd derivative zero crossing (d²Ie/dV² goes from + to -)
    try:
        idx = _find_second_deriv_zero_crossing(d2Ie_dV2)
        if idx is not None:
            vp_candidates.append((V_search[idx], weights[0]))
            method_names.append("2nd_deriv_zero")
    except Exception:
        pass

    # Method 2: 1st derivative maximum
    try:
        idx = np.argmax(dIe_dV)
        vp_candidates.append((V_search[idx], weights[1]))
        method_names.append("1st_deriv_max")
    except Exception:
        pass

    # Method 3: Kneedle algorithm
    try:
        idx = _kneedle_algorithm(V_search, Ie_smooth)
        if idx is not None:
            vp_candidates.append((V_search[idx], weights[2]))
            method_names.append("kneedle")
    except Exception:
        pass

    # Method 4: Bisector/segmented fit method
    try:
        idx = _bisector_knee_find(V_search, Ie_smooth)
        if idx is not None:
            vp_candidates.append((V_search[idx], weights[3]))
            method_names.append("bisector")
    except Exception:
        pass

    # Method 5: Theoretical potential method (if Te available)
    if electron_temperature is not None and ion_mass_kg is not None:
        try:
            vp_theory = _potential_method_vp(
                floating_potential, electron_temperature, ion_mass_kg
            )
            # Find nearest index in search region
            idx = np.argmin(np.abs(V_search - vp_theory))
            if 0 <= idx < len(V_search):
                vp_candidates.append((V_search[idx], weights[4]))
                method_names.append("potential_method")
        except Exception:
            pass

    # If no methods succeeded, use fallback
    if len(vp_candidates) == 0:
        return _fallback_plasma_potential(
            V, Ie, floating_potential, electron_temperature, ion_mass_kg
        )

    # Extract values and weights
    vp_values = np.array([v for v, w in vp_candidates])
    vp_weights = np.array([w for v, w in vp_candidates])

    # Remove outliers using IQR method (matching MATLAB approach)
    Q1 = np.percentile(vp_values, 25)
    Q3 = np.percentile(vp_values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    valid_mask = (vp_values >= lower_bound) & (vp_values <= upper_bound)
    vp_values_valid = vp_values[valid_mask]
    vp_weights_valid = vp_weights[valid_mask]
    method_names_valid = [m for m, v in zip(method_names, valid_mask) if v]

    if len(vp_values_valid) == 0:
        # All outliers; use the closest to median
        vp = float(np.median(vp_values))
    else:
        # Weighted average of remaining methods
        vp = float(np.average(vp_values_valid, weights=vp_weights_valid))

    # Estimate uncertainty
    dv_step = _median_voltage_step(V)
    vp_unc = max(2.0 * dv_step, 0.5)

    diagnostics = {
        "method": "multi-method-average",
        "methods_used": method_names_valid,
        "vp_candidates": vp_values,
        "vp_weights": vp_weights,
        "smoothed_electron_current": Ie_smooth,
        "derivative": dIe_dV,
        "second_derivative": d2Ie_dV2,
        "search_indices": search_indices,
    }

    return float(vp), float(vp_unc), diagnostics


def _find_second_deriv_zero_crossing(second_deriv: np.ndarray) -> int:
    """Find where 2nd derivative changes from positive to negative.

    This marks the inflection point of the exponential curve (where curvature changes).
    """
    sign_changes = np.diff(np.sign(second_deriv))
    # Look for +2 → -2 transition (positive to negative)
    zero_crossings = np.flatnonzero(sign_changes == -2)

    if len(zero_crossings) > 0:
        return int(zero_crossings[0]) + 1
    return None


def _kneedle_algorithm(x: np.ndarray, y: np.ndarray) -> int:
    """Kneedle algorithm: find point furthest from diagonal.

    Normalizes x and y to [0,1], then finds the point with maximum distance
    from the line connecting (0,0) to (1,1).

    This is robust to scale and finds the "elbow" point.
    """
    # Normalize to [0, 1]
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)

    # Distance from diagonal (y - x for normalized coordinates)
    distances = y_norm - x_norm

    # Return index of maximum distance
    idx = np.argmax(distances)
    return int(idx)


def _bisector_knee_find(x: np.ndarray, y: np.ndarray) -> int:
    """Bisector method: find point that best separates two linear regions.

    Tests every point as a potential knee. For each point, fits linear segments
    before and after, and picks the point with minimum total fitting error.

    Very robust to noise and artifacts.
    """
    errors = np.zeros(len(x))

    for i in range(1, len(x) - 1):
        try:
            # Fit left segment (0 to i)
            if i > 1:
                z1 = np.polyfit(x[:i + 1], y[:i + 1], 1)
                y1_fit = np.polyval(z1, x[:i + 1])
                sse1 = np.sum((y[:i + 1] - y1_fit) ** 2)
            else:
                sse1 = 0.0

            # Fit right segment (i to end)
            if len(x) - i > 1:
                z2 = np.polyfit(x[i:], y[i:], 1)
                y2_fit = np.polyval(z2, x[i:])
                sse2 = np.sum((y[i:] - y2_fit) ** 2)
            else:
                sse2 = 0.0

            errors[i] = np.sqrt(sse1 + sse2)
        except Exception:
            errors[i] = np.inf

    # Ignore endpoints
    errors[0] = np.inf
    errors[-1] = np.inf

    # Find point with minimum error
    idx = np.argmin(errors)
    return int(idx)


def _potential_method_vp(
    floating_potential: float, electron_temperature: float, ion_mass_kg: float
) -> float:
    """Theoretical Vp estimate from electron temperature and ion mass.

    Vp ≈ Vf + Te * ln(sqrt(mi / (2π*me)))
    """
    from .constants import ME, QE

    te_j = electron_temperature * QE
    ratio = np.sqrt(ion_mass_kg / (2.0 * np.pi * ME))
    vp = floating_potential + electron_temperature * np.log(ratio)
    return float(vp)


def _fallback_plasma_potential(
    bias_voltage: np.ndarray,
    electron_current: np.ndarray,
    floating_potential: float,
    electron_temperature: float = None,
    ion_mass_kg: float = None,
) -> Tuple[float, float, dict]:
    """Fallback Vp estimation when all methods fail."""
    V = np.asarray(bias_voltage, dtype=float)

    if electron_temperature is not None and ion_mass_kg is not None:
        vp = _potential_method_vp(floating_potential, electron_temperature, ion_mass_kg)
        vp_unc = max(0.5 * electron_temperature, 2.0)
        method = "potential-method-fallback"
    else:
        # Last resort: floating potential + 10V offset
        vp = floating_potential + 10.0
        vp_unc = 5.0
        method = "floating-plus-offset"

    diagnostics = {
        "method": method,
        "methods_used": [method],
        "vp_candidates": np.array([vp]),
        "vp_weights": np.array([1.0]),
        "smoothed_electron_current": electron_current,
        "derivative": np.zeros_like(electron_current),
        "second_derivative": np.zeros_like(electron_current),
    }

    return float(vp), float(vp_unc), diagnostics


def _smooth_positive_series(values: np.ndarray, polyorder: int = 3) -> np.ndarray:
    """Smooth a positive series using Savitzky-Golay filter.

    Parameters
    ----------
    values : ndarray
        Values to smooth (will be clipped to ≥ 0)
    polyorder : int
        Polynomial order for savgol_filter

    Returns
    -------
    smoothed : ndarray
        Smoothed values, clipped to ≥ 0
    """
    values = np.maximum(np.asarray(values, dtype=float), 0.0)

    n_points = len(values)
    if n_points < 7:
        return values

    # Choose window length (must be odd)
    window = min(21, n_points if n_points % 2 == 1 else n_points - 1)
    if window < 5:
        return values

    try:
        smoothed = savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")
        smoothed = np.maximum(smoothed, 0.0)
        return smoothed
    except Exception:
        return values


def _median_voltage_step(bias_voltage: np.ndarray) -> float:
    """Compute median voltage step in array.

    Returns
    -------
    float
        Median absolute difference between consecutive voltages
    """
    V = np.asarray(bias_voltage, dtype=float)
    diffs = np.abs(np.diff(V))
    diffs = diffs[diffs > 1e-10]  # Filter out zero/near-zero steps

    if len(diffs) == 0:
        return 1.0

    return float(np.median(diffs))
