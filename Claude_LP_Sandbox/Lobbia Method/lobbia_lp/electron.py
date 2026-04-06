"""Electron temperature and density extraction (Lobbia Steps 5–7).

Implements semilog slope method for Te and saturation current method for ne.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats
from .constants import QE, ME


def fit_electron_temperature(
    bias_voltage: np.ndarray,
    electron_current: np.ndarray,
    floating_potential: float,
    plasma_potential: float,
    electron_temperature_initial: Optional[float] = None,
) -> Tuple[float, float, dict]:
    """Extract electron temperature using semilog slope method (Lobbia Steps 5–6).

    Fits ln(Ie) vs VB in the region [Vf, Vp - 2*Te], following Eq. 18.
    The slope of the linear fit is 1/Te.

    Parameters
    ----------
    bias_voltage : ndarray
        Bias voltages [V]
    electron_current : ndarray
        Electron currents [A], should be ≥ 0
    floating_potential : float
        Floating potential [V]
    plasma_potential : float
        Plasma potential [V]
    electron_temperature_initial : float, optional
        Initial Te estimate [eV] for refined window selection.
        If None, uses simple window [Vf, Vp - 1V].

    Returns
    -------
    te : float
        Electron temperature [eV]
    te_uncertainty : float
        Uncertainty in Te [eV]
    fit_info : dict
        Fit diagnostic info (slope, intercept, r²,  window mask, etc.)
    """
    V = np.asarray(bias_voltage, dtype=float)
    Ie = np.maximum(np.asarray(electron_current, dtype=float), 1e-15)

    # Define fitting window: [Vf, Vp - 2*Te] (Eq. 18)
    # Start with initial estimate
    if electron_temperature_initial is not None and electron_temperature_initial > 0:
        vp_upper = plasma_potential - 2.0 * electron_temperature_initial
    else:
        vp_upper = plasma_potential - 1.0

    vp_upper = max(vp_upper, floating_potential + 1.0)

    fit_mask = (V >= floating_potential) & (V <= vp_upper) & (Ie > 1e-15)

    if np.sum(fit_mask) < 5:
        # Not enough points; use broader window
        fit_mask = (V >= floating_potential) & (V < plasma_potential) & (Ie > 1e-15)

    if np.sum(fit_mask) < 3:
        # Fall back to potential method
        te, te_unc = _te_from_potentials(floating_potential, plasma_potential)
        return te, te_unc, {
            "method": "potential-method",
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "fit_window_mask": fit_mask,
        }

    V_fit = V[fit_mask]
    Ie_fit = Ie[fit_mask]

    # Fit ln(Ie) vs V
    ln_Ie_fit = np.log(Ie_fit)

    result = stats.linregress(V_fit, ln_Ie_fit)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue**2) if np.isfinite(result.rvalue) else np.nan

    if slope <= 0.0:
        # Invalid fit; fall back to potential method
        te, te_unc = _te_from_potentials(floating_potential, plasma_potential)
        return te, te_unc, {
            "method": "potential-method-fallback-invalid-slope",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "fit_window_mask": fit_mask,
        }

    te = 1.0 / slope
    te_unc = abs(float(result.stderr) / slope**2) if result.stderr else 0.1 * te

    fit_info = {
        "method": "semilog-slope",
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "slope_stderr": float(result.stderr) if result.stderr else np.nan,
        "fit_window_mask": fit_mask,
        "fit_voltage_range": (float(V_fit[0]), float(V_fit[-1])),
    }

    return float(te), float(max(te_unc, 0.05)), fit_info


def compute_electron_density(
    electron_sat_current: float,
    electron_temperature: float,
    probe_area_m2: float,
) -> Tuple[float, float]:
    """Compute electron density from saturation current (Lobbia Step 7).

    ne = Ie,sat / (e * Ap * sqrt(e*Te / (2π*me)))

    This is Eq. (3) rearranged for ne.

    Parameters
    ----------
    electron_sat_current : float
        Electron saturation current [A]
    electron_temperature : float
        Electron temperature [eV]
    probe_area_m2 : float
        Probe collection area [m²]

    Returns
    -------
    ne : float
        Electron density [m^-3]
    ne_uncertainty : float
        Uncertainty in ne [m^-3]
    """
    if electron_temperature <= 0.0 or probe_area_m2 <= 0.0:
        return np.nan, np.nan

    # Denominator: e * Ap * sqrt(e*Te / (2π*me))
    # Note: electron_temperature is in eV, so multiply by QE to convert to Joules in the sqrt
    denominator = QE * probe_area_m2 * np.sqrt(electron_temperature * QE / (2.0 * np.pi * ME))

    ne = electron_sat_current / denominator

    # Uncertainty: assume 5% current uncertainty, 10% Te uncertainty
    d_ne_from_current = 0.05 * ne
    d_ne_from_te = 0.5 * 0.10 * ne  # dn/dte ∝ Te^-0.5
    ne_unc = np.sqrt(d_ne_from_current**2 + d_ne_from_te**2)

    return float(ne), float(ne_unc)


def _te_from_potentials(
    floating_potential: float,
    plasma_potential: float,
    ion_mass_kg: float = 131.293 * 1.66053906660e-27,  # Xenon default
) -> Tuple[float, float]:
    """Estimate Te from Vp and Vf using the potential method (Lobbia Eq. in Step 6 text).

    Te ≈ (Vp - Vf) / ln(sqrt(mi / (2π*me)))

    Parameters
    ----------
    floating_potential : float
        Floating potential [V]
    plasma_potential : float
        Plasma potential [V]
    ion_mass_kg : float
        Ion mass [kg]

    Returns
    -------
    te : float
        Electron temperature [eV]
    te_uncertainty : float
        Uncertainty estimate [eV]
    """
    ratio_term = np.log(np.sqrt(ion_mass_kg / (2.0 * np.pi * ME)))
    denom = max(ratio_term, 1e-6)

    te = (plasma_potential - floating_potential) / denom
    te = max(te, 0.1)

    # Uncertainty from potential difference uncertainties
    te_unc = max(0.1 * te, 0.2)

    return float(te), float(te_unc)
