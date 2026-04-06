"""Error propagation for Langmuir probe quantities (Lobbia Section V, Eq. 32).

Implements uncertainty quantification for derived plasma parameters.
"""

from typing import Dict
import numpy as np


def propagate_uncertainties(
    vf: float,
    vp: float,
    te: float,
    ne: float,
    ni: float,
    ie_sat: float,
    ii_sat: float,
    probe_area_m2: float,
    probe_radius_m: float,
    probe_length_m: float,
    d_vb: float = 0.1,
    d_ie_sat_frac: float = 0.01,
    d_ii_sat_frac: float = 0.10,
    d_rp_frac: float = 0.05,
    d_L_frac: float = 0.05,
) -> Dict[str, float]:
    """Compute uncertainties in Langmuir probe derived quantities (Lobbia Eq. 32).

    All uncertainties are computed via error propagation formulas in Eq. 32.

    Parameters
    ----------
    vf : float
        Floating potential [V]
    vp : float
        Plasma potential [V]
    te : float
        Electron temperature [eV]
    ne : float
        Electron density [m^-3]
    ni : float
        Ion density [m^-3]
    ie_sat : float
        Electron saturation current [A]
    ii_sat : float
        Ion saturation current [A]
    probe_area_m2 : float
        Probe collection area [m²]
    probe_radius_m : float
        Probe radius [m]
    probe_length_m : float
        Probe length [m]
    d_vb : float
        Voltage discretization uncertainty [V]
    d_ie_sat_frac : float
        Fractional uncertainty in Ie,sat (default 1%)
    d_ii_sat_frac : float
        Fractional uncertainty in Ii,sat (default 10%)
    d_rp_frac : float
        Fractional uncertainty in probe radius (default 5%)
    d_L_frac : float
        Fractional uncertainty in probe length (default 5%)

    Returns
    -------
    uncertainties : dict
        Dictionary with keys: 'd_Vf', 'd_Vp', 'd_Te', 'd_ne', 'd_ni', 'd_n'
        All values in same units as input quantities
    """
    # Input uncertainties
    d_Ie_sat = abs(d_ie_sat_frac * ie_sat)
    d_Ii_sat = abs(d_ii_sat_frac * ii_sat)
    d_rp = abs(d_rp_frac * probe_radius_m)
    d_L = abs(d_L_frac * probe_length_m)

    # Compute probe area uncertainty
    # Ap = π * L * (2*rp + L/4) for cylindrical
    # d_Ap / Ap ≈ sqrt((dL/L)² + (drp/(2rp))²)
    if probe_area_m2 > 0:
        d_Ap_frac = np.sqrt((d_L_frac)**2 + (d_rp_frac / 2.0)**2)
        d_Ap = d_Ap_frac * probe_area_m2
    else:
        d_Ap = 0.0
        d_Ap_frac = 0.0

    # Floating potential uncertainty (Eq. 32)
    d_Vf = np.sqrt(d_vb**2 + (d_Ii_sat / abs(ii_sat))**2 * te**2) if ii_sat != 0 else d_vb

    # Plasma potential uncertainty (Eq. 32)
    d_Vp = np.sqrt(d_vb**2 + d_vb**2) if d_vb > 0 else 0.1  # Simplified

    # Electron temperature uncertainty (Eq. 32)
    if te > 0:
        d_Te_frac = np.sqrt(
            2.0 * (d_vb / abs(vp - vf))**2 +
            2.0 * (d_Ie_sat / abs(ie_sat) * te / abs(vp - vf))**2
        )
        d_Te = d_Te_frac * te
    else:
        d_Te = 0.1

    # Electron density uncertainty (Eq. 32)
    if ne > 0 and ie_sat > 0 and te > 0:
        d_ne_frac = np.sqrt(
            (d_Ie_sat / ie_sat)**2 +
            (d_Ap_frac)**2 +
            (0.5 * d_vb / abs(vp - vf))**2
        )
        d_ne = d_ne_frac * ne
    else:
        d_ne = 0.1 * ne if ne > 0 else 1e15

    # Ion density uncertainty (Eq. 32)
    if ni > 0 and ii_sat > 0 and te > 0:
        d_ni_frac = np.sqrt(
            (d_Ii_sat / abs(ii_sat))**2 +
            (d_Ap_frac)**2 +
            (0.5 * d_vb / abs(vp - vf))**2
        )
        d_ni = d_ni_frac * ni
    else:
        d_ni = 0.5 * ni if ni > 0 else 1e15

    # Quasineutral density uncertainty (average of ne and ni with uncertainty)
    if np.isfinite(ne) and np.isfinite(ni):
        n = np.mean([ne, ni])
        if n > 0:
            # Combine uncertainties from both densities
            d_n = np.sqrt(d_ne**2 + d_ni**2) / 2.0
        else:
            d_n = 1e15
    else:
        n = ne if np.isfinite(ne) else ni
        d_n = d_ne if np.isfinite(ne) else d_ni

    uncertainties = {
        "d_Vf": float(d_Vf),
        "d_Vp": float(d_Vp),
        "d_Te": float(d_Te),
        "d_ne": float(d_ne),
        "d_ni": float(d_ni),
        "d_n": float(d_n),
        "d_Ie_sat": float(d_Ie_sat),
        "d_Ii_sat": float(d_Ii_sat),
    }

    return uncertainties


def compute_relative_uncertainties(
    absolute_uncertainties: Dict[str, float],
    values: Dict[str, float],
) -> Dict[str, float]:
    """Compute relative uncertainties from absolute values.

    Parameters
    ----------
    absolute_uncertainties : dict
        Dictionary with absolute uncertainty values
    values : dict
        Dictionary with quantity values

    Returns
    -------
    relative_uncertainties : dict
        Dictionary with relative uncertainties (as fractions, multiply by 100 for %)
    """
    relative = {}

    for key in ["Vf", "Vp", "Te", "ne", "ni", "n", "Ie_sat", "Ii_sat"]:
        abs_key = f"d_{key}"
        if abs_key in absolute_uncertainties and key in values:
            val = values[key]
            if val != 0:
                relative[f"r_{key}"] = abs(absolute_uncertainties[abs_key] / val)
            else:
                relative[f"r_{key}"] = np.nan
        else:
            relative[f"r_{key}"] = np.nan

    return relative
