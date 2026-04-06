"""Ion current collection models (Lobbia Steps 3, 8–11).

Implements thin-sheath (Child-Langmuir), OML, and transitional sheath models
for ion current collection to Langmuir probes.
"""

from typing import Tuple, Dict, Optional
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from .constants import QE, ME, debye_length


def select_sheath_regime(
    rp_over_lambda_d: float,
) -> str:
    """Select ion current collection model based on rp/λD ratio.

    Parameters
    ----------
    rp_over_lambda_d : float
        Probe radius to Debye length ratio

    Returns
    -------
    str
        Sheath regime: 'thin', 'transitional', or 'oml'
    """
    if rp_over_lambda_d >= 50.0:
        return "thin"
    elif rp_over_lambda_d <= 3.0:
        return "oml"
    else:
        return "transitional"


def model_ion_current(
    bias_voltage: np.ndarray,
    plasma_potential: float,
    electron_temperature: float,
    probe_geometry: str,
    rp_over_lambda_d: float,
    probe_radius_m: float,
    probe_area_m2: float,
    ion_density: float,
    ion_mass_kg: float,
) -> np.ndarray:
    """Compute ion current model across full V range.

    Parameters
    ----------
    bias_voltage : ndarray
        Bias voltages [V]
    plasma_potential : float
        Plasma potential [V]
    electron_temperature : float
        Electron temperature [eV]
    probe_geometry : str
        'cylindrical', 'spherical', or 'planar'
    rp_over_lambda_d : float
        Probe radius to Debye length ratio
    probe_radius_m : float
        Probe radius [m]
    probe_area_m2 : float
        Probe collection area [m²]
    ion_density : float
        Ion density [m^-3]
    ion_mass_kg : float
        Ion mass [kg]

    Returns
    -------
    ndarray
        Ion current [A] at each bias voltage
    """
    regime = select_sheath_regime(rp_over_lambda_d)

    if regime == "thin":
        return _ion_current_thin_sheath(
            bias_voltage, plasma_potential, electron_temperature,
            probe_geometry, probe_radius_m, probe_area_m2,
            ion_density, ion_mass_kg
        )
    elif regime == "oml":
        return _ion_current_oml(
            bias_voltage, plasma_potential, electron_temperature,
            probe_geometry, probe_area_m2, ion_density, ion_mass_kg
        )
    else:  # transitional
        return _ion_current_transitional(
            bias_voltage, plasma_potential, electron_temperature,
            probe_geometry, rp_over_lambda_d, probe_area_m2,
            ion_density, ion_mass_kg
        )


def compute_ion_density(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    floating_potential: float,
    plasma_potential: float,
    electron_temperature: float,
    probe_geometry: str,
    rp_over_lambda_d: float,
    probe_radius_m: float,
    probe_area_m2: float,
    ion_mass_kg: float,
) -> Tuple[float, float, str]:
    """Compute ion density from ion saturation current using sheath-corrected model.

    Implements Lobbia Steps 8–11 (sheath correction and iteration).

    Parameters
    ----------
    bias_voltage : ndarray
        Bias voltages [V]
    probe_current : ndarray
        Probe currents [A]
    floating_potential : float
        Floating potential [V]
    plasma_potential : float
        Plasma potential [V]
    electron_temperature : float
        Electron temperature [eV]
    probe_geometry : str
        'cylindrical', 'spherical', or 'planar'
    rp_over_lambda_d : float
        Probe radius to Debye length ratio
    probe_radius_m : float
        Probe radius [m]
    probe_area_m2 : float
        Probe collection area [m²]
    ion_mass_kg : float
        Ion mass [kg]

    Returns
    -------
    ni : float
        Ion density [m^-3]
    ni_uncertainty : float
        Uncertainty estimate [m^-3]
    sheath_regime : str
        Applied sheath model ('thin', 'transitional', or 'oml')
    """
    V = np.asarray(bias_voltage, dtype=float)
    I = np.asarray(probe_current, dtype=float)

    regime = select_sheath_regime(rp_over_lambda_d)

    # Use data from negative biases (ion saturation region)
    ion_mask = V <= floating_potential
    if np.sum(ion_mask) < 3:
        return np.nan, np.nan, regime

    V_ion = V[ion_mask]
    I_ion = I[ion_mask]

    if regime == "thin":
        ni, ni_unc = _ion_density_thin_sheath(
            I_ion, electron_temperature, probe_area_m2, ion_mass_kg
        )
    elif regime == "oml":
        ni, ni_unc = _ion_density_oml(
            V_ion, I_ion, probe_area_m2, probe_geometry,
            electron_temperature, ion_mass_kg
        )
    else:  # transitional
        ni, ni_unc = _ion_density_transitional(
            V_ion, I_ion, probe_area_m2, probe_geometry,
            electron_temperature, ion_mass_kg, rp_over_lambda_d
        )

    return float(ni), float(ni_unc), regime


def _ion_current_thin_sheath(
    bias_voltage: np.ndarray,
    plasma_potential: float,
    electron_temperature: float,
    probe_geometry: str,
    probe_radius_m: float,
    probe_area_m2: float,
    ion_density: float,
    ion_mass_kg: float,
) -> np.ndarray:
    """Compute ion current using thin-sheath (Child-Langmuir) model (Lobbia Eq. 4–5)."""
    V = np.asarray(bias_voltage, dtype=float)
    delta_V = np.clip(plasma_potential - V, 0.0, None)

    # Sheath thickness from Child-Langmuir (Eq. 5)
    lambda_d = debye_length(ion_density, electron_temperature)
    if not np.isfinite(lambda_d) or lambda_d <= 0:
        return -np.exp(-0.5) * np.ones_like(V) * QE * ion_density * probe_area_m2 * \
               np.sqrt(QE * electron_temperature * QE / ion_mass_kg)

    xs = lambda_d * np.sqrt(2.0) / 3.0 * np.power(
        np.clip(2.0 * delta_V / electron_temperature, 0.0, None),
        0.75
    )

    # Sheath area (Eq. 5)
    geometry = probe_geometry.strip().lower()
    if geometry == "spherical":
        As = probe_area_m2 * np.power(1.0 + xs / probe_radius_m, 2.0)
    elif geometry == "planar":
        As = np.full_like(V, probe_area_m2, dtype=float)
    else:  # cylindrical
        As = probe_area_m2 * (1.0 + xs / probe_radius_m)

    # Ion saturation current (Eq. 4)
    bohm_speed = np.sqrt(QE * electron_temperature * QE / ion_mass_kg)
    Ii = -np.exp(-0.5) * QE * ion_density * As * bohm_speed

    return Ii


def _ion_current_oml(
    bias_voltage: np.ndarray,
    plasma_potential: float,
    electron_temperature: float,
    probe_geometry: str,
    probe_area_m2: float,
    ion_density: float,
    ion_mass_kg: float,
) -> np.ndarray:
    """Compute ion current using OML (orbital motion limited) model (Lobbia Eq. 6)."""
    V = np.asarray(bias_voltage, dtype=float)
    delta_V = np.clip(plasma_potential - V, 0.0, None)

    geometry = probe_geometry.strip().lower()

    if geometry == "cylindrical":
        # Cylindrical OML: Ii ∝ sqrt(Vp - VB)
        Ii = -QE * ion_density * probe_area_m2 * np.pi * np.sqrt(
            2.0 * delta_V * QE / ion_mass_kg
        )
    else:
        # Spherical and planar OML
        Ii = -QE * ion_density * probe_area_m2 * np.sqrt(
            QE * electron_temperature * QE / (2.0 * np.pi * ion_mass_kg)
        ) * np.sqrt(delta_V / electron_temperature)

    return Ii


def _ion_current_transitional(
    bias_voltage: np.ndarray,
    plasma_potential: float,
    electron_temperature: float,
    probe_geometry: str,
    rp_over_lambda_d: float,
    probe_area_m2: float,
    ion_density: float,
    ion_mass_kg: float,
) -> np.ndarray:
    """Compute ion current using transitional sheath model (Lobbia Eq. 7–8)."""
    V = np.asarray(bias_voltage, dtype=float)
    delta_V = np.clip(plasma_potential - V, 0.0, None)
    normalized_delta = delta_V / electron_temperature

    a, b = _transitional_sheath_coefficients(probe_geometry, rp_over_lambda_d)

    bohm_speed = np.sqrt(QE * electron_temperature * QE / ion_mass_kg)
    Ii = -QE * ion_density * probe_area_m2 * bohm_speed * a * np.power(normalized_delta, b)

    return Ii


def _ion_density_thin_sheath(
    ion_current: np.ndarray,
    electron_temperature: float,
    probe_area_m2: float,
    ion_mass_kg: float,
) -> Tuple[float, float]:
    """Extract ion density from thin-sheath formula (Lobbia Eq. 4 rearranged)."""
    # Ii = -exp(-0.5) * e * ni * As * v_B
    # ni = -Ii / (exp(-0.5) * e * As * v_B)

    bohm_speed = np.sqrt(QE * electron_temperature * QE / ion_mass_kg)

    # Use first (most negative) ion current as saturation
    Ii_sat = ion_current[0]

    ni = -Ii_sat / (np.exp(-0.5) * QE * probe_area_m2 * bohm_speed)
    ni_unc = 0.1 * ni  # ~10% uncertainty

    return float(max(ni, 0.0)), float(ni_unc)


def _ion_density_oml(
    bias_voltage: np.ndarray,
    ion_current: np.ndarray,
    probe_area_m2: float,
    probe_geometry: str,
    electron_temperature: float,
    ion_mass_kg: float,
) -> Tuple[float, float]:
    """Extract ion density from OML formula (Lobbia Eq. 6 rearranged, via Eq. 11)."""
    V = np.asarray(bias_voltage, dtype=float)
    I = np.asarray(ion_current, dtype=float)

    geometry = probe_geometry.strip().lower()

    if geometry == "cylindrical":
        a = 2.0 / np.sqrt(np.pi)
        b = 0.5
    else:
        a = 1.0
        b = 1.0

    # Fit I^(1/b) vs V to extract density
    I_transformed = np.power(np.clip(-I, 1e-18, None), 1.0 / b)

    if len(V) < 2:
        return np.nan, np.nan

    result = stats.linregress(V, I_transformed)
    slope = float(result.slope)

    # Use absolute value of slope (may be negative due to voltage sweep direction)
    slope_term = max(abs(slope), 1e-18)

    bohm_speed = np.sqrt(QE * electron_temperature * QE / ion_mass_kg)

    ni = (
        1.0 / (a * probe_area_m2)
        * np.sqrt(2.0 * np.pi * ion_mass_kg)
        / (QE ** 1.5)
        * (electron_temperature * QE) ** (b - 0.5)
        * slope_term**b
    )

    ni_unc = 0.15 * ni  # ~15% uncertainty

    return float(max(ni, 0.0)), float(ni_unc)


def _ion_density_transitional(
    bias_voltage: np.ndarray,
    ion_current: np.ndarray,
    probe_area_m2: float,
    probe_geometry: str,
    electron_temperature: float,
    ion_mass_kg: float,
    rp_over_lambda_d: float,
) -> Tuple[float, float]:
    """Extract ion density from transitional sheath formula (Lobbia Eq. 11)."""
    V = np.asarray(bias_voltage, dtype=float)
    I = np.asarray(ion_current, dtype=float)

    a, b = _transitional_sheath_coefficients(probe_geometry, rp_over_lambda_d)

    # Fit (-I)^(1/b) vs V
    I_transformed = np.power(np.clip(-I, 1e-18, None), 1.0 / b)

    if len(V) < 2:
        return np.nan, np.nan

    result = stats.linregress(V, I_transformed)
    slope = float(result.slope)

    # Use absolute value of slope (may be negative due to voltage sweep direction)
    slope_term = max(abs(slope), 1e-18)

    ni = (
        1.0 / (a * probe_area_m2)
        * np.sqrt(2.0 * np.pi * ion_mass_kg)
        / (QE ** 1.5)
        * (electron_temperature * QE) ** (b - 0.5)
        * slope_term**b
    )

    ni_unc = 0.20 * ni  # ~20% uncertainty

    return float(max(ni, 0.0)), float(ni_unc)


def _transitional_sheath_coefficients(
    probe_geometry: str, rp_over_lambda_d: float
) -> Tuple[float, float]:
    """Get transitional sheath model coefficients (a, b) from Lobbia Eq. 7–8.

    Parameters
    ----------
    probe_geometry : str
        'cylindrical', 'spherical', or 'planar'
    rp_over_lambda_d : float
        Probe radius to Debye length ratio

    Returns
    -------
    a, b : float
        Model coefficients for Ii = e*no*A*v_B * a * (V/Te)^b
    """
    ratio = float(rp_over_lambda_d)
    geometry = probe_geometry.strip().lower()

    if geometry == "planar":
        # Sheridan (Lobbia Eq. 8)
        a = 3.47 * ratio**(-0.749)
        b = 0.806 * ratio**(-0.0692)
    elif geometry == "spherical":
        # Narasimhan (Lobbia Eq. 7)
        a = 1.58 + (-0.056 + 0.816 * ratio)**(-0.744)
        b = -0.933 + (0.0148 + 0.119 * ratio)**(-0.125)
    else:  # cylindrical
        # Narasimhan (Lobbia Eq. 7)
        a = 1.18 - 0.00080 * ratio**1.35
        b = 0.0684 + (0.722 + 0.928 * ratio)**(-0.729)

    return float(a), float(b)
