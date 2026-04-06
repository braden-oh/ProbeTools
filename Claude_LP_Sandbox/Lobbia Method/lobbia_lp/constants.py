"""Physical constants and gas mass table for Langmuir probe analysis.

References:
- CODATA 2018 fundamental constants
- Lobbia & Beal (2017) Recommended Practice for Use of Langmuir Probes in EP Testing
"""

import numpy as np

# Fundamental constants
QE = 1.602176634e-19  # Elementary charge [C]
ME = 9.1093837015e-31  # Electron mass [kg]
MI_E = 1836.15267  # Ion mass to electron mass ratio (ratio of proton to electron mass)
AMU_TO_KG = 1.66053906660e-27  # Atomic mass unit to kg
EPS0 = 8.8541878128e-12  # Permittivity of free space [F/m]
KB = 1.380649e-23  # Boltzmann constant [J/K]

# Conversion constants
EV_TO_J = QE  # 1 eV in joules

# Gas ion masses (in AMU)
GAS_MASSES_AMU = {
    "xe": 131.293,  # Xenon
    "kr": 83.798,   # Krypton
    "ar": 39.948,   # Argon
    "n": 14.007,    # Nitrogen
    "zn": 65.38,    # Zinc
}


def ion_mass_kg(gas: str) -> float:
    """Get ion mass in kg for a given gas species.

    Parameters
    ----------
    gas : str
        Gas species ('xe', 'kr', 'ar', 'n', 'zn')

    Returns
    -------
    float
        Ion mass in kg
    """
    key = gas.strip().lower()
    if key not in GAS_MASSES_AMU:
        raise ValueError(f"Unknown gas '{gas}'. Supported: {list(GAS_MASSES_AMU.keys())}")
    return GAS_MASSES_AMU[key] * AMU_TO_KG


def debye_length(electron_density: float, electron_temperature: float) -> float:
    """Compute electron Debye length.

    λ_D = sqrt(ε₀ * Te / (ne * e))

    Parameters
    ----------
    electron_density : float
        Electron density [m^-3]
    electron_temperature : float
        Electron temperature [eV]

    Returns
    -------
    float
        Debye length [m]
    """
    if electron_density <= 0.0 or electron_temperature <= 0.0:
        return np.nan

    # λ_D = sqrt(ε₀ * Te / (ne * e)) where Te is in eV, denominator is ne*e
    # The formula avoids double-conversion: Te[eV] / (ne * e) rather than Te[J] / (ne * e)
    return np.sqrt(EPS0 * electron_temperature / (electron_density * QE))


def bohm_speed(electron_temperature: float, ion_mass_kg: float) -> float:
    """Compute Bohm ion speed.

    v_B = sqrt(e*Te / mi)

    Parameters
    ----------
    electron_temperature : float
        Electron temperature [eV]
    ion_mass_kg : float
        Ion mass [kg]

    Returns
    -------
    float
        Bohm speed [m/s]
    """
    if electron_temperature <= 0.0 or ion_mass_kg <= 0.0:
        return np.nan

    # Te[eV] * e = energy in Joules
    return np.sqrt(electron_temperature * QE / ion_mass_kg)


def electron_thermal_velocity(electron_temperature: float) -> float:
    """Compute mean electron thermal velocity (Maxwellian).

    v_e = sqrt(8*e*Te / (π*me))

    Parameters
    ----------
    electron_temperature : float
        Electron temperature [eV]

    Returns
    -------
    float
        Mean electron thermal velocity [m/s]
    """
    if electron_temperature <= 0.0:
        return np.nan

    # Te[eV] * e = energy in Joules
    return np.sqrt(8.0 * electron_temperature * QE / (np.pi * ME))
