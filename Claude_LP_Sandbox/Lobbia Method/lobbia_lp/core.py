"""Core Langmuir probe analysis implementing the 11-step Lobbia algorithm.

Main entry point for users. Implements the complete single probe analysis
with iteration loop for sheath-corrected ion density determination.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from . import constants
from . import io
from . import potentials
from . import electron
from . import ion
from . import eedf
from . import uncertainty


@dataclass
class LPResult:
    """Result of Langmuir probe analysis."""

    # Key plasma parameters
    vf: float  # Floating potential [V]
    vp: float  # Plasma potential [V]
    te: float  # Electron temperature [eV]
    ne: float  # Electron density [m^-3]
    ni: float  # Ion density [m^-3]
    n: float  # Quasineutral density [m^-3]
    ie_sat: float  # Electron saturation current [A]
    ii_sat: float  # Ion saturation current [A]

    # Derived quantities
    debye_length: float  # Debye length [m]
    rp_lambda: float  # rp / λD ratio
    sheath_regime: str  # 'thin', 'transitional', or 'oml'

    # Uncertainties
    uncertainties: Dict[str, float]  # d_Vf, d_Vp, d_Te, d_ne, d_ni, d_n

    # EEDF (if computed)
    eedf_energy: Optional[np.ndarray] = None
    eedf: Optional[np.ndarray] = None
    ne_from_eedf: Optional[float] = None

    # Diagnostic info
    n_iterations: int = 1
    converged: bool = True
    diagnostics: Dict[str, Any] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for easy export."""
        data = {
            "Value": [
                self.vf, self.vp, self.te, self.n, self.ne, self.ni,
                self.ie_sat, self.ii_sat, self.debye_length, self.rp_lambda
            ],
            "Uncertainty": [
                self.uncertainties.get("d_Vf", np.nan),
                self.uncertainties.get("d_Vp", np.nan),
                self.uncertainties.get("d_Te", np.nan),
                self.uncertainties.get("d_n", np.nan),
                self.uncertainties.get("d_ne", np.nan),
                self.uncertainties.get("d_ni", np.nan),
                self.uncertainties.get("d_Ie_sat", np.nan),
                self.uncertainties.get("d_Ii_sat", np.nan),
                np.nan, np.nan
            ],
            "Unit": [
                "V", "V", "eV", "m^-3", "m^-3", "m^-3",
                "A", "A", "m", "1"
            ]
        }
        index = [
            "Vf", "Vp", "Te", "n", "ne", "ni",
            "Ie_sat", "Ii_sat", "lambda_D", "rp_lambda"
        ]
        return pd.DataFrame(data, index=index)


def analyze(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    probe_radius_m: float,
    probe_area_m2: float,
    probe_length_m: float,
    gas: str = "xe",
    probe_geometry: str = "cylindrical",
    max_iterations: int = 10,
    convergence_tolerance: float = 1e-3,
    compute_eedf: bool = False,
    verbose: bool = False,
) -> LPResult:
    """Analyze a single Langmuir probe I-V trace using the Lobbia method.

    Implements the 11-step algorithm from Lobbia & Beal (2017) Section II.F
    with sheath-corrected iteration loop for ion density.

    Parameters
    ----------
    bias_voltage : array-like
        Bias voltages [V]
    probe_current : array-like
        Probe currents [A]
    probe_radius_m : float
        Probe radius [m]
    probe_area_m2 : float
        Probe collection area [m²]
    probe_length_m : float
        Probe length [m] (for cylindrical probes)
    gas : str
        Gas species ('xe', 'kr', 'ar', 'n', 'zn')
    probe_geometry : str
        Probe shape ('cylindrical', 'spherical', 'planar')
    max_iterations : int
        Maximum iteration count for sheath-correction loop
    convergence_tolerance : float
        Relative tolerance for ni convergence criterion
    compute_eedf : bool
        Whether to compute EEDF using Druyvesteyn method
    verbose : bool
        Print progress information

    Returns
    -------
    LPResult
        Analysis results with all derived quantities and uncertainties

    Raises
    ------
    ValueError
        If gas species or probe geometry not supported, or insufficient data
    """
    # Step 0: Load and normalize trace
    V, I = io._normalize_trace(bias_voltage, probe_current)

    if len(V) < 10:
        raise ValueError("Insufficient data points (need at least 10)")

    mi_kg = constants.ion_mass_kg(gas)

    # Step 1: Compute log scale for plotting (will be used in potentials)
    # (Not explicitly needed, but good for diagnostics)

    # Step 2: Find floating potential
    vf, d_vf = potentials.find_floating_potential(V, I)
    if verbose:
        print(f"Step 2: Vf = {vf:.3f} V ± {d_vf:.3f} V")

    # Step 3: Preliminary ion current fit (constant with V < Vf)
    ion_mask = V <= vf
    if np.sum(ion_mask) > 1:
        from scipy import stats as sp_stats
        p = sp_stats.linregress(V[ion_mask], I[ion_mask])
        Ii_prelim = np.polyval([p.slope, p.intercept], V)
    else:
        Ii_prelim = np.zeros_like(V)
    Ii_prelim = np.minimum(Ii_prelim, 0.0)  # Ensure non-positive

    # Step 4a: Compute electron current
    Ie = I - Ii_prelim
    Ie = np.maximum(Ie, 1e-15)

    # Step 5: Find plasma potential
    vp, d_vp, vp_diag = potentials.find_plasma_potential(
        V, I, vf, ion_current=Ii_prelim
    )
    if verbose:
        print(f"Step 5: Vp = {vp:.3f} V ± {d_vp:.3f} V")

    # Step 6: Fit electron temperature
    te, d_te, te_fit_info = electron.fit_electron_temperature(
        V, Ie, vf, vp
    )
    if verbose:
        print(f"Step 6: Te = {te:.3f} eV ± {d_te:.3f} eV")

    # Step 7: Compute electron density
    ie_sat = np.interp(vp, V, Ie)
    ne, d_ne = electron.compute_electron_density(ie_sat, te, probe_area_m2)
    if verbose:
        print(f"Step 7: ne = {ne:.3e} m^-3 ± {d_ne:.3e} m^-3")

    # Step 8-11: Iteration loop with sheath-corrected ion current
    ni = np.nan
    ni_prev = 0.0
    ii_sat = np.interp(np.min(V), V, I)
    sheath_regime = "unknown"

    for iteration in range(max_iterations):
        # Steps 8-11: Iteration loop refines ion density until convergence
        # Note: Te and ne are fixed from the initial semilog fit (Steps 6-7)
        # Only the ion current model and ni are refined each iteration

        # Steps 8-10: Compute sheath-corrected ion density
        lambda_d = constants.debye_length(ne, te)
        rp_lambda = probe_radius_m / lambda_d if lambda_d > 0 else np.inf

        ni, d_ni, sheath_regime = ion.compute_ion_density(
            V, I, vf, vp, te, probe_geometry, rp_lambda,
            probe_radius_m, probe_area_m2, mi_kg
        )

        if verbose:
            print(f"  Iteration {iteration}: ni = {ni:.3e} m^-3, "
                  f"rp/λD = {rp_lambda:.1f}, regime = {sheath_regime}")

        # Check convergence
        if np.isfinite(ni) and np.isfinite(ni_prev) and ni_prev > 0:
            rel_change = abs(ni - ni_prev) / ni_prev
            if rel_change < convergence_tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

        ni_prev = ni if np.isfinite(ni) else 0.0

    # Step 11: Final quasineutral density
    if np.isfinite(ne) and np.isfinite(ni):
        n = np.mean([ne, ni])
    elif np.isfinite(ne):
        n = ne
    elif np.isfinite(ni):
        n = ni
    else:
        n = np.nan

    # Compute Debye length
    lambda_d = constants.debye_length(ne, te)
    rp_lambda = probe_radius_m / lambda_d if lambda_d > 0 else np.inf

    # Propagate uncertainties
    unc = uncertainty.propagate_uncertainties(
        vf, vp, te, ne, ni, ie_sat, ii_sat,
        probe_area_m2, probe_radius_m, probe_length_m
    )

    # Optional: Compute EEDF
    eedf_energy = None
    eedf_vals = None
    ne_from_eedf = None
    if compute_eedf:
        try:
            eedf_energy, eedf_vals, ne_from_eedf, eedf_diag = eedf.compute_eedf_druyvesteyn(
                V, Ie, vp, ion_current=Ii_model if iteration > 0 else Ii_prelim
            )
        except Exception as e:
            if verbose:
                print(f"EEDF computation failed: {e}")

    diagnostics = {
        "vf_method": "sign-change",
        "vp_method": vp_diag.get("method", "unknown"),
        "te_method": te_fit_info.get("method", "unknown"),
        "sheath_regime": sheath_regime,
        "n_iterations": iteration + 1,
        "converged": abs(ni - ni_prev) / max(ni, 1e-20) < convergence_tolerance if np.isfinite(ni) else False,
    }

    return LPResult(
        vf=vf,
        vp=vp,
        te=te,
        ne=ne,
        ni=ni,
        n=n,
        ie_sat=ie_sat,
        ii_sat=ii_sat,
        debye_length=lambda_d,
        rp_lambda=rp_lambda,
        sheath_regime=sheath_regime,
        uncertainties=unc,
        eedf_energy=eedf_energy,
        eedf=eedf_vals,
        ne_from_eedf=ne_from_eedf,
        n_iterations=iteration + 1,
        converged=diagnostics["converged"],
        diagnostics=diagnostics,
    )
