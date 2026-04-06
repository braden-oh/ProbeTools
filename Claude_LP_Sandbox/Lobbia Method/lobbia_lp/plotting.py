"""Plotting utilities for Langmuir probe diagnostic visualization.

Provides functions for plotting I-V characteristics, semilog fits, derivatives, and EEDF.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_iv_characteristic(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    vf: Optional[float] = None,
    vp: Optional[float] = None,
    ion_current: Optional[np.ndarray] = None,
    electron_current: Optional[np.ndarray] = None,
    title: str = "Langmuir Probe I-V Characteristic",
    figsize: tuple = (10, 6),
) -> tuple:
    """Plot I-V characteristic with identified potentials and components.

    Parameters
    ----------
    bias_voltage : ndarray
        Bias voltages [V]
    probe_current : ndarray
        Probe currents [A]
    vf : float, optional
        Floating potential [V], marked on plot
    vp : float, optional
        Plasma potential [V], marked on plot
    ion_current : ndarray, optional
        Ion current [A], plotted separately
    electron_current : ndarray, optional
        Electron current [A], plotted separately
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    V = np.asarray(bias_voltage)
    I = np.asarray(probe_current)

    ax.plot(V, I * 1000, "k-", linewidth=1.5, label="Probe current (total)")

    if ion_current is not None:
        ax.plot(V, np.asarray(ion_current) * 1000, "b--", linewidth=1, label="Ion current")

    if electron_current is not None:
        ax.plot(V, np.asarray(electron_current) * 1000, "r--", linewidth=1, label="Electron current")

    # Mark potentials
    if vf is not None:
        ax.axvline(vf, color="green", linestyle=":", alpha=0.7, label=f"Vf = {vf:.2f} V")

    if vp is not None:
        ax.axvline(vp, color="orange", linestyle=":", alpha=0.7, label=f"Vp = {vp:.2f} V")

    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Bias Voltage (V)", fontsize=11)
    ax.set_ylabel("Current (mA)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return fig, ax


def plot_semilog_fit(
    bias_voltage: np.ndarray,
    electron_current: np.ndarray,
    vf: float,
    vp: float,
    te: float,
    fit_window_mask: Optional[np.ndarray] = None,
    title: str = "Semilog Fit for Electron Temperature",
    figsize: tuple = (10, 6),
) -> tuple:
    """Plot ln(Ie) vs V with semilog fit for Te extraction.

    Parameters
    ----------
    bias_voltage : ndarray
        Bias voltages [V]
    electron_current : ndarray
        Electron currents [A]
    vf : float
        Floating potential [V]
    vp : float
        Plasma potential [V]
    te : float
        Electron temperature [eV] (used to label plot)
    fit_window_mask : ndarray, optional
        Boolean mask of points used in fit
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    V = np.asarray(bias_voltage)
    Ie = np.maximum(np.asarray(electron_current), 1e-15)

    ln_Ie = np.log(Ie)

    # Plot all data
    ax.semilogy(V, Ie, "k.", alpha=0.5, label="Measured")

    # Plot fit window if provided
    if fit_window_mask is not None:
        mask = np.asarray(fit_window_mask, dtype=bool)
        if np.any(mask):
            from scipy import stats
            p = stats.linregress(V[mask], ln_Ie[mask])
            V_fit = np.array([V[mask].min(), V[mask].max()])
            Ie_fit = np.exp(p.slope * V_fit + p.intercept)
            ax.semilogy(V_fit, Ie_fit, "r-", linewidth=2, label=f"Fit (Te = {te:.2f} eV)")

    # Mark potentials
    ax.axvline(vf, color="green", linestyle=":", alpha=0.7, label=f"Vf = {vf:.2f} V")
    ax.axvline(vp, color="orange", linestyle=":", alpha=0.7, label=f"Vp = {vp:.2f} V")

    ax.set_xlabel("Bias Voltage (V)", fontsize=11)
    ax.set_ylabel("Electron Current (A)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    return fig, ax


def plot_derivative(
    bias_voltage: np.ndarray,
    electron_current: np.ndarray,
    vp: Optional[float] = None,
    vf: Optional[float] = None,
    derivative: Optional[np.ndarray] = None,
    title: str = "Derivative dIe/dV",
    figsize: tuple = (10, 6),
) -> tuple:
    """Plot first derivative of electron current for Vp identification.

    Parameters
    ----------
    bias_voltage : ndarray
        Bias voltages [V]
    electron_current : ndarray
        Electron currents [A]
    vp : float, optional
        Plasma potential [V]
    vf : float, optional
        Floating potential [V]
    derivative : ndarray, optional
        Pre-computed derivative, otherwise computed via gradient
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    V = np.asarray(bias_voltage)
    Ie = np.asarray(electron_current)

    if derivative is None:
        # Smooth and compute derivative
        from scipy.signal import savgol_filter
        Ie_smooth = savgol_filter(Ie, min(21, len(Ie) if len(Ie) % 2 == 1 else len(Ie) - 1), 3)
        derivative = np.gradient(Ie_smooth, V)
    else:
        derivative = np.asarray(derivative)

    ax.plot(V, derivative, "b-", linewidth=1.5, label="dIe/dV")

    if vp is not None:
        ax.axvline(vp, color="orange", linestyle=":", linewidth=1.5, alpha=0.7,
                   label=f"Vp = {vp:.2f} V")

    if vf is not None:
        ax.axvline(vf, color="green", linestyle=":", linewidth=1.5, alpha=0.7,
                   label=f"Vf = {vf:.2f} V")

    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Bias Voltage (V)", fontsize=11)
    ax.set_ylabel("dIe/dV (A/V)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return fig, ax


def plot_eedf(
    energy: np.ndarray,
    eedf: np.ndarray,
    te: Optional[float] = None,
    title: str = "Electron Energy Distribution Function",
    figsize: tuple = (10, 6),
) -> tuple:
    """Plot electron energy distribution function.

    Parameters
    ----------
    energy : ndarray
        Energy array [eV]
    eedf : ndarray
        EEDF normalized distribution
    te : float, optional
        Electron temperature [eV] for reference
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    E = np.asarray(energy)
    f = np.asarray(eedf)

    ax.semilogy(E, f, "b-", linewidth=2, label="Measured EEDF")

    if te is not None:
        ax.axvline(te, color="r", linestyle="--", alpha=0.7, label=f"Te = {te:.2f} eV")

    ax.set_xlabel("Electron Energy (eV)", fontsize=11)
    ax.set_ylabel("EEDF (a.u.)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    return fig, ax


def plot_analysis_summary(
    result,
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    figsize: tuple = (14, 10),
) -> tuple:
    """Create a summary figure with multiple subplots showing the full analysis.

    Parameters
    ----------
    result : LPResult
        Analysis result object
    bias_voltage : ndarray
        Bias voltages [V]
    probe_current : ndarray
        Probe currents [A]
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # I-V characteristic
    ax1 = plt.subplot(2, 3, 1)
    V = np.asarray(bias_voltage)
    I = np.asarray(probe_current)
    ax1.plot(V, I * 1000, "k-", linewidth=1)
    ax1.axvline(result.vf, color="g", linestyle=":", alpha=0.7, label="Vf")
    ax1.axvline(result.vp, color="orange", linestyle=":", alpha=0.7, label="Vp")
    ax1.set_xlabel("Voltage (V)")
    ax1.set_ylabel("Current (mA)")
    ax1.set_title("I-V Characteristic")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Results text
    ax2 = plt.subplot(2, 3, 2)
    ax2.axis("off")
    results_text = f"""
Plasma Parameters:
Vf = {result.vf:.3f} V
Vp = {result.vp:.3f} V
Te = {result.te:.3f} eV
ne = {result.ne:.3e} m⁻³
ni = {result.ni:.3e} m⁻³
n  = {result.n:.3e} m⁻³

λD = {result.debye_length:.3e} m
rp/λD = {result.rp_lambda:.2f}
Sheath: {result.sheath_regime}

Iterations: {result.n_iterations}
Converged: {result.converged}
"""
    ax2.text(0.1, 0.5, results_text, fontsize=10, family="monospace",
             verticalalignment="center")

    # Uncertainties
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis("off")
    unc_text = f"""
Uncertainties:
ΔVf = {result.uncertainties.get('d_Vf', np.nan):.3f} V
ΔVp = {result.uncertainties.get('d_Vp', np.nan):.3f} V
ΔTe = {result.uncertainties.get('d_Te', np.nan):.3f} eV
Δne = {result.uncertainties.get('d_ne', np.nan):.3e} m⁻³
Δni = {result.uncertainties.get('d_ni', np.nan):.3e} m⁻³
Δn  = {result.uncertainties.get('d_n', np.nan):.3e} m⁻³
"""
    ax3.text(0.1, 0.5, unc_text, fontsize=10, family="monospace",
             verticalalignment="center")

    # Placeholder for other plots (would need electron current, derivative, etc.)
    ax4 = plt.subplot(2, 3, 4)
    ax4.text(0.5, 0.5, "(Additional diagnostic plots)", ha="center", va="center")
    ax4.axis("off")

    ax5 = plt.subplot(2, 3, 5)
    ax5.axis("off")

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    plt.tight_layout()
    return fig
