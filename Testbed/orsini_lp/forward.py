from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma

from .schema import TraceData

E_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31


def eedf_shape_coefficients(p_value: float) -> tuple[float, float]:
    if p_value <= 0:
        raise ValueError("The EEDF shape exponent p must be positive.")
    gamma_3 = gamma(3.0 / (2.0 * p_value))
    gamma_5 = gamma(5.0 / (2.0 * p_value))
    b_coeff = (2.0 * gamma_5 / (3.0 * gamma_3)) ** p_value
    a_coeff = p_value * (b_coeff ** (3.0 / (2.0 * p_value))) / gamma_3
    return a_coeff, b_coeff


def normalized_eedf(energy_ev: np.ndarray, te_ev: float, p_value: float) -> np.ndarray:
    energy = np.asarray(energy_ev, dtype=float)
    if te_ev <= 0:
        raise ValueError("Electron temperature must be positive.")
    a_coeff, b_coeff = eedf_shape_coefficients(p_value)
    return (a_coeff / (te_ev ** 1.5)) * np.sqrt(np.clip(energy, 0.0, None)) * np.exp(
        -b_coeff * np.power(np.clip(energy, 0.0, None) / te_ev, p_value)
    )


def electron_current(
    bias_voltage: np.ndarray,
    probe_area_m2: float,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    p_value: float,
    quadrature_nodes: int = 512,
    quadrature_xmax: float = 18.0,
) -> np.ndarray:
    bias = np.asarray(bias_voltage, dtype=float)
    if electron_density_m3 <= 0 or electron_temperature_ev <= 0:
        return np.full_like(bias, np.nan, dtype=float)

    a_coeff, b_coeff = eedf_shape_coefficients(p_value)
    x_grid = np.linspace(0.0, quadrature_xmax, quadrature_nodes)
    u = np.maximum((plasma_potential_v - bias) / electron_temperature_ev, 0.0)
    kernel = np.clip(x_grid[None, :] - u[:, None], 0.0, None) * np.exp(
        -b_coeff * np.power(x_grid[None, :], p_value)
    )
    integral = np.trapz(kernel, x_grid, axis=1)
    prefactor = (
        probe_area_m2
        * electron_density_m3
        * math.sqrt((E_CHARGE ** 3) / (8.0 * ELECTRON_MASS))
        * a_coeff
        * math.sqrt(electron_temperature_ev)
    )
    return prefactor * integral


def total_probe_current(
    bias_voltage: np.ndarray,
    probe_area_m2: float,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    p_value: float,
    ion_slope_a_per_v: float,
    ion_intercept_a: float,
    quadrature_nodes: int = 512,
    quadrature_xmax: float = 18.0,
) -> np.ndarray:
    electron = electron_current(
        bias_voltage=bias_voltage,
        probe_area_m2=probe_area_m2,
        plasma_potential_v=plasma_potential_v,
        electron_density_m3=electron_density_m3,
        electron_temperature_ev=electron_temperature_ev,
        p_value=p_value,
        quadrature_nodes=quadrature_nodes,
        quadrature_xmax=quadrature_xmax,
    )
    ion_linear = ion_slope_a_per_v * (plasma_potential_v - np.asarray(bias_voltage, dtype=float)) + ion_intercept_a
    return electron + ion_linear


def default_energy_grid(samples_te_ev: np.ndarray, points: int = 250) -> np.ndarray:
    te_values = np.asarray(samples_te_ev, dtype=float)
    upper = max(8.0, 8.0 * float(np.percentile(te_values, 95)))
    return np.linspace(0.0, upper, points)


def density_weighted_eedf_samples(energy_grid_ev: np.ndarray, posterior_samples: np.ndarray) -> np.ndarray:
    eedf_samples = []
    for vp, ne, te, p_value, *_rest in posterior_samples:
        _ = vp
        eedf_samples.append(ne * normalized_eedf(energy_grid_ev, te, p_value))
    return np.asarray(eedf_samples)


def synthetic_trace(
    trace_id: str,
    bias_voltage: np.ndarray,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    p_value: float,
    ion_slope_a_per_v: float,
    ion_intercept_a: float,
    noise_std_a: float,
    probe_area_m2: float = 1.0e-6,
    probe_radius_m: float = 1.0e-4,
    gas: str = "Kr",
    probe_geometry: str = "cylindrical",
    seed: int = 42,
) -> TraceData:
    rng = np.random.default_rng(seed)
    clean_current = total_probe_current(
        bias_voltage=bias_voltage,
        probe_area_m2=probe_area_m2,
        plasma_potential_v=plasma_potential_v,
        electron_density_m3=electron_density_m3,
        electron_temperature_ev=electron_temperature_ev,
        p_value=p_value,
        ion_slope_a_per_v=ion_slope_a_per_v,
        ion_intercept_a=ion_intercept_a,
    )
    noisy_current = clean_current + rng.normal(0.0, noise_std_a, size=bias_voltage.size)
    return TraceData(
        trace_id=trace_id,
        trace_path=None,  # type: ignore[arg-type]
        bias_voltage=np.asarray(bias_voltage, dtype=float),
        probe_current=noisy_current,
        gas=gas,
        probe_geometry=probe_geometry,
        probe_radius_m=probe_radius_m,
        probe_area_m2=probe_area_m2,
        flow_sccm=None,
        discharge_current_a=None,
        current_std_a=noise_std_a,
        metadata={"synthetic": True},
    )

