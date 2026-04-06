from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma

from .schema import TraceData

E_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31
EPS0 = 8.8541878128e-12


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


def cylindrical_probe_area(probe_radius_m: float, probe_length_m: float) -> float:
    if probe_radius_m <= 0.0 or probe_length_m <= 0.0:
        raise ValueError("Probe radius and length must both be positive.")
    return float(math.pi * probe_radius_m * (probe_radius_m + 2.0 * probe_length_m))


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


def debye_length(electron_density_m3: float, electron_temperature_ev: float) -> float:
    if electron_density_m3 <= 0.0 or electron_temperature_ev <= 0.0:
        return float("nan")
    return float(np.sqrt(EPS0 * electron_temperature_ev / (electron_density_m3 * E_CHARGE)))


def ion_current_linear(
    bias_voltage: np.ndarray,
    plasma_potential_v: float,
    ion_slope_a_per_v: float,
    ion_intercept_a: float,
) -> np.ndarray:
    return ion_slope_a_per_v * (plasma_potential_v - np.asarray(bias_voltage, dtype=float)) + ion_intercept_a


def ion_current_oml_cylindrical(
    bias_voltage: np.ndarray,
    probe_area_m2: float,
    plasma_potential_v: float,
    ion_density_m3: float,
    ion_mass_kg: float,
) -> np.ndarray:
    delta_v = np.clip(plasma_potential_v - np.asarray(bias_voltage, dtype=float), 0.0, None)
    prefactor = -E_CHARGE * ion_density_m3 * probe_area_m2 / math.pi
    return prefactor * np.sqrt(2.0 * E_CHARGE * delta_v / ion_mass_kg)


def cylindrical_transitional_coefficients(radius_over_debye: float) -> tuple[float, float]:
    a_param = 1.18 - 0.00080 * radius_over_debye**1.35
    b_param = 0.0684 + (0.722 + 0.928 * radius_over_debye) ** -0.729
    return float(a_param), float(b_param)


def ion_current_transitional_cylindrical(
    bias_voltage: np.ndarray,
    probe_area_m2: float,
    probe_radius_m: float,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    ion_mass_kg: float,
) -> np.ndarray:
    lambda_debye = debye_length(electron_density_m3, electron_temperature_ev)
    ratio = probe_radius_m / lambda_debye
    a_param, b_param = cylindrical_transitional_coefficients(ratio)
    delta = np.clip((plasma_potential_v - np.asarray(bias_voltage, dtype=float)) / electron_temperature_ev, 0.0, None)
    prefactor = (
        -a_param
        * E_CHARGE
        * electron_density_m3
        * probe_area_m2
        * math.sqrt(E_CHARGE * electron_temperature_ev / (2.0 * math.pi * ion_mass_kg))
    )
    return prefactor * np.power(delta, b_param)


def total_model_current(
    model_name: str,
    bias_voltage: np.ndarray,
    probe_area_m2: float,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    p_value: float,
    ion_mass_kg: float | None = None,
    probe_radius_m: float | None = None,
    ion_slope_a_per_v: float | None = None,
    ion_intercept_a: float | None = None,
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
    model_key = model_name.strip().lower()
    if model_key == "linear":
        if ion_slope_a_per_v is None or ion_intercept_a is None:
            raise ValueError("The linear ion-current model requires I1 and I2.")
        ion = ion_current_linear(
            bias_voltage=bias_voltage,
            plasma_potential_v=plasma_potential_v,
            ion_slope_a_per_v=ion_slope_a_per_v,
            ion_intercept_a=ion_intercept_a,
        )
    elif model_key == "oml":
        if ion_mass_kg is None:
            raise ValueError("The OML ion-current model requires an ion mass.")
        ion = ion_current_oml_cylindrical(
            bias_voltage=bias_voltage,
            probe_area_m2=probe_area_m2,
            plasma_potential_v=plasma_potential_v,
            ion_density_m3=electron_density_m3,
            ion_mass_kg=ion_mass_kg,
        )
    elif model_key == "transitional":
        if ion_mass_kg is None or probe_radius_m is None:
            raise ValueError("The transitional ion-current model requires an ion mass and probe radius.")
        ion = ion_current_transitional_cylindrical(
            bias_voltage=bias_voltage,
            probe_area_m2=probe_area_m2,
            probe_radius_m=probe_radius_m,
            plasma_potential_v=plasma_potential_v,
            electron_density_m3=electron_density_m3,
            electron_temperature_ev=electron_temperature_ev,
            ion_mass_kg=ion_mass_kg,
        )
    else:
        raise ValueError(f"Unsupported ion-current model {model_name!r}.")
    return electron + ion


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
    return total_model_current(
        model_name="linear",
        bias_voltage=bias_voltage,
        probe_area_m2=probe_area_m2,
        plasma_potential_v=plasma_potential_v,
        electron_density_m3=electron_density_m3,
        electron_temperature_ev=electron_temperature_ev,
        p_value=p_value,
        ion_slope_a_per_v=ion_slope_a_per_v,
        ion_intercept_a=ion_intercept_a,
        quadrature_nodes=quadrature_nodes,
        quadrature_xmax=quadrature_xmax,
    )


def estimate_floating_potential(bias_voltage: np.ndarray, probe_current: np.ndarray) -> float:
    bias = np.asarray(bias_voltage, dtype=float)
    current = np.asarray(probe_current, dtype=float)
    sign_product = current[:-1] * current[1:]
    crossing_indices = np.where(sign_product <= 0.0)[0]
    if crossing_indices.size == 0:
        return float("nan")
    index = int(crossing_indices[0])
    x0 = float(bias[index])
    x1 = float(bias[index + 1])
    y0 = float(current[index])
    y1 = float(current[index + 1])
    if y1 == y0:
        return x0
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


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


def normalized_eedf_samples(energy_grid_ev: np.ndarray, posterior_samples: np.ndarray) -> np.ndarray:
    eedf_samples = []
    for row in posterior_samples:
        te = float(row[2])
        p_value = float(row[3])
        eedf_samples.append(normalized_eedf(energy_grid_ev, te, p_value))
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


def synthetic_trace_for_model(
    model_name: str,
    trace_id: str,
    bias_voltage: np.ndarray,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    p_value: float,
    noise_std_a: float,
    probe_radius_m: float,
    probe_length_m: float,
    ion_mass_kg: float,
    ion_slope_a_per_v: float | None = None,
    ion_intercept_a: float | None = None,
    gas: str = "Kr",
    seed: int = 42,
) -> TraceData:
    probe_area_m2 = cylindrical_probe_area(probe_radius_m, probe_length_m)
    rng = np.random.default_rng(seed)
    clean_current = total_model_current(
        model_name=model_name,
        bias_voltage=bias_voltage,
        probe_area_m2=probe_area_m2,
        probe_radius_m=probe_radius_m,
        plasma_potential_v=plasma_potential_v,
        electron_density_m3=electron_density_m3,
        electron_temperature_ev=electron_temperature_ev,
        p_value=p_value,
        ion_mass_kg=ion_mass_kg,
        ion_slope_a_per_v=ion_slope_a_per_v,
        ion_intercept_a=ion_intercept_a,
    )
    noisy_current = clean_current + rng.normal(0.0, noise_std_a, size=np.asarray(bias_voltage).size)
    return TraceData(
        trace_id=trace_id,
        trace_path=None,  # type: ignore[arg-type]
        bias_voltage=np.asarray(bias_voltage, dtype=float),
        probe_current=noisy_current,
        gas=gas,
        probe_geometry="cylindrical",
        probe_radius_m=probe_radius_m,
        probe_area_m2=probe_area_m2,
        flow_sccm=None,
        discharge_current_a=None,
        current_std_a=noise_std_a,
        metadata={
            "synthetic": True,
            "model_name": model_name,
            "probe_length_m": probe_length_m,
            "ion_mass_kg": ion_mass_kg,
        },
    )
