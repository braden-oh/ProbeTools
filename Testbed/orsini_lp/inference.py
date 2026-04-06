from __future__ import annotations

from dataclasses import asdict, dataclass

import dynesty
import numpy as np
import pandas as pd
from dynesty import utils as dyfunc
from scipy.signal import savgol_filter

from .forward import default_energy_grid, density_weighted_eedf_samples, total_probe_current
from .schema import BayesianResult, TraceData, summarize_samples, summarize_vector_quantiles


@dataclass(slots=True)
class BayesianConfig:
    nlive: int = 125
    dlogz: float = 0.5
    sample: str = "rwalk"
    bound: str = "multi"
    maxiter: int | None = None
    maxcall: int | None = None
    random_seed: int = 7
    quadrature_nodes: int = 512
    quadrature_xmax: float = 18.0
    posterior_draws: int = 300
    max_points: int | None = 160
    min_sub_vp_points: int = 6
    prior_overrides: dict[str, tuple[float, float]] | None = None


def fit_bayesian(trace: TraceData, config: BayesianConfig | None = None) -> BayesianResult:
    cfg = config or BayesianConfig()
    full_bias_voltage = np.asarray(trace.bias_voltage, dtype=float)
    full_probe_current = np.asarray(trace.probe_current, dtype=float)
    full_current_std = None if trace.current_std_a is None else np.asarray(trace.current_std_a, dtype=float)

    floating_potential = _estimate_floating_potential(full_bias_voltage, full_probe_current)
    dense_sweep_mask = _select_dense_sweep_mask(full_bias_voltage)
    dense_bias_voltage = full_bias_voltage[dense_sweep_mask]
    dense_probe_current = full_probe_current[dense_sweep_mask]
    dense_current_std = None if full_current_std is None else _masked_current_std(full_current_std, dense_sweep_mask, dense_probe_current.shape)

    fit_lower_voltage = float(np.min(dense_bias_voltage))
    fit_upper_voltage = _estimate_fit_upper_voltage(dense_bias_voltage, dense_probe_current, floating_potential)
    full_fit_mask = dense_sweep_mask & (full_bias_voltage <= fit_upper_voltage)
    fit_bias_voltage = full_bias_voltage[full_fit_mask]
    fit_probe_current = full_probe_current[full_fit_mask]
    fit_current_std = None if full_current_std is None else _masked_current_std(full_current_std, full_fit_mask, fit_probe_current.shape)

    if cfg.max_points is not None and fit_bias_voltage.size > cfg.max_points:
        fit_indices = np.linspace(0, fit_bias_voltage.size - 1, cfg.max_points, dtype=int)
        fit_bias_voltage = fit_bias_voltage[fit_indices]
        fit_probe_current = fit_probe_current[fit_indices]
        if fit_current_std is not None:
            fit_current_std = fit_current_std[fit_indices]

    fixed_fit_mask = dense_sweep_mask & (full_bias_voltage <= fit_bias_voltage.max())
    if int(np.count_nonzero(fixed_fit_mask)) < cfg.min_sub_vp_points:
        fit_bias_voltage = dense_bias_voltage
        fit_probe_current = dense_probe_current
        fit_current_std = dense_current_std
        fit_upper_voltage = float(np.max(dense_bias_voltage))
        fit_lower_voltage = float(np.min(dense_bias_voltage))

    priors = _build_prior_bounds(
        trace=trace,
        bias_voltage=full_bias_voltage,
        probe_current=full_probe_current,
        floating_potential=floating_potential,
        fit_upper_voltage=fit_upper_voltage,
        fit_bias_max=float(np.max(fit_bias_voltage)),
    )
    if cfg.prior_overrides:
        priors.update(cfg.prior_overrides)

    parameter_names = ["Vp", "ne", "Te", "p", "I1", "I2"]
    infer_sigma = fit_current_std is None
    if infer_sigma:
        parameter_names.append("log10_sigma_I")

    rng = np.random.default_rng(cfg.random_seed)

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        transformed = np.empty_like(unit_cube)
        for index, name in enumerate(parameter_names):
            lower, upper = priors[name]
            transformed[index] = lower + (upper - lower) * unit_cube[index]
        return transformed

    def log_likelihood(theta: np.ndarray) -> float:
        parameters = dict(zip(parameter_names, theta, strict=True))
        if parameters["Vp"] < float(np.max(fit_bias_voltage)):
            return -np.inf

        sigma = (
            np.full_like(fit_probe_current, 10.0 ** parameters["log10_sigma_I"])
            if infer_sigma
            else np.asarray(fit_current_std, dtype=float)
        )
        sigma = np.maximum(sigma, 1.0e-12)
        model_current = total_probe_current(
            bias_voltage=fit_bias_voltage,
            probe_area_m2=trace.probe_area_m2,
            plasma_potential_v=parameters["Vp"],
            electron_density_m3=parameters["ne"],
            electron_temperature_ev=parameters["Te"],
            p_value=parameters["p"],
            ion_slope_a_per_v=parameters["I1"],
            ion_intercept_a=parameters["I2"],
            quadrature_nodes=cfg.quadrature_nodes,
            quadrature_xmax=cfg.quadrature_xmax,
        )
        residual = fit_probe_current - model_current
        return float(
            -0.5
            * (
                np.sum((residual / sigma) ** 2)
                + np.sum(np.log(2.0 * np.pi * sigma**2))
            )
        )

    sampler = dynesty.NestedSampler(
        loglikelihood=log_likelihood,
        prior_transform=prior_transform,
        ndim=len(parameter_names),
        nlive=cfg.nlive,
        sample=cfg.sample,
        bound=cfg.bound,
        rstate=rng,
    )
    sampler.run_nested(dlogz=cfg.dlogz, maxiter=cfg.maxiter, maxcall=cfg.maxcall, print_progress=False)
    results = sampler.results

    weights = np.exp(results.logwt - results.logz[-1])
    posterior_equal = dyfunc.resample_equal(results.samples, weights)
    if posterior_equal.shape[0] > cfg.posterior_draws:
        indices = np.linspace(0, posterior_equal.shape[0] - 1, cfg.posterior_draws, dtype=int)
        posterior_equal = posterior_equal[indices]

    posterior_frame = pd.DataFrame(posterior_equal, columns=parameter_names)
    if infer_sigma:
        posterior_frame["sigma_I"] = np.power(10.0, posterior_frame["log10_sigma_I"])

    summary = summarize_samples(posterior_frame.to_numpy(), list(posterior_frame.columns))
    summary.loc["n_qn"] = summary.loc["ne"]

    model_current_samples = _model_current_samples(
        trace=trace,
        bias_voltage=full_bias_voltage,
        posterior_frame=posterior_frame,
        config=cfg,
        fit_lower_voltage=fit_lower_voltage,
    )
    current_quantiles = summarize_vector_quantiles(
        model_current_samples,
        coordinate=full_bias_voltage,
        coordinate_name="bias_voltage_v",
        value_name="probe_current_a",
    )

    energy_grid = default_energy_grid(posterior_frame["Te"].to_numpy())
    eedf_samples = density_weighted_eedf_samples(energy_grid, posterior_frame[["Vp", "ne", "Te", "p", "I1", "I2"]].to_numpy())
    eedf_quantiles = summarize_vector_quantiles(
        eedf_samples,
        coordinate=energy_grid,
        coordinate_name="energy_ev",
        value_name="eedf_m3_ev",
    )

    diagnostics = {
        "niter": int(results.niter),
        "ncall": int(np.sum(results.ncall)),
        "efficiency": float(results.eff),
        "prior_bounds": priors,
        "fit_point_count": int(fit_bias_voltage.size),
        "original_point_count": int(trace.point_count),
        "floating_potential_v": float(floating_potential),
        "fit_window_lower_v": float(fit_lower_voltage),
        "fit_window_upper_v": float(fit_upper_voltage),
        "fit_window_point_count": int(fit_bias_voltage.size),
    }

    return BayesianResult(
        trace_id=trace.trace_id,
        posterior_samples=posterior_frame,
        summary=summary,
        log_evidence=float(results.logz[-1]),
        model_current_samples=model_current_samples,
        model_current_quantiles=current_quantiles,
        energy_grid_ev=energy_grid,
        eedf_density_samples=eedf_samples,
        eedf_density_quantiles=eedf_quantiles,
        config=asdict(cfg),
        diagnostics=diagnostics,
    )


def _build_prior_bounds(
    trace: TraceData,
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    floating_potential: float,
    fit_upper_voltage: float,
    fit_bias_max: float,
) -> dict[str, tuple[float, float]]:
    voltage_span = float(np.ptp(bias_voltage))
    fit_window_span = max(fit_upper_voltage - floating_potential, 1.0)
    current_span = max(float(np.ptp(probe_current)), 1.0e-8)
    rough_vp = fit_upper_voltage
    vp_width = max(1.0, 0.75 * fit_window_span)
    vp_min = max(
        floating_potential + 1.0 * np.median(np.diff(bias_voltage)),
        fit_bias_max + 0.25 * np.median(np.diff(bias_voltage)),
        rough_vp - vp_width,
    )
    vp_max = max(
        fit_bias_max + 4.0 * np.median(np.diff(bias_voltage)),
        rough_vp + vp_width,
    )
    if vp_min >= vp_max:
        vp_min = fit_bias_max + 0.25 * np.median(np.diff(bias_voltage))
        vp_max = fit_bias_max + max(2.0, 0.2 * voltage_span)

    ion_count = max(6, int(0.2 * bias_voltage.size))
    ion_fit = np.polyfit(bias_voltage[:ion_count], probe_current[:ion_count], 1)
    ion_baseline = np.polyval(ion_fit, bias_voltage)
    electron_guess = probe_current - ion_baseline
    positive_mask = (electron_guess > 0.0) & (bias_voltage < rough_vp)
    te_guess = 3.0
    if np.count_nonzero(positive_mask) >= 8:
        fit_v = bias_voltage[positive_mask]
        fit_i = np.log(np.clip(electron_guess[positive_mask], 1.0e-12, None))
        slope, _intercept = np.polyfit(fit_v, fit_i, 1)
        if slope > 0:
            te_guess = float(np.clip(1.0 / slope, 0.2, 20.0))

    ies_guess = max(float(np.max(probe_current) - np.polyval(ion_fit, rough_vp)), 1.0e-8)
    ne_guess = ies_guess / (
        trace.probe_area_m2
        * np.sqrt((1.602176634e-19 ** 3) / (2.0 * np.pi * 9.1093837015e-31))
        * np.sqrt(te_guess)
    )
    ne_guess = float(np.clip(ne_guess, 1.0e15, 1.0e21))

    slope_scale = current_span / max(voltage_span, 1.0)
    intercept_scale = max(abs(float(np.min(probe_current))), abs(float(np.max(probe_current))), current_span)
    priors = {
        "Vp": (vp_min, vp_max),
        "ne": (max(1.0e14, ne_guess / 10.0), min(1.0e22, ne_guess * 10.0)),
        "Te": (max(0.15, te_guess / 2.5), min(10.0, max(3.0, te_guess * 2.5))),
        "p": (1.0, 3.0),
        "I1": (-8.0 * slope_scale, 8.0 * slope_scale),
        "I2": (-3.0 * intercept_scale, 3.0 * intercept_scale),
        "log10_sigma_I": (
            float(np.log10(max(current_span * 1.0e-5, 1.0e-10))),
            float(np.log10(max(current_span, 1.0e-9))),
        ),
    }
    return priors


def _estimate_fit_upper_voltage(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    floating_potential: float,
) -> float:
    mask = bias_voltage >= floating_potential
    fit_bias = bias_voltage[mask]
    fit_current = probe_current[mask]

    if fit_bias.size < 11:
        return float(np.max(fit_bias))

    window = min(21, fit_bias.size if fit_bias.size % 2 == 1 else fit_bias.size - 1)
    if window < 5:
        return float(np.max(fit_bias))
    if window % 2 == 0:
        window -= 1

    delta_v = float(np.median(np.diff(fit_bias)))
    derivative = savgol_filter(fit_current, window, 3, deriv=1, delta=delta_v)
    peak_index = int(np.argmax(derivative))
    peak_index = min(max(peak_index + 2, 1), fit_bias.size - 1)
    return float(fit_bias[peak_index])


def _estimate_rough_plasma_potential(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
) -> float:
    unique_bias, unique_indices = np.unique(bias_voltage, return_index=True)
    unique_current = probe_current[unique_indices]

    if unique_bias.size >= 3 and np.ptp(unique_bias) > 0.0:
        gradient = np.gradient(unique_current, unique_bias)
        gradient = np.where(np.isfinite(gradient), gradient, -np.inf)
        if np.isfinite(gradient).any():
            return float(unique_bias[np.argmax(gradient)])

    index_gradient = np.gradient(probe_current)
    return float(bias_voltage[int(np.argmax(index_gradient))])


def _estimate_floating_potential(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
) -> float:
    sign_product = probe_current[:-1] * probe_current[1:]
    crossing_indices = np.where(sign_product <= 0.0)[0]
    if crossing_indices.size == 0:
        return float(np.median(bias_voltage))

    index = int(crossing_indices[0])
    x0 = float(bias_voltage[index])
    x1 = float(bias_voltage[index + 1])
    y0 = float(probe_current[index])
    y1 = float(probe_current[index + 1])
    if y1 == y0:
        return x0
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _model_current_samples(
    trace: TraceData,
    bias_voltage: np.ndarray,
    posterior_frame: pd.DataFrame,
    config: BayesianConfig,
    fit_lower_voltage: float,
) -> np.ndarray:
    current_samples = []
    for row in posterior_frame.itertuples(index=False):
        sample_current = np.full_like(bias_voltage, np.nan, dtype=float)
        fit_mask = (bias_voltage >= float(fit_lower_voltage)) & (bias_voltage <= row.Vp)
        sample_current[fit_mask] = total_probe_current(
            bias_voltage=bias_voltage[fit_mask],
            probe_area_m2=trace.probe_area_m2,
            plasma_potential_v=row.Vp,
            electron_density_m3=row.ne,
            electron_temperature_ev=row.Te,
            p_value=row.p,
            ion_slope_a_per_v=row.I1,
            ion_intercept_a=row.I2,
            quadrature_nodes=config.quadrature_nodes,
            quadrature_xmax=config.quadrature_xmax,
        )
        current_samples.append(sample_current)
    return np.asarray(current_samples)


def _masked_current_std(
    current_std: np.ndarray | None,
    fit_mask: np.ndarray,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    sigma = np.asarray(current_std, dtype=float)
    if sigma.ndim == 0:
        return np.full(output_shape, float(sigma))
    return sigma[fit_mask]


def _select_dense_sweep_mask(
    bias_voltage: np.ndarray,
    gap_factor: float = 5.0,
) -> np.ndarray:
    bias = np.asarray(bias_voltage, dtype=float)
    if bias.size < 3:
        return np.ones_like(bias, dtype=bool)

    diffs = np.diff(bias)
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.size == 0:
        return np.ones_like(bias, dtype=bool)

    typical_step = float(np.median(positive_diffs))
    gap_threshold = gap_factor * typical_step
    break_indices = np.where(diffs > gap_threshold)[0]
    if break_indices.size == 0:
        return np.ones_like(bias, dtype=bool)

    segment_bounds: list[tuple[int, int]] = []
    start = 0
    for break_index in break_indices:
        stop = int(break_index) + 1
        segment_bounds.append((start, stop))
        start = stop
    segment_bounds.append((start, bias.size))

    segment_start, segment_stop = max(
        segment_bounds,
        key=lambda bounds: (bounds[1] - bounds[0], bias[bounds[1] - 1] - bias[bounds[0]]),
    )
    mask = np.zeros_like(bias, dtype=bool)
    mask[segment_start:segment_stop] = True
    return mask
