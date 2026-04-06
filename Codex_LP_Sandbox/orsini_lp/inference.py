from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import dynesty
import numpy as np
import pandas as pd
from dynesty import utils as dyfunc
from scipy.signal import savgol_filter

from .forward import (
    cylindrical_probe_area,
    debye_length,
    default_energy_grid,
    density_weighted_eedf_samples,
    estimate_floating_potential as estimate_model_floating_potential,
    normalized_eedf_samples,
    total_model_current,
    total_probe_current,
)
from .schema import (
    BayesianModelResult,
    BayesianResult,
    ModelComparisonAnalysis,
    ProbeGeometry,
    TraceData,
    nan_summary,
    summarize_samples,
    summarize_vector_quantiles,
)


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
    fit_strategy: str = "retarding"
    fit_upper_voltage_v: float | None = None
    fit_lower_voltage_v: float | None = None
    energy_points: int = 250


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

    try:
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
    except RuntimeError as exc:
        raise RuntimeError(f"Nested sampling failed for trace {trace.trace_id}: {exc}") from exc

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


SUPPORTED_PHYSICAL_MODELS = ("linear", "oml", "transitional")


def fit_bayesian_model(
    trace: TraceData,
    model_name: str,
    *,
    ion_mass_kg: float,
    probe_length_m: float,
    prior_bounds: dict[str, tuple[float, float]] | None = None,
    current_covariance: np.ndarray | None = None,
    config: BayesianConfig | None = None,
) -> BayesianModelResult:
    cfg = config or BayesianConfig()
    model_key = model_name.strip().lower()
    if model_key not in SUPPORTED_PHYSICAL_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}. Supported models are {SUPPORTED_PHYSICAL_MODELS}."
        )
    if ion_mass_kg <= 0.0:
        raise ValueError("ion_mass_kg must be positive.")
    if probe_length_m <= 0.0:
        raise ValueError("probe_length_m must be positive.")

    prepared = _prepare_model_fit_data(trace, config=cfg, current_covariance=current_covariance)
    priors = _build_model_prior_bounds(
        trace=trace,
        model_name=model_key,
        prior_bounds=prior_bounds,
        config=cfg,
        fit_bias_voltage=prepared["fit_bias_voltage"],
        fit_upper_voltage=float(prepared["fit_upper_voltage"]),
        floating_potential=float(prepared["floating_potential"]),
    )

    parameter_names = ["Vp", "ne", "Te", "p"]
    if model_key == "linear":
        parameter_names.extend(["I1", "I2"])

    infer_sigma = prepared["fit_current_std"] is None and prepared["fit_covariance"] is None
    if infer_sigma:
        parameter_names.append("log10_sigma_I")

    rng = np.random.default_rng(cfg.random_seed)

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        transformed = np.empty_like(unit_cube)
        for index, name in enumerate(parameter_names):
            lower, upper = priors[name]
            transformed[index] = lower + (upper - lower) * unit_cube[index]
        return transformed

    observed_vf = float(prepared["floating_potential"])
    fit_covariance_terms = _prepare_covariance_terms(prepared["fit_covariance"])

    def log_likelihood(theta: np.ndarray) -> float:
        parameters = dict(zip(parameter_names, theta, strict=True))
        if parameters["ne"] <= 0.0 or parameters["Te"] <= 0.0 or parameters["p"] <= 0.0:
            return -np.inf
        if not np.isfinite(observed_vf) or parameters["Vp"] <= observed_vf:
            return -np.inf

        if not _model_sample_is_valid(
            model_name=model_key,
            parameters=parameters,
            probe_radius_m=trace.probe_radius_m,
        ):
            return -np.inf

        try:
            full_model_current = total_model_current(
                model_name=model_key,
                bias_voltage=prepared["full_bias_voltage"],
                probe_area_m2=trace.probe_area_m2,
                plasma_potential_v=parameters["Vp"],
                electron_density_m3=parameters["ne"],
                electron_temperature_ev=parameters["Te"],
                p_value=parameters["p"],
                ion_mass_kg=ion_mass_kg,
                probe_radius_m=trace.probe_radius_m,
                ion_slope_a_per_v=parameters.get("I1"),
                ion_intercept_a=parameters.get("I2"),
                quadrature_nodes=cfg.quadrature_nodes,
                quadrature_xmax=cfg.quadrature_xmax,
            )
        except ValueError:
            return -np.inf

        if not np.all(np.isfinite(full_model_current)):
            return -np.inf

        return _gaussian_log_likelihood(
            observed=prepared["fit_probe_current"],
            model=full_model_current[prepared["fit_indices"]],
            current_std=prepared["fit_current_std"],
            covariance_terms=fit_covariance_terms,
            log10_sigma=parameters.get("log10_sigma_I"),
        )

    try:
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
    except RuntimeError as exc:
        raise RuntimeError(f"Nested sampling failed for trace {trace.trace_id}: {exc}") from exc

    weights = np.exp(results.logwt - results.logz[-1])
    posterior_equal = dyfunc.resample_equal(results.samples, weights)
    if posterior_equal.shape[0] > cfg.posterior_draws:
        indices = np.linspace(0, posterior_equal.shape[0] - 1, cfg.posterior_draws, dtype=int)
        posterior_equal = posterior_equal[indices]

    posterior_frame = pd.DataFrame(posterior_equal, columns=parameter_names)
    if infer_sigma:
        posterior_frame["sigma_I"] = np.power(10.0, posterior_frame["log10_sigma_I"])

    current_samples = _model_current_samples_for_model(
        trace=trace,
        bias_voltage=prepared["full_bias_voltage"],
        fit_lower_voltage=prepared["fit_lower_voltage"],
        posterior_frame=posterior_frame,
        config=cfg,
        model_name=model_key,
        ion_mass_kg=ion_mass_kg,
    )
    current_quantiles = summarize_vector_quantiles(
        current_samples,
        coordinate=prepared["full_bias_voltage"],
        coordinate_name="bias_voltage_v",
        value_name="probe_current_a",
    )

    energy_grid = default_energy_grid(posterior_frame["Te"].to_numpy(), points=cfg.energy_points)
    eedf_samples = normalized_eedf_samples(
        energy_grid,
        posterior_frame[["Vp", "ne", "Te", "p"]].to_numpy(),
    )
    eedf_quantiles = summarize_vector_quantiles(
        eedf_samples,
        coordinate=energy_grid,
        coordinate_name="energy_ev",
        value_name="g_e",
    )

    summary = summarize_samples(posterior_frame.to_numpy(), list(posterior_frame.columns))
    vf_value = float(prepared["floating_potential"])
    vf_samples = np.full(posterior_frame.shape[0], vf_value, dtype=float)
    vf_summary = summarize_samples(vf_samples.reshape(-1, 1), ["Vf"])
    summary = pd.concat([vf_summary, summary])
    ie_sat_samples = np.asarray(
        [_interpolate_current_at_voltage(prepared["full_bias_voltage"], prepared["full_probe_current"], vp) for vp in posterior_frame["Vp"]],
        dtype=float,
    )
    ie_sat_summary = summarize_samples(ie_sat_samples.reshape(-1, 1), ["Ie_sat"])
    summary = pd.concat([summary, ie_sat_summary])
    if model_key in {"oml", "transitional"}:
        summary = pd.concat([summary, summarize_samples(posterior_frame[["ne"]].to_numpy(), ["ni"])])
    else:
        summary = pd.concat([summary, nan_summary("ni")])

    ordered_parameters = ["Vf", "Vp", "Ie_sat", "ne", "ni", "Te", "p"]
    ordered_parameters.extend(
        name for name in summary.index if name not in ordered_parameters
    )
    summary = summary.loc[ordered_parameters]

    ratio_samples = np.asarray(
        [trace.probe_radius_m / debye_length(row.ne, row.Te) for row in posterior_frame.itertuples(index=False)],
        dtype=float,
    )
    diagnostics = {
        "model_name": model_key,
        "niter": int(results.niter),
        "ncall": int(np.sum(results.ncall)),
        "efficiency": float(results.eff),
        "fit_point_count": int(prepared["fit_bias_voltage"].size),
        "original_point_count": int(trace.point_count),
        "floating_potential_v": vf_value,
        "fit_window_lower_v": float(prepared["fit_lower_voltage"]),
        "fit_window_upper_v": float(prepared["fit_upper_voltage"]),
        "observed_vp_guess_v": float(prepared["fit_upper_voltage"]),
        "electron_saturation_current_a": float(_interpolate_current_at_voltage(prepared["full_bias_voltage"], prepared["full_probe_current"], float(np.median(posterior_frame["Vp"])))),
        "prior_bounds": priors,
        "probe_length_m": float(probe_length_m),
        "ion_mass_kg": float(ion_mass_kg),
        "rp_over_lambda_median": float(np.nanmedian(ratio_samples)),
        "rp_over_lambda_q16": float(np.nanpercentile(ratio_samples, 15.865)),
        "rp_over_lambda_q84": float(np.nanpercentile(ratio_samples, 84.135)),
    }

    return BayesianModelResult(
        trace_id=trace.trace_id,
        model_name=model_key,
        posterior_samples=posterior_frame,
        summary=summary,
        log_evidence=float(results.logz[-1]),
        log_evidence_error=float(results.logzerr[-1]),
        current_samples=current_samples,
        current_quantiles=current_quantiles,
        energy_grid_ev=energy_grid,
        eedf_samples=eedf_samples,
        eedf_quantiles=eedf_quantiles,
        config=asdict(cfg),
        diagnostics=diagnostics,
    )


def compare_physical_models(
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    *,
    probe_radius_m: float,
    probe_length_m: float,
    ion_mass_kg: float,
    prior_bounds: dict[str, tuple[float, float]] | None = None,
    current_std_a: float | np.ndarray | None = None,
    current_covariance: np.ndarray | None = None,
    trace_id: str = "trace",
    gas: str = "",
    config: BayesianConfig | None = None,
    make_plots: bool = True,
) -> ModelComparisonAnalysis:
    trace, sorted_covariance = _trace_from_arrays(
        trace_id=trace_id,
        bias_voltage=bias_voltage,
        probe_current=probe_current,
        probe_radius_m=probe_radius_m,
        probe_length_m=probe_length_m,
        gas=gas,
        current_std_a=current_std_a,
        current_covariance=current_covariance,
    )
    geometry = ProbeGeometry(radius_m=probe_radius_m, length_m=probe_length_m, area_m2=trace.probe_area_m2)

    model_results = {
        model_name: fit_bayesian_model(
            trace,
            model_name,
            ion_mass_kg=ion_mass_kg,
            probe_length_m=probe_length_m,
            prior_bounds=prior_bounds,
            current_covariance=sorted_covariance,
            config=config,
        )
        for model_name in SUPPORTED_PHYSICAL_MODELS
    }

    log_evidence_table = pd.DataFrame.from_records(
        [
            {
                "model": model_name,
                "log_evidence": result.log_evidence,
                "log_evidence_error": result.log_evidence_error,
            }
            for model_name, result in model_results.items()
        ]
    ).set_index("model").sort_values("log_evidence", ascending=False)

    bayes_factor_rows = []
    reference_log_evidence = model_results["linear"].log_evidence
    for model_name in ("oml", "transitional"):
        log_bayes_factor = model_results[model_name].log_evidence - reference_log_evidence
        bayes_factor_rows.append(
            {
                "comparison": f"{model_name}_vs_linear",
                "candidate_model": model_name,
                "reference_model": "linear",
                "log_bayes_factor": float(log_bayes_factor),
                "log10_bayes_factor": float(log_bayes_factor / np.log(10.0)),
                "bayes_factor": float(np.exp(np.clip(log_bayes_factor, -700.0, 700.0))),
            }
        )
    bayes_factor_table = pd.DataFrame.from_records(bayes_factor_rows).set_index("comparison")

    winning_model = str(log_evidence_table.index[0])
    figures: dict[str, object] = {}
    if make_plots:
        from .plotting import plot_model_eedf, plot_model_iv_overlay

        figures["iv_overlay"] = plot_model_iv_overlay(trace, model_results[winning_model])
        figures["eedf_overlay"] = plot_model_eedf(model_results[winning_model])

    diagnostics = {
        "probe_area_m2": trace.probe_area_m2,
        "probe_radius_m": probe_radius_m,
        "probe_length_m": probe_length_m,
        "ion_mass_kg": ion_mass_kg,
        "fit_strategy": (config or BayesianConfig()).fit_strategy,
    }

    return ModelComparisonAnalysis(
        trace_id=trace.trace_id,
        geometry=geometry,
        ion_mass_kg=ion_mass_kg,
        model_results=model_results,
        log_evidence_table=log_evidence_table,
        bayes_factor_table=bayes_factor_table,
        winning_model=winning_model,
        figures=figures,
        diagnostics=diagnostics,
    )


def _trace_from_arrays(
    *,
    trace_id: str,
    bias_voltage: np.ndarray,
    probe_current: np.ndarray,
    probe_radius_m: float,
    probe_length_m: float,
    gas: str,
    current_std_a: float | np.ndarray | None,
    current_covariance: np.ndarray | None,
) -> tuple[TraceData, np.ndarray | None]:
    bias = np.asarray(bias_voltage, dtype=float)
    current = np.asarray(probe_current, dtype=float)
    if bias.shape != current.shape:
        raise ValueError("bias_voltage and probe_current must have the same shape.")

    order = np.argsort(bias)
    sorted_bias = bias[order]
    sorted_current = current[order]

    if current_std_a is None:
        sorted_std: float | np.ndarray | None = None
    else:
        sigma = np.asarray(current_std_a, dtype=float)
        sorted_std = float(sigma) if sigma.ndim == 0 else sigma[order]

    sorted_covariance = None
    if current_covariance is not None:
        covariance = np.asarray(current_covariance, dtype=float)
        if covariance.shape != (bias.size, bias.size):
            raise ValueError("current_covariance must be square with dimensions matching the trace length.")
        sorted_covariance = covariance[np.ix_(order, order)]

    area_m2 = cylindrical_probe_area(probe_radius_m, probe_length_m)
    trace = TraceData(
        trace_id=trace_id,
        trace_path=Path(trace_id),
        bias_voltage=sorted_bias,
        probe_current=sorted_current,
        gas=gas,
        probe_geometry="cylindrical",
        probe_radius_m=probe_radius_m,
        probe_area_m2=area_m2,
        flow_sccm=None,
        discharge_current_a=None,
        current_std_a=sorted_std,
        metadata={"probe_length_m": probe_length_m},
    )
    return trace, sorted_covariance


def _prepare_model_fit_data(
    trace: TraceData,
    *,
    config: BayesianConfig,
    current_covariance: np.ndarray | None,
) -> dict[str, np.ndarray | float | None]:
    full_bias_voltage = np.asarray(trace.bias_voltage, dtype=float)
    full_probe_current = np.asarray(trace.probe_current, dtype=float)
    full_current_std = None if trace.current_std_a is None else np.asarray(trace.current_std_a, dtype=float)

    dense_sweep_mask = _select_dense_sweep_mask(full_bias_voltage)
    floating_potential = _estimate_floating_potential(full_bias_voltage, full_probe_current)
    dense_bias_voltage = full_bias_voltage[dense_sweep_mask]
    dense_probe_current = full_probe_current[dense_sweep_mask]
    dense_current_std = None if full_current_std is None else _masked_current_std(full_current_std, dense_sweep_mask, dense_probe_current.shape)
    dense_indices = np.flatnonzero(dense_sweep_mask)
    dense_covariance = None if current_covariance is None else np.asarray(current_covariance)[np.ix_(dense_indices, dense_indices)]

    fit_lower_voltage = (
        float(config.fit_lower_voltage_v)
        if config.fit_lower_voltage_v is not None
        else float(np.min(full_bias_voltage))   # No lower bound on fit window
        #else float(floating_potential)     # Set lower bound on fit window to Vf
    )
    fit_upper_voltage = (
        float(config.fit_upper_voltage_v)
        if config.fit_upper_voltage_v is not None
        else _estimate_fit_upper_voltage(dense_bias_voltage, dense_probe_current, floating_potential)
    )

    if config.fit_strategy.strip().lower() == "all":
        full_fit_mask = np.ones_like(full_bias_voltage, dtype=bool)
        fit_lower_voltage = float(np.min(full_bias_voltage))
        fit_upper_voltage = float(np.max(full_bias_voltage))
    else:
        full_fit_mask = dense_sweep_mask & (full_bias_voltage >= fit_lower_voltage) & (full_bias_voltage <= fit_upper_voltage)

    fit_bias_voltage = full_bias_voltage[full_fit_mask]
    fit_probe_current = full_probe_current[full_fit_mask]
    fit_current_std = None if full_current_std is None else _masked_current_std(full_current_std, full_fit_mask, fit_probe_current.shape)
    fit_mask_indices = np.flatnonzero(full_fit_mask)
    fit_covariance = None if current_covariance is None else np.asarray(current_covariance)[np.ix_(fit_mask_indices, fit_mask_indices)]

    if config.max_points is not None and fit_bias_voltage.size > config.max_points:
        fit_indices = np.linspace(0, fit_bias_voltage.size - 1, config.max_points, dtype=int)
        fit_bias_voltage = fit_bias_voltage[fit_indices]
        fit_probe_current = fit_probe_current[fit_indices]
        if fit_current_std is not None:
            fit_current_std = fit_current_std[fit_indices]
        if fit_covariance is not None:
            fit_covariance = fit_covariance[np.ix_(fit_indices, fit_indices)]
        fit_index_array = np.flatnonzero(full_fit_mask)[fit_indices]
    else:
        fit_index_array = np.flatnonzero(full_fit_mask)

    if fit_bias_voltage.size < config.min_sub_vp_points:
        fit_bias_voltage = dense_bias_voltage
        fit_probe_current = dense_probe_current
        fit_current_std = dense_current_std
        fit_covariance = dense_covariance
        fit_index_array = np.flatnonzero(dense_sweep_mask)
        fit_lower_voltage = float(np.min(dense_bias_voltage))
        fit_upper_voltage = float(np.max(dense_bias_voltage))

    return {
        "full_bias_voltage": full_bias_voltage,
        "full_probe_current": full_probe_current,
        "fit_bias_voltage": fit_bias_voltage,
        "fit_probe_current": fit_probe_current,
        "fit_current_std": fit_current_std,
        "fit_covariance": fit_covariance,
        "fit_indices": fit_index_array,
        "floating_potential": floating_potential,
        "fit_lower_voltage": fit_lower_voltage,
        "fit_upper_voltage": fit_upper_voltage,
        "fit_lower_voltage": fit_lower_voltage
    }


def _build_model_prior_bounds(
    *,
    trace: TraceData,
    model_name: str,
    prior_bounds: dict[str, tuple[float, float]] | None,
    config: BayesianConfig,
    fit_bias_voltage: np.ndarray,
    fit_upper_voltage: float,
    floating_potential: float,
) -> dict[str, tuple[float, float]]:
    auto_priors = _build_prior_bounds(
        trace=trace,
        bias_voltage=np.asarray(trace.bias_voltage, dtype=float),
        probe_current=np.asarray(trace.probe_current, dtype=float),
        floating_potential=floating_potential,
        fit_upper_voltage=fit_upper_voltage,
        fit_bias_max=float(np.max(fit_bias_voltage)),
    )
    user_priors = {} if prior_bounds is None else dict(prior_bounds)
    if config.prior_overrides:
        user_priors.update(config.prior_overrides)

    required = ["Vp", "ne", "Te", "p"]
    if model_name == "linear":
        required.extend(["I1", "I2"])
    elif "ni" in user_priors:
        user_priors["ne"] = _intersect_bounds(user_priors["ne"], user_priors["ni"], "ne", "ni")

    priors = {name: user_priors.get(name, auto_priors[name]) for name in required}
    if prior_bounds is not None:
        missing = [name for name in required if name not in user_priors]
        if missing:
            raise ValueError(f"Missing required prior bounds for {model_name!r}: {missing}")

    if "Vf" in user_priors:
        priors["Vf"] = user_priors["Vf"]

    if "log10_sigma_I" in user_priors:
        priors["log10_sigma_I"] = user_priors["log10_sigma_I"]
    else:
        priors["log10_sigma_I"] = auto_priors["log10_sigma_I"]

    bias_step = max(float(np.median(np.diff(np.unique(np.asarray(trace.bias_voltage, dtype=float))))), 1.0e-6)
    vp_lower, vp_upper = priors["Vp"]
    vp_lower = max(vp_lower, floating_potential + bias_step)
    vp_upper = min(vp_upper, fit_upper_voltage + 2.0 * bias_step)
    if vp_upper <= vp_lower:
        vp_upper = vp_lower + max(2.0 * bias_step, 0.25)
    priors["Vp"] = (vp_lower, vp_upper)
    return priors


def _intersect_bounds(
    bounds_a: tuple[float, float],
    bounds_b: tuple[float, float],
    name_a: str,
    name_b: str,
) -> tuple[float, float]:
    lower = max(bounds_a[0], bounds_b[0])
    upper = min(bounds_a[1], bounds_b[1])
    if lower >= upper:
        raise ValueError(
            f"The prior bounds for {name_a} and {name_b} do not overlap: {bounds_a} vs {bounds_b}."
        )
    return (lower, upper)


def _model_sample_is_valid(
    *,
    model_name: str,
    parameters: dict[str, float],
    probe_radius_m: float,
) -> bool:
    if not np.isfinite(parameters["Vp"]) or not np.isfinite(parameters["ne"]) or not np.isfinite(parameters["Te"]):
        return False
    if parameters["ne"] <= 0.0 or parameters["Te"] <= 0.0:
        return False
    if model_name == "linear":
        return True
    lambda_debye = debye_length(parameters["ne"], parameters["Te"])
    if not np.isfinite(lambda_debye) or lambda_debye <= 0.0:
        return False
    ratio = probe_radius_m / lambda_debye
    if model_name == "oml":
        return ratio <= 3.0
    if model_name == "transitional":
        return 3.0 < ratio < 50.0
    return False


def _prepare_covariance_terms(covariance: np.ndarray | None) -> tuple[np.ndarray, float] | None:
    if covariance is None:
        return None
    cholesky = np.linalg.cholesky(covariance)
    log_det = 2.0 * np.sum(np.log(np.diag(cholesky)))
    return cholesky, float(log_det)


def _gaussian_log_likelihood(
    *,
    observed: np.ndarray,
    model: np.ndarray,
    current_std: np.ndarray | None,
    covariance_terms: tuple[np.ndarray, float] | None,
    log10_sigma: float | None,
) -> float:
    residual = observed - model
    if covariance_terms is not None:
        cholesky, log_det = covariance_terms
        solve = np.linalg.solve(cholesky, residual)
        quadratic = float(np.dot(solve, solve))
        return float(-0.5 * (quadratic + log_det + observed.size * np.log(2.0 * np.pi)))

    sigma = (
        np.full_like(observed, 10.0 ** float(log10_sigma))
        if current_std is None
        else np.asarray(current_std, dtype=float)
    )
    sigma = np.maximum(sigma, 1.0e-12)
    return float(
        -0.5
        * (
            np.sum((residual / sigma) ** 2)
            + np.sum(np.log(2.0 * np.pi * sigma**2))
        )
    )


def _model_current_samples_for_model(
    trace: TraceData,
    bias_voltage: np.ndarray,
    fit_lower_voltage: float,
    posterior_frame: pd.DataFrame,
    config: BayesianConfig,
    model_name: str,
    ion_mass_kg: float,
) -> np.ndarray:
    current_samples = []
    measured_vf = estimate_model_floating_potential(trace.bias_voltage, trace.probe_current)
    for row in posterior_frame.itertuples(index=False):
        sample_current = np.full_like(bias_voltage, np.nan, dtype=float)
        #fit_mask = (bias_voltage >= measured_vf) & (bias_voltage <= row.Vp)
        fit_mask = (bias_voltage >= fit_lower_voltage) & (bias_voltage <= row.Vp)
        sample_current[fit_mask] = total_model_current(
                model_name=model_name,
                bias_voltage=bias_voltage[fit_mask],
                probe_area_m2=trace.probe_area_m2,
                probe_radius_m=trace.probe_radius_m,
                plasma_potential_v=row.Vp,
                electron_density_m3=row.ne,
                electron_temperature_ev=row.Te,
                p_value=row.p,
                ion_mass_kg=ion_mass_kg,
                ion_slope_a_per_v=getattr(row, "I1", None),
                ion_intercept_a=getattr(row, "I2", None),
                quadrature_nodes=config.quadrature_nodes,
                quadrature_xmax=config.quadrature_xmax,
            )
        current_samples.append(sample_current)
    return np.asarray(current_samples)


def _interpolate_current_at_voltage(bias_voltage: np.ndarray, probe_current: np.ndarray, target_voltage: float) -> float:
    bias = np.asarray(bias_voltage, dtype=float)
    current = np.asarray(probe_current, dtype=float)
    if bias.size == 0:
        return float("nan")
    if target_voltage <= float(np.min(bias)):
        return float(current[np.argmin(bias)])
    if target_voltage >= float(np.max(bias)):
        return float(current[np.argmax(bias)])
    return float(np.interp(target_voltage, bias, current))


def _failed_model_result(
    *,
    trace: TraceData,
    model_name: str,
    config: BayesianConfig,
    probe_length_m: float,
    ion_mass_kg: float,
    priors: dict[str, tuple[float, float]],
    prepared: dict[str, np.ndarray | float | None],
    error: str,
) -> BayesianModelResult:
    summary_names = ["Vf", "Vp", "Ie_sat", "ne", "ni", "Te", "p"]
    if model_name == "linear":
        summary_names.extend(["I1", "I2"])
    summary = pd.concat([nan_summary(name) for name in summary_names])
    current_quantiles = pd.DataFrame(
        {
            "bias_voltage_v": prepared["full_bias_voltage"],
            "probe_current_a_median": np.nan,
            "probe_current_a_q16": np.nan,
            "probe_current_a_q84": np.nan,
            "probe_current_a_q2.25": np.nan,
            "probe_current_a_q97.75": np.nan,
            "probe_current_a_valid_fraction": 0.0,
        }
    )
    energy_grid = np.linspace(0.0, 20.0, config.energy_points)
    eedf_quantiles = pd.DataFrame(
        {
            "energy_ev": energy_grid,
            "g_e_median": np.nan,
            "g_e_q16": np.nan,
            "g_e_q84": np.nan,
            "g_e_q2.25": np.nan,
            "g_e_q97.75": np.nan,
            "g_e_valid_fraction": 0.0,
        }
    )
    return BayesianModelResult(
        trace_id=trace.trace_id,
        model_name=model_name,
        posterior_samples=pd.DataFrame(),
        summary=summary,
        log_evidence=float("-inf"),
        log_evidence_error=float("inf"),
        current_samples=np.empty((0, len(prepared["full_bias_voltage"]))),
        current_quantiles=current_quantiles,
        energy_grid_ev=energy_grid,
        eedf_samples=np.empty((0, config.energy_points)),
        eedf_quantiles=eedf_quantiles,
        config=asdict(config),
        diagnostics={
            "model_name": model_name,
            "probe_length_m": probe_length_m,
            "ion_mass_kg": ion_mass_kg,
            "fit_point_count": int(len(prepared["fit_bias_voltage"])),
            "original_point_count": int(trace.point_count),
            "fit_window_lower_v": float(prepared["fit_lower_voltage"]),
            "fit_window_upper_v": float(prepared["fit_upper_voltage"]),
            "floating_potential_v": float(prepared["floating_potential"]),
            "prior_bounds": priors,
            "error": error,
        },
    )
