from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .schema import BayesianModelResult, BayesianResult, LegacyResult, TraceData


def plot_iv_fit(trace: TraceData, bayes: BayesianResult, legacy: LegacyResult | None = None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(trace.bias_voltage, trace.probe_current, "k.", ms=3, label="Measured trace")
    valid_mask = bayes.model_current_quantiles["probe_current_a_valid_fraction"] > 0.5
    ax.plot(
        bayes.model_current_quantiles.loc[valid_mask, "bias_voltage_v"],
        bayes.model_current_quantiles.loc[valid_mask, "probe_current_a_median"],
        color="tab:blue",
        lw=2,
        label="Bayesian median (Vb <= Vp)",
    )
    ax.fill_between(
        bayes.model_current_quantiles.loc[valid_mask, "bias_voltage_v"],
        bayes.model_current_quantiles.loc[valid_mask, "probe_current_a_q2.25"],
        bayes.model_current_quantiles.loc[valid_mask, "probe_current_a_q97.75"],
        color="tab:blue",
        alpha=0.25,
        label="Bayesian 95.5%",
    )
    if legacy is not None and legacy.success and legacy.diagnostic_trace is not None:
        legacy_trace = legacy.diagnostic_trace
        legacy_model_mask = legacy_trace["legacy_model_valid"].astype(bool)
        if np.any(legacy_model_mask):
            ax.plot(
                legacy_trace.loc[legacy_model_mask, "bias_voltage_v"],
                legacy_trace.loc[legacy_model_mask, "legacy_total_model_a"],
                color="tab:orange",
                ls="--",
                lw=2,
                label="Lobbia retarding fit",
            )
    if "Vp" in bayes.summary.index:
        ax.axvline(
            bayes.summary.loc["Vp", "median"],
            color="tab:blue",
            ls=":",
            lw=1.5,
            label="Bayesian Vp",
        )
    if legacy is not None and legacy.success and "Vp" in legacy.summary.index:
        ax.axvline(
            legacy.summary.loc["Vp", "median"],
            color="tab:red",
            ls="--",
            lw=1.5,
            label="Legacy Vp",
        )
    ax.set_xlabel("Bias Voltage [V]")
    ax.set_ylabel("Probe Current [A]")
    ax.set_title(f"Langmuir Probe Fit: {trace.trace_id}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_legacy_diagnostics(
    trace: TraceData,
    legacy: LegacyResult,
    bayes: BayesianResult | None = None,
):
    if legacy.diagnostic_trace is None:
        raise ValueError("Legacy diagnostics are not available for this trace.")

    diagnostic = legacy.diagnostic_trace
    metadata = legacy.metadata

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax_raw = axes[0, 0]
    ax_derivative = axes[0, 1]
    ax_semilog = axes[1, 0]
    ax_zoom = axes[1, 1]

    ax_raw.plot(trace.bias_voltage, trace.probe_current, "k.", ms=3, label="Measured trace")
    ax_raw.plot(diagnostic["bias_voltage_v"], diagnostic["ion_current_a"], color="0.45", lw=1.5, label="Lobbia ion model")
    if "corrected_ion_current_a" in diagnostic.columns:
        ax_raw.plot(
            diagnostic["bias_voltage_v"],
            diagnostic["corrected_ion_current_a"],
            color="tab:brown",
            lw=1.25,
            alpha=0.8,
            label="Corrected ion model",
        )
    model_mask = diagnostic["legacy_model_valid"].astype(bool)
    ax_raw.plot(
        diagnostic.loc[model_mask, "bias_voltage_v"],
        diagnostic.loc[model_mask, "legacy_total_model_a"],
        color="tab:orange",
        ls="--",
        lw=2,
        label="Lobbia retarding fit",
    )
    if "Vf" in legacy.summary.index:
        ax_raw.axvline(legacy.summary.loc["Vf", "median"], color="tab:green", ls=":", lw=1.25, label="Lobbia Vf")
    if "Vp" in legacy.summary.index:
        ax_raw.axvline(legacy.summary.loc["Vp", "median"], color="tab:red", ls="--", lw=1.25, label="Lobbia Vp")
    if bayes is not None and "Vp" in bayes.summary.index:
        ax_raw.axvline(bayes.summary.loc["Vp", "median"], color="tab:blue", ls=":", lw=1.25, label="Bayesian Vp")
    ax_raw.set_xlabel("Bias Voltage [V]")
    ax_raw.set_ylabel("Probe Current [A]")
    ax_raw.set_title("Raw Trace And Legacy Model")
    ax_raw.grid(True, alpha=0.25)
    ax_raw.legend(loc="best")

    ax_derivative.plot(
        diagnostic["bias_voltage_v"],
        diagnostic["electron_current_a"],
        color="tab:purple",
        lw=1.5,
        label="Electron current",
    )
    ax_derivative.plot(
        diagnostic["bias_voltage_v"],
        diagnostic["electron_current_smooth_a"],
        color="tab:orange",
        lw=2,
        label="Smoothed electron current",
    )
    ax_derivative.set_xlabel("Bias Voltage [V]")
    ax_derivative.set_ylabel("Electron Current [A]")
    ax_derivative.set_title("Electron Current And Derivative")
    ax_derivative.grid(True, alpha=0.25)
    derivative_axis = ax_derivative.twinx()
    derivative_axis.plot(
        diagnostic["bias_voltage_v"],
        diagnostic["dIe_dV_a_per_v"],
        color="tab:red",
        alpha=0.8,
        label="dIe/dV",
    )
    if "Vp" in legacy.summary.index:
        derivative_axis.axvline(legacy.summary.loc["Vp", "median"], color="tab:red", ls="--", lw=1.0)
    derivative_axis.set_ylabel("dIe/dV [A/V]")
    lines_left, labels_left = ax_derivative.get_legend_handles_labels()
    lines_right, labels_right = derivative_axis.get_legend_handles_labels()
    ax_derivative.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    semilog_mask = diagnostic["electron_current_a"] > 0.0
    ax_semilog.plot(
        diagnostic.loc[semilog_mask, "bias_voltage_v"],
        diagnostic.loc[semilog_mask, "ln_electron_current"],
        "k.",
        ms=3,
        label="ln(Ie)",
    )
    fit_mask = diagnostic["semilog_window"].astype(bool)
    ax_semilog.plot(
        diagnostic.loc[fit_mask, "bias_voltage_v"],
        diagnostic.loc[fit_mask, "semilog_fit_ln_current"],
        color="tab:orange",
        lw=2,
        label="Lobbia semilog fit",
    )
    ax_semilog.axvline(metadata.get("floating_potential_v", np.nan), color="tab:green", ls=":", lw=1.0, label="Vf")
    ax_semilog.axvline(legacy.summary.loc["Vp", "median"], color="tab:red", ls="--", lw=1.0, label="Vp")
    upper_bound = metadata.get("semilog_upper_bound_v")
    if upper_bound is not None and np.isfinite(upper_bound):
        ax_semilog.axvline(upper_bound, color="tab:orange", ls=":", lw=1.0, label="Semilog upper bound")
    ax_semilog.set_xlabel("Bias Voltage [V]")
    ax_semilog.set_ylabel("ln(Ie)")
    ax_semilog.set_title("Semilog Temperature Fit")
    ax_semilog.grid(True, alpha=0.25)
    ax_semilog.legend(loc="best")

    ax_zoom.plot(trace.bias_voltage, trace.probe_current, "k.", ms=3, label="Measured trace")
    if bayes is not None:
        valid_mask = bayes.model_current_quantiles["probe_current_a_valid_fraction"] > 0.5
        ax_zoom.plot(
            bayes.model_current_quantiles.loc[valid_mask, "bias_voltage_v"],
            bayes.model_current_quantiles.loc[valid_mask, "probe_current_a_median"],
            color="tab:blue",
            lw=2,
            label="Bayesian median",
        )
        ax_zoom.fill_between(
            bayes.model_current_quantiles.loc[valid_mask, "bias_voltage_v"],
            bayes.model_current_quantiles.loc[valid_mask, "probe_current_a_q2.25"],
            bayes.model_current_quantiles.loc[valid_mask, "probe_current_a_q97.75"],
            color="tab:blue",
            alpha=0.15,
            label="Bayesian 95.5%",
        )
    ax_zoom.plot(
        diagnostic.loc[model_mask, "bias_voltage_v"],
        diagnostic.loc[model_mask, "legacy_total_model_a"],
        color="tab:orange",
        ls="--",
        lw=2,
        label="Lobbia retarding fit",
    )
    if "Vp" in legacy.summary.index:
        ax_zoom.axvline(legacy.summary.loc["Vp", "median"], color="tab:red", ls="--", lw=1.0, label="Lobbia Vp")
    if bayes is not None and "Vp" in bayes.summary.index:
        ax_zoom.axvline(bayes.summary.loc["Vp", "median"], color="tab:blue", ls=":", lw=1.0, label="Bayesian Vp")
    vf_value = legacy.summary.loc["Vf", "median"] if "Vf" in legacy.summary.index else np.nan
    vp_value = legacy.summary.loc["Vp", "median"] if "Vp" in legacy.summary.index else np.nan
    te_value = legacy.summary.loc["Te", "median"] if "Te" in legacy.summary.index else np.nan
    left = vf_value - 2.0
    right = vp_value + max(3.0 * te_value, 3.0)
    if np.isfinite(left) and np.isfinite(right) and right > left:
        ax_zoom.set_xlim(left, right)
    ax_zoom.set_xlabel("Bias Voltage [V]")
    ax_zoom.set_ylabel("Probe Current [A]")
    ax_zoom.set_title("Knee Comparison")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(loc="best")

    fig.suptitle(f"Lobbia Diagnostics: {trace.trace_id}")
    fig.tight_layout()
    return fig


def plot_eedf_fit(bayes: BayesianResult):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        bayes.eedf_density_quantiles["energy_ev"],
        bayes.eedf_density_quantiles["eedf_m3_ev_median"],
        color="tab:green",
        lw=2,
        label="Bayesian EEDF median",
    )
    ax.fill_between(
        bayes.eedf_density_quantiles["energy_ev"],
        bayes.eedf_density_quantiles["eedf_m3_ev_q16"],
        bayes.eedf_density_quantiles["eedf_m3_ev_q84"],
        color="tab:green",
        alpha=0.25,
        label="Bayesian 68%",
    )
    ax.set_xlabel("Electron Energy [eV]")
    ax.set_ylabel("EEDF [m^-3 eV^-1]")
    ax.set_title("Posterior EEDF")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_model_iv_overlay(trace: TraceData, model_result: BayesianModelResult):
    quantiles = model_result.current_quantiles
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(trace.bias_voltage, trace.probe_current, "k.", ms=3, label="Measured trace")
    valid_mask = quantiles["probe_current_a_valid_fraction"] > 0.5
    ax.plot(
        quantiles.loc[valid_mask, "bias_voltage_v"],
        quantiles.loc[valid_mask, "probe_current_a_median"],
        color="tab:blue",
        lw=2.0,
        label=f"{model_result.model_name.title()} median",
    )
    ax.fill_between(
        quantiles.loc[valid_mask, "bias_voltage_v"],
        quantiles.loc[valid_mask, "probe_current_a_q2.25"],
        quantiles.loc[valid_mask, "probe_current_a_q97.75"],
        color="tab:blue",
        alpha=0.18,
        label="95.5% credible band",
    )
    ax.fill_between(
        quantiles.loc[valid_mask, "bias_voltage_v"],
        quantiles.loc[valid_mask, "probe_current_a_q16"],
        quantiles.loc[valid_mask, "probe_current_a_q84"],
        color="tab:blue",
        alpha=0.30,
        label="68.3% credible band",
    )

    vf_value = float(model_result.summary.loc["Vf", "median"])
    vp_value = float(model_result.summary.loc["Vp", "median"])
    current_at_vp = float(np.interp(vp_value, trace.bias_voltage, trace.probe_current))
    if np.isfinite(vf_value):
        ax.scatter([vf_value], [0.0], color="tab:green", s=45, zorder=4, label="Vf")
        ax.axvline(vf_value, color="tab:green", ls=":", lw=1.2)
    if np.isfinite(vp_value):
        ax.scatter([vp_value], [current_at_vp], color="tab:red", s=45, zorder=4, label="Vp")
        ax.axvline(vp_value, color="tab:red", ls="--", lw=1.2)

    ax.set_xlabel("Bias Voltage [V]")
    ax.set_ylabel("Probe Current [A]")
    ax.set_title(f"Winning Model Overlay: {trace.trace_id} ({model_result.model_name})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_model_eedf(model_result: BayesianModelResult):
    quantiles = model_result.eedf_quantiles
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        quantiles["energy_ev"],
        quantiles["g_e_median"],
        color="tab:green",
        lw=2.0,
        label="Median g_e(E)",
    )
    ax.fill_between(
        quantiles["energy_ev"],
        quantiles["g_e_q2.25"],
        quantiles["g_e_q97.75"],
        color="tab:green",
        alpha=0.18,
        label="95.5% credible band",
    )
    ax.fill_between(
        quantiles["energy_ev"],
        quantiles["g_e_q16"],
        quantiles["g_e_q84"],
        color="tab:green",
        alpha=0.30,
        label="68.3% credible band",
    )
    ax.set_xlabel("Electron Energy [eV]")
    ax.set_ylabel("g_e(E) [eV^-1]")
    ax.set_title(f"Reconstructed EEDF: {model_result.model_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
