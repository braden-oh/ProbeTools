from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from orsini_lp import BayesianConfig, compare_physical_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = PROJECT_ROOT / "SciTech Sample Data" / "Kr_15sccm_10A1.txt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "scitech_compare"

# These values mirror the cylindrical probe geometry used elsewhere in the workspace.
PROBE_RADIUS_M = 25.4e-6
PROBE_LENGTH_M = 2.54e-3
KRYPTON_ION_MASS_KG = 83.798 * 1.66053906660e-27


def main() -> None:
    frame = pd.read_csv(TRACE_PATH, sep="\t")
    bias = frame["Bias Voltage (V)"].to_numpy(dtype=float)
    current = frame["Probe Current (A)"].to_numpy(dtype=float)

    priors = {
        "Vp": (6.0, 20.0),
        "Vf": (0.0, 12.0),
        "ne": (1.0e16, 1.0e19),
        "ni": (1.0e16, 1.0e19),
        "Te": (0.5, 15.0),
        "p": (1.0, 3.0),
        "I1": (-5.0e-4, 5.0e-4),
        "I2": (-5.0e-3, 5.0e-4),
    }

    analysis = compare_physical_models(
        bias,
        current,
        probe_radius_m=PROBE_RADIUS_M,
        probe_length_m=PROBE_LENGTH_M,
        ion_mass_kg=KRYPTON_ION_MASS_KG,
        prior_bounds=priors,
        trace_id=TRACE_PATH.stem,
        config=BayesianConfig(
            nlive=80,
            dlogz=1.0,
            posterior_draws=120,
            max_points=140,
            random_seed=17,
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis.log_evidence_table.to_csv(OUTPUT_DIR / "log_evidence.csv")
    analysis.bayes_factor_table.to_csv(OUTPUT_DIR / "bayes_factors.csv")
    for model_name, result in analysis.model_results.items():
        result.summary.to_csv(OUTPUT_DIR / f"{model_name}_summary.csv")
        result.current_quantiles.to_csv(OUTPUT_DIR / f"{model_name}_iv_quantiles.csv", index=False)
        result.eedf_quantiles.to_csv(OUTPUT_DIR / f"{model_name}_eedf_quantiles.csv", index=False)

    analysis.figures["iv_overlay"].savefig(OUTPUT_DIR / "winning_model_iv.png", dpi=180, bbox_inches="tight")
    analysis.figures["eedf_overlay"].savefig(OUTPUT_DIR / "winning_model_eedf.png", dpi=180, bbox_inches="tight")

    print(analysis.log_evidence_table)
    print()
    print(analysis.bayes_factor_table)
    print()
    print(f"Winning model: {analysis.winning_model}")


if __name__ == "__main__":
    main()
