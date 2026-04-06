# Lobbia LP: Langmuir Probe Analysis Library

A Python implementation of the **Lobbia & Beal (2017)** recommended practices for analyzing single Langmuir probe I-V characteristics in electric propulsion plasma environments.

## Overview

This package provides a complete pipeline for extracting plasma parameters from Langmuir probe measurements:
- **Floating potential (Vf)** via zero-crossing detection
- **Plasma potential (Vp)** via electron current derivative  
- **Electron temperature (Te)** via semilog slope method (Eq. 18)
- **Electron density (ne)** via saturation current (Eq. 3)
- **Ion density (ni)** via sheath-corrected models (Eq. 4, 6–8)
- **Quasineutral density (n)** as the mean of ne and ni
- **Debye length (λD)** and probe sheath regime classification

The analysis implements the **11-step algorithm** from Lobbia Section II.F with an **iterative sheath-correction loop** (Steps 8–11) that converges to self-consistent plasma parameters.

## Installation

```bash
cd "/Users/braden/Documents/ProbeTools/Claude_LP_Sandbox/Lobbia Method"
pip install -e .
```

Or simply ensure the package directory is in your Python path.

## Quick Start

### Interactive Jupyter Notebook

For a comprehensive walkthrough with examples and visualizations, see **`example_usage.ipynb`**:
- Single-file analysis with plots
- Batch processing directories
- Visualizations: I-V curves with Vf/Vp markers, Te semilog fit, batch comparison plots

### Command-Line Example

```python
from lobbia_lp import analyze, load_trace

# Load trace data
V, I = load_trace("trace_file.txt")

# Analyze
result = analyze(
    V, I,
    probe_radius_m=5e-5,        # 50 μm
    probe_area_m2=8.27e-7,      # mm²
    probe_length_m=2.54e-3,     # 2.54 mm
    gas="xe",                    # Xenon
    probe_geometry="cylindrical"
)

# View results
print(result.to_dataframe())

# Access individual parameters
print(f"Te = {result.te:.2f} eV")
print(f"ne = {result.ne:.2e} m^-3")
print(f"ni = {result.ni:.2e} m^-3")
```

## Input Data Format

The package expects trace data in **tab-delimited** or **CSV** format with columns:
```
Bias Voltage (V)    Probe Current (A)
-21.0               -0.002291
-9.789              -0.002132
...
39.96               0.009338
```

Alternatively, provide numpy arrays directly:
```python
analyze(bias_voltage_array, probe_current_array, ...)
```

## Key Functions

### Main Entry Point
- **`analyze()`** — Run full 11-step Langmuir probe analysis pipeline
  - Returns `LPResult` object with all derived quantities and uncertainties
  - Supports optional EEDF computation via Druyvesteyn method
  - Verbose mode for debugging

### Data Loading
- **`load_trace(filepath)`** — Load trace from .txt, .csv, or .hdf5 file
  - Auto-detects format and column names
  - Handles sign convention correction
  - Returns sorted (V, I) arrays

### Analysis Modules
- **`potentials.py`** — Vf and Vp detection
- **`electron.py`** — Te and ne extraction via semilog and saturation methods
- **`ion.py`** — Ion current models (thin-sheath, OML, transitional)
- **`eedf.py`** — Electron energy distribution function computation
- **`uncertainty.py`** — Error propagation per Eq. 32
- **`plotting.py`** — Diagnostic visualization

## Algorithm Summary

The 11-step algorithm (Lobbia Section II.F):

1. **Normalize trace** — Ensure ascending V, correct sign convention
2. **Find Vf** — Linear interpolation of zero-crossing in probe current
3. **Preliminary ion fit** — Linear fit to ion-saturation region (V < Vf)
4. **Compute Ie** — Electron current as I_probe − I_ion
5. **Find Vp** — Maximum of smoothed dIe/dV
6. **Fit Te** — Slope method: Te = (d ln Ie / dV)^-1 over [Vf, Vp − 2Te]
7. **Compute ne** — From saturation current: ne = Ie,sat / (e*Ap*√(eTe / 2πme))
8. **Compute λD** — Debye length from ne and Te
9. **Select sheath regime** — Based on rp/λD ratio
10. **Sheath-corrected ni** — Using thin-sheath, transitional, or OML models
11. **Iterate** — Refine ni until convergence (steps 4–10 repeated)

## Sheath Models

The package implements three ion current collection models selectable by rp/λD:

| Regime | rp/λD | Model | Reference |
|--------|-------|-------|-----------|
| **Thin** | ≥ 50 | Child-Langmuir sheath (Eq. 4–5) | Lobbia §II.B |
| **Transitional** | 3–50 | Narasimhan/Sheridan fits (Eq. 7–8) | Lobbia §II.D |
| **OML** | ≤ 3 | Orbital-motion-limited (Eq. 6) | Lobbia §II.C |

## Output

`LPResult` object contains:
- **Plasma parameters:** `vf`, `vp`, `te`, `ne`, `ni`, `n`, `ie_sat`, `ii_sat`
- **Derived:** `debye_length`, `rp_lambda`, `sheath_regime`
- **Uncertainties:** `uncertainties` dict with `d_Vf`, `d_Vp`, `d_Te`, `d_ne`, `d_ni`, `d_n`
- **EEDF:** (optional) `eedf_energy`, `eedf`, `ne_from_eedf`
- **Diagnostics:** `n_iterations`, `converged`, `diagnostics`

## Example Output

```
Results:
                 Value   Uncertainty  Unit
Vf        8.721e+00    3.229e-01       V
Vp        1.027e+01    1.414e-01       V
Te        3.070e+00    2.936e-01      eV
n         2.382e+28    1.185e+28    m^-3
ne        2.275e+26    1.487e+25    m^-3
ni        4.742e+28    2.371e+28    m^-3
Ie_sat    3.537e-03    3.537e-05      A
Ii_sat   -2.291e-03    2.291e-04      A
lambda_D  3.457e-19       NaN         m
rp_lambda 1.470e+14       NaN         1
```

## Testing

Run the test suite:
```bash
cd tests
python test_lobbia.py
```

Or with pytest:
```bash
pytest tests/test_lobbia.py -v
```

Test data is located at `/Users/braden/Documents/ProbeTools/Testbed/LP_*.txt`.

## Supported Gases

- `xe` — Xenon (131.293 AMU)
- `kr` — Krypton (83.798 AMU)
- `ar` — Argon (39.948 AMU)
- `n` — Nitrogen (14.007 AMU)
- `zn` — Zinc (65.38 AMU)

Add more by editing `constants.GAS_MASSES_AMU`.

## Probe Geometries

- `cylindrical` (default) — Thin-wire cylindrical probes
- `spherical` — Spherical electrode probes
- `planar` — Planar collector probes

## Limitations & Assumptions

Per Lobbia §II.A, the theory assumes:
1. Cold ions: Ti/Te ≪ 1
2. Maxwellian electron velocity distribution
3. Collisionless plasma (Kn ≫ 1)
4. Electrostatic (∂B/∂t ≈ 0)
5. Non-magnetized electrons (rL,e/rp ≫ 1)
6. Quasi-neutral plasma (ni ≈ ne)
7. Isotropic and homogeneous plasma

The accuracy of extracted plasma parameters is typically **20–50%** (see Lobbia §II.M).

## References

Lobbia, R. B., & Beal, B. E. (2017).  
**Recommended Practice for Use of Langmuir Probes in Electric Propulsion Testing.**  
*Journal of Propulsion and Power*, **33**(3), 566–581.  
DOI: 10.2514/1.B35531

## License

This code was developed for educational and research purposes.

## Contact

For questions or issues, contact the package author or refer to the Lobbia & Beal paper.
