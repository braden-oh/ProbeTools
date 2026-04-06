#!/usr/bin/env python3
"""
Quick script to process all CSV files in the Axial Langmuir directory.

Usage:
    python process_axial_langmuir.py

Results are saved to:
    - Console output (terminal)
    - CSV summary: Axial Langmuir/analysis_summary.csv
    - Individual plots (if you add visualization code)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lobbia_lp import analyze, load_trace, io, potentials, electron as electron_module

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data directory (your Axial Langmuir traces)
DATA_DIR = Path("/Users/braden/Documents/ProbeTools/Claude_LP_Sandbox/Axial Langmuir")

# Probe parameters (customize for your probe)
PROBE_PARAMS = {
    'probe_radius_m': 50.8e-6,          # 0.002 inch = 50.8 μm
    'probe_length_m': 2.54e-3,          # 0.1 inch = 2.54 mm
    'probe_area_m2': np.pi * 2 * 50.8e-6 * (2.54e-3 + 50.8e-6),  # Cylindrical surface
    'gas': 'xe',                         # Xenon (change to 'kr', 'ar', 'n', 'zn' as needed)
    'probe_geometry': 'cylindrical',     # cylindrical, spherical, or planar
}

# Analysis parameters
MAX_ITERATIONS = 10
CONVERGENCE_TOLERANCE = 1e-3
VERBOSE = False  # Set to True to see iteration details

# Output directory for plots
OUTPUT_DIR = DATA_DIR / "plots"

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def save_iv_plot(V, I, result, output_path):
    """Save I-V characteristic plot with Vf and Vp marked."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(V, I * 1e3, 'k-', linewidth=2, label='Measured trace')

    # Mark floating potential
    ax.axvline(result.vf, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Vf = {result.vf:.2f} V')
    ax.plot(result.vf, np.interp(result.vf, V, I) * 1e3, 'ro', markersize=10,
            markerfacecolor='red', markeredgewidth=2, markeredgecolor='darkred')

    # Mark plasma potential
    ax.axvline(result.vp, color='blue', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Vp = {result.vp:.2f} V')
    ax.plot(result.vp, np.interp(result.vp, V, I) * 1e3, 'bo', markersize=10,
            markerfacecolor='blue', markeredgewidth=2, markeredgecolor='darkblue')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Bias Voltage [V]', fontsize=12)
    ax.set_ylabel('Probe Current [mA]', fontsize=12)
    ax.set_title(f'I-V Characteristic', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_semilog_fit_plot(V, I, result, output_path):
    """Save semilog fit plot showing Te extraction."""
    try:
        # Normalize trace
        V_norm, I_norm = io._normalize_trace(V, I)
        vf, _ = potentials.find_floating_potential(V_norm, I_norm)
        vp, _, _ = potentials.find_plasma_potential(V_norm, I_norm, vf)

        # Preliminary ion fit
        ion_mask = V_norm <= vf
        Ii_prelim = np.polyfit(V_norm[ion_mask], I_norm[ion_mask], 1)
        Ii_prelim = np.poly1d(Ii_prelim)(V_norm)
        Ii_prelim = np.minimum(Ii_prelim, 0.0)

        # Extract electron current
        Ie = I_norm - Ii_prelim
        Ie = np.maximum(Ie, 1e-15)

        # Get Te and fit info
        te, d_te, fit_info = electron_module.fit_electron_temperature(V_norm, Ie, vf, vp)

        # Extract fit window
        fit_mask = fit_info['fit_window_mask']
        V_fit = V_norm[fit_mask]
        Ie_fit = Ie[fit_mask]

        # Reconstruct fit line
        slope = fit_info['slope']
        intercept = fit_info['intercept']
        ln_Ie_fit_line_full = slope * V_fit + intercept
        Ie_fit_line_full = np.exp(ln_Ie_fit_line_full)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot all electron current data
        electron_region = V_norm >= vf
        ax.semilogy(V_norm[electron_region], Ie[electron_region], 'k.', markersize=6,
                    alpha=0.5, label='Ie (all data)')

        # Highlight fit region
        ax.semilogy(V_fit, Ie_fit, 'b.-', markersize=8, linewidth=2, alpha=0.8,
                    label='Ie (fit region)')

        # Plot the fitted exponential
        ax.semilogy(V_fit, Ie_fit_line_full, 'r-', linewidth=3,
                    label=f'Linear fit (Te = {te:.2f} ± {d_te:.2f} eV)', zorder=10)

        # Mark potentials
        ax.axvline(vf, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvline(vp, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(vf, ax.get_ylim()[1] * 0.5, f'Vf={vf:.2f}V', rotation=90, va='top',
                ha='right', fontsize=10, color='red')
        ax.text(vp, ax.get_ylim()[1] * 0.5, f'Vp={vp:.2f}V', rotation=90, va='top',
                ha='right', fontsize=10, color='blue')

        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlabel('Bias Voltage [V]', fontsize=12)
        ax.set_ylabel('Electron Current [A] (log scale)', fontsize=12)
        ax.set_title(f'Semilog Fit for Te', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.set_xlim([vf - 0.5, vp + 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"\n    ⚠️  Could not generate semilog plot: {e}")


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def main():
    """Process all CSV files in the directory."""

    # Create output directory for plots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")

    # Find all CSV files
    trace_files = sorted(DATA_DIR.glob("*.csv"))

    if not trace_files:
        print(f"❌ No CSV files found in {DATA_DIR}")
        print("   Check the directory path and file extension.")
        return

    print(f"📂 Found {len(trace_files)} CSV file(s) in {DATA_DIR.name}")
    print()

    # Process each file
    results_list = []

    for i, trace_file in enumerate(trace_files, 1):
        print(f"[{i}/{len(trace_files)}] Processing: {trace_file.name}...", end=" ", flush=True)

        try:
            # Load trace
            V, I = load_trace(str(trace_file))

            # Analyze
            result = analyze(
                V, I,
                **PROBE_PARAMS,
                max_iterations=MAX_ITERATIONS,
                convergence_tolerance=CONVERGENCE_TOLERANCE,
                compute_eedf=False,
                verbose=VERBOSE
            )

            # Store results
            results_list.append({
                'Filename': trace_file.name,
                'Vf (V)': result.vf,
                'Vp (V)': result.vp,
                'Te (eV)': result.te,
                'ne (m⁻³)': result.ne,
                'ni (m⁻³)': result.ni,
                'n (m⁻³)': result.n,
                'λD (m)': result.debye_length,
                'rp/λD': result.rp_lambda,
                'Sheath_Regime': result.sheath_regime,
                'Converged': result.converged,
                'Iterations': result.n_iterations,
            })

            # Save plots
            stem = trace_file.stem
            iv_plot_path = OUTPUT_DIR / f"{stem}_IV_characteristic.png"
            semilog_plot_path = OUTPUT_DIR / f"{stem}_Te_semilog_fit.png"

            save_iv_plot(V, I, result, iv_plot_path)
            save_semilog_fit_plot(V, I, result, semilog_plot_path)

            print(f"✓ Te={result.te:6.2f} eV, ne={result.ne:.2e} m⁻³, ni={result.ni:.2e} m⁻³")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            results_list.append({
                'Filename': trace_file.name,
                'Error': str(e),
            })

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Create DataFrame
    df_results = pd.DataFrame(results_list)

    # Display results
    print(df_results.to_string(index=False))

    # Save to CSV
    output_csv = DATA_DIR / "analysis_summary.csv"
    df_results.to_csv(output_csv, index=False)
    print()
    print(f"✓ CSV results saved to: {output_csv}")
    print(f"✓ Individual plots saved to: {OUTPUT_DIR}/")
    print(f"   - Each trace has two plots:")
    print(f"     * filename_IV_characteristic.png (I-V with Vf/Vp marked)")
    print(f"     * filename_Te_semilog_fit.png (Te exponential fit)")

    # Summary statistics
    successful = len([r for r in results_list if 'Error' not in r])
    failed = len(results_list) - successful
    print(f"\n📊 Summary: {successful} successful, {failed} failed")

    if successful > 0:
        df_success = df_results[~df_results['Error'].isna()] if 'Error' in df_results.columns else df_results
        if len(df_success) > 0:
            print(f"\n   Te range: {df_success['Te (eV)'].min():.2f} – {df_success['Te (eV)'].max():.2f} eV")
            print(f"   ne range: {df_success['ne (m⁻³)'].min():.2e} – {df_success['ne (m⁻³)'].max():.2e} m⁻³")
            print(f"   ni range: {df_success['ni (m⁻³)'].min():.2e} – {df_success['ni (m⁻³)'].max():.2e} m⁻³")


if __name__ == "__main__":
    main()
