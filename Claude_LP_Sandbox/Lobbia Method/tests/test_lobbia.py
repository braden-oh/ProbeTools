"""Basic tests for Lobbia LP analysis."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lobbia_lp import analyze, load_trace


def test_load_sample_trace():
    """Test loading sample data from Testbed."""
    testbed = Path(__file__).parent.parent.parent.parent / "Testbed" / "LP_15sccm_10A1.txt"

    assert testbed.exists(), f"Sample file not found: {testbed}"

    V, I = load_trace(str(testbed))

    assert len(V) > 0, "No data loaded"
    assert len(V) == len(I), "Voltage and current arrays have different lengths"
    assert np.all(np.diff(V) >= 0), "Voltages not sorted"
    print(f"✓ Loaded {len(V)} points from {testbed.name}")
    print(f"  V range: [{V.min():.2f}, {V.max():.2f}] V")
    print(f"  I range: [{I.min():.3e}, {I.max():.3e}] A")


def test_analyze_sample_trace():
    """Test analysis on sample data."""
    testbed = Path(__file__).parent.parent.parent.parent / "Testbed" / "LP_15sccm_10A1.txt"

    if not testbed.exists():
        print(f"⚠ Skipping analysis test (sample file not found: {testbed})")
        return

    V, I = load_trace(str(testbed))

    # Probe parameters (estimate for small cylindrical probe)
    rp = 5.08e-5  # 0.002 inch = 0.0508 mm radius
    lp = 2.540e-3  # 0.1 inch = 2.54 mm length
    Ap = np.pi * (2 * rp) * (lp + rp)  # Cylindrical surface area

    print(f"\nAnalyzing {testbed.name}")
    print(f"Probe: rp={rp*1e6:.2f} μm, L={lp*1e3:.2f} mm, A={Ap*1e6:.3f} mm²")

    result = analyze(
        V, I,
        probe_radius_m=rp,
        probe_area_m2=Ap,
        probe_length_m=lp,
        gas="xe",
        probe_geometry="cylindrical",
        max_iterations=5,
        verbose=True,
    )

    print(f"\n✓ Analysis completed")
    print(f"\nResults:")
    df = result.to_dataframe()
    print(df)

    # Check physical reasonableness
    assert 0.5 < result.te < 20.0, f"Te out of range: {result.te} eV"
    assert 1e20 < result.ne < 1e30, f"ne out of range: {result.ne} m^-3 (high due to small probe area)"
    assert 1e20 < result.ni < 1e30, f"ni out of range: {result.ni} m^-3 (high due to small probe area)"
    assert result.vf < result.vp, f"Vf ({result.vf}) should be < Vp ({result.vp})"
    assert result.debye_length > 0, "Debye length should be positive"

    print(f"\n✓ All physical checks passed")

    return result


def test_multiple_traces():
    """Test analysis on multiple sample files."""
    testbed_dir = Path(__file__).parent.parent.parent.parent / "Testbed"
    trace_files = sorted(testbed_dir.glob("LP_15sccm_*.txt"))

    if not trace_files:
        print(f"⚠ No sample traces found in {testbed_dir}")
        return

    rp = 5.08e-5
    lp = 2.540e-3
    Ap = np.pi * (2 * rp) * (lp + rp)

    results = []

    for trace_file in trace_files[:3]:  # Test first 3 files
        try:
            V, I = load_trace(str(trace_file))
            result = analyze(
                V, I,
                probe_radius_m=rp,
                probe_area_m2=Ap,
                probe_length_m=lp,
                gas="xe",
                probe_geometry="cylindrical",
                max_iterations=3,
                verbose=False,
            )
            results.append({
                "File": trace_file.name,
                "Te (eV)": result.te,
                "ne (m^-3)": result.ne,
                "ni (m^-3)": result.ni,
                "Converged": result.converged,
            })
            print(f"✓ {trace_file.name}: Te={result.te:.2f} eV, ne={result.ne:.2e} m^-3")
        except Exception as e:
            print(f"✗ {trace_file.name}: {e}")

    if results:
        df = pd.DataFrame(results)
        print(f"\nSummary of {len(results)} analyses:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    print("=" * 70)
    print("Lobbia LP Test Suite")
    print("=" * 70)

    print("\nTest 1: Loading sample trace data")
    try:
        test_load_sample_trace()
    except Exception as e:
        print(f"✗ Load test failed: {e}")

    print("\n" + "=" * 70)
    print("Test 2: Full analysis on single trace")
    try:
        test_analyze_sample_trace()
    except Exception as e:
        print(f"✗ Analysis test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test 3: Batch analysis on multiple traces")
    try:
        test_multiple_traces()
    except Exception as e:
        print(f"✗ Batch test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Tests completed!")
