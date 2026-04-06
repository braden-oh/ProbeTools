from __future__ import annotations

from pathlib import Path

from orsini_lp.adapters import cylindrical_probe_metadata, infer_local_manifest_rows, parse_discharge_filename
from orsini_lp.io import load_manifest, load_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_PROBE = cylindrical_probe_metadata(exposed_length_m=2.540e-3, diameter_m=5.08e-5)


def test_local_adapter_discovers_delimited_and_hdf5_examples():
    manifest = infer_local_manifest_rows(PROJECT_ROOT, **FALLBACK_PROBE)
    assert set(manifest["file_format"]) == {"delimited", "hdf5_matrix"}
    assert {"kr_10sccm_10a", "lp_33_34deg_150v", "lp_15sccm_10a1"}.issubset(set(manifest["trace_id"]))


def test_load_trace_from_sample_manifest_delimited():
    manifest = load_manifest(PROJECT_ROOT / "sample_data" / "local_manifest.csv")
    manifest["manifest_path"] = manifest.attrs["manifest_path"]
    row = manifest.loc[manifest["trace_id"] == "kr_10sccm_10a"].iloc[0]
    trace = load_trace(row)
    assert trace.point_count == 250
    assert trace.bias_voltage[0] < trace.bias_voltage[-1]
    assert trace.gas == "Kr"


def test_load_trace_from_sample_manifest_hdf5():
    manifest = load_manifest(PROJECT_ROOT / "sample_data" / "local_manifest.csv")
    manifest["manifest_path"] = manifest.attrs["manifest_path"]
    row = manifest.loc[manifest["trace_id"] == "lp_33_34deg_150v"].iloc[0]
    trace = load_trace(row)
    assert trace.point_count == 234
    assert trace.bias_voltage.shape == trace.probe_current.shape


def test_adapter_loads_scitech_style_txt_trace():
    manifest = infer_local_manifest_rows(PROJECT_ROOT, gas="Kr", **FALLBACK_PROBE)
    row = manifest.loc[manifest["trace_id"] == "lp_15sccm_10a1"].iloc[0]
    trace = load_trace(row)
    assert trace.point_count == 250
    assert trace.gas == "Kr"
    assert trace.probe_area_m2 == FALLBACK_PROBE["probe_area_m2"]


def test_parse_discharge_filename_supports_p_decimal_notation():
    metadata = parse_discharge_filename(Path("Kr_15sccm_12p5A.csv"))
    assert metadata["gas"] == "Kr"
    assert metadata["flow_sccm"] == 15.0
    assert metadata["discharge_current_a"] == 12.5


def test_adapter_requires_explicit_probe_metadata():
    try:
        infer_local_manifest_rows(PROJECT_ROOT)
    except ValueError as exc:
        assert "requires explicit probe metadata" in str(exc)
    else:
        raise AssertionError("Expected infer_local_manifest_rows to require explicit probe metadata.")
