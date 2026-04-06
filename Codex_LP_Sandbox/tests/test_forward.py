from __future__ import annotations

import numpy as np

from orsini_lp.forward import (
    cylindrical_probe_area,
    cylindrical_transitional_coefficients,
    electron_current,
    ion_current_oml_cylindrical,
    ion_current_transitional_cylindrical,
    normalized_eedf,
)


def test_normalized_eedf_normalizes_and_preserves_mean_energy():
    energy = np.linspace(0.0, 40.0, 4000)
    for p_value in (1.0, 1.5, 2.0):
        eedf = normalized_eedf(energy, te_ev=3.0, p_value=p_value)
        assert np.isclose(np.trapz(eedf, energy), 1.0, rtol=2.5e-4)
        assert np.isclose(np.trapz(energy * eedf, energy), 4.5, rtol=3.0e-4)


def test_electron_current_is_monotonic_and_continuous_near_plasma_potential():
    bias = np.linspace(-6.0, 5.0, 240)
    current = electron_current(
        bias_voltage=bias,
        probe_area_m2=1.0e-6,
        plasma_potential_v=2.0,
        electron_density_m3=5.0e17,
        electron_temperature_ev=3.5,
        p_value=1.5,
    )
    left_mask = bias <= 2.0
    assert np.all(np.diff(current[left_mask]) >= -1.0e-12)

    near_bias = np.array([1.999, 2.0, 2.001])
    near_current = electron_current(
        bias_voltage=near_bias,
        probe_area_m2=1.0e-6,
        plasma_potential_v=2.0,
        electron_density_m3=5.0e17,
        electron_temperature_ev=3.5,
        p_value=1.5,
    )
    assert abs(near_current[1] - near_current[2]) < 1.0e-12
    assert near_current[1] >= near_current[0]


def test_cylindrical_probe_area_matches_specification():
    radius = 1.5e-4
    length = 2.8e-3
    expected = np.pi * radius * (radius + 2.0 * length)
    assert np.isclose(cylindrical_probe_area(radius, length), expected)


def test_oml_and_transitional_currents_are_negative_below_plasma_potential():
    bias = np.linspace(-8.0, 8.0, 100)
    oml = ion_current_oml_cylindrical(
        bias_voltage=bias,
        probe_area_m2=1.5e-6,
        plasma_potential_v=5.0,
        ion_density_m3=4.0e17,
        ion_mass_kg=83.798 * 1.66053906660e-27,
    )
    transitional = ion_current_transitional_cylindrical(
        bias_voltage=bias,
        probe_area_m2=1.5e-6,
        probe_radius_m=4.0e-4,
        plasma_potential_v=5.0,
        electron_density_m3=5.0e16,
        electron_temperature_ev=3.5,
        ion_mass_kg=83.798 * 1.66053906660e-27,
    )
    assert np.all(oml[bias < 5.0] <= 0.0)
    assert np.all(transitional[bias < 5.0] <= 0.0)
    assert np.allclose(oml[bias >= 5.0], 0.0)
    assert np.allclose(transitional[bias >= 5.0], 0.0)


def test_transitional_coefficients_are_positive_in_valid_regime():
    a_param, b_param = cylindrical_transitional_coefficients(10.0)
    assert a_param > 0.0
    assert b_param > 0.0
