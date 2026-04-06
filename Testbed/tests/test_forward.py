from __future__ import annotations

import numpy as np

from orsini_lp.forward import electron_current, normalized_eedf


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

