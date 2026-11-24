"""用解析 Airy 截面和独立单点积分验证 XZ/YZ 物理位置。"""

import numpy as np
import pytest
from scipy.special import j1

from optics import Numerics, OpticalConfig, ZernikeTerm
from optics.axial import fixed_point_field
from optics.simulation import simulate_focus


def test_sections_match_axial_line_and_analytic_focal_airy():
    result = simulate_focus(OpticalConfig(), Numerics(grid_size=128, padding_factor=2,
                            axial_samples=65, section_samples=129))
    arrays = result.arrays
    center = len(arrays["xz_x_coordinate_m"])//2
    np.testing.assert_allclose(arrays["xz_intensity_W_m2"][:, center],
                               arrays["axial_intensity_W_m2"], rtol=1e-11, atol=1e-5)
    np.testing.assert_allclose(arrays["yz_intensity_W_m2"], arrays["xz_intensity_W_m2"],
                               rtol=1e-10, atol=1e-5)
    at_focus = np.argmin(abs(arrays["section_z_offset_m"]))
    assert arrays["section_z_offset_m"][at_focus] == 0
    arguments = 2*np.pi*result.config.effective_radius*arrays["xz_x_coordinate_m"]/(
        result.config.wavelength*result.config.focal_length
    )
    expected = np.ones_like(arguments)
    np.divide(2*j1(arguments), arguments, out=expected, where=arguments != 0)
    line = arrays["xz_intensity_W_m2"][at_focus]
    np.testing.assert_allclose(line/line[center], expected**2, atol=0.001)


def test_aberrated_sections_use_fixed_physical_planes_and_shared_scale():
    config = OpticalConfig(aberrations=(ZernikeTerm(3, 1, 0.2), ZernikeTerm(2, -2, 0.1)))
    result = simulate_focus(config, Numerics(grid_size=128, padding_factor=4,
                            axial_samples=65, section_samples=65), aberrated=True)
    arrays = result.arrays
    support = arrays["pupil_support"]
    pupil = support*np.exp(1j*arrays["pupil_phase_unwrapped_rad"])/np.sqrt(np.pi*config.beam_radius**2)
    center = len(arrays["xz_x_coordinate_m"])//2
    assert arrays["xz_x_coordinate_m"][center]*1e6 == pytest.approx(result.metrics["peak_x_um"])
    assert arrays["yz_y_coordinate_m"][center]*1e6 == pytest.approx(result.metrics["peak_y_um"])
    for plane in ("xz", "yz"):
        np.testing.assert_allclose(arrays[f"{plane}_intensity_W_m2"][:, center],
                                   arrays["axial_intensity_W_m2"], rtol=1e-10, atol=1e-5)
    for row, column in ((7, 16), (20, 40)):
        z = config.focal_length+arrays["section_z_offset_m"][row]
        for plane, x, y in (
            ("xz", arrays["xz_x_coordinate_m"][column], result.metrics["peak_y_um"]*1e-6),
            ("yz", result.metrics["peak_x_um"]*1e-6, arrays["yz_y_coordinate_m"][column]),
        ):
            field = fixed_point_field(pupil, arrays["pupil_coordinate_m"], config.wavelength, config.focal_length, x, y)
            assert arrays[f"{plane}_intensity_W_m2"][row, column] == pytest.approx(abs(field(z))**2, rel=1e-10)
    expected_scale = max(arrays["xz_intensity_W_m2"].max(), arrays["yz_intensity_W_m2"].max())
    assert result.metadata["section_normalization_W_m2"] == expected_scale
