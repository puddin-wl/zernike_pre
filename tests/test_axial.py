"""用圆孔径解析轴向场、离焦焦移及 FFT 焦面一致性验证轴向扫描。"""

from dataclasses import replace
import numpy as np
import pytest

from optics import Numerics, OpticalConfig, ZernikeTerm
from optics.axial import fixed_point_field
from optics.simulation import simulate_focus


def test_axial_integral_matches_analytic_uniform_disk_at_physical_z():
    size = 256
    radius, wavelength, focal = 2e-3, 550e-9, 0.05
    coordinate = (np.arange(size)-size//2)*5e-3/size
    support = np.hypot(coordinate[:, None], coordinate[None, :]) <= radius
    pupil = support.astype(complex)
    evaluate = fixed_point_field(pupil, coordinate, wavelength, focal, 0, 0)
    z = focal+np.array([-1e-3, -0.3e-3, 0, 0.3e-3, 1e-3])
    actual = np.array([abs(evaluate(position))**2 for position in z])
    q = radius**2/(2*wavelength)*(1/z-1/focal)
    expected = (np.pi*radius**2/(wavelength*z))**2*np.sinc(q)**2
    np.testing.assert_allclose(actual/actual[2], expected/expected[2], atol=0.0007)


def test_axial_fwhm_minima_and_grid_convergence():
    config = OpticalConfig()
    coarse = simulate_focus(config, Numerics(grid_size=128, padding_factor=2, axial_samples=65))
    fine = simulate_focus(config, Numerics(grid_size=256, padding_factor=2, axial_samples=129))
    expected_width = 1.771785883*config.wavelength/config.effective_na**2*1e6
    assert fine.metrics["axial_fwhm_um"] == pytest.approx(expected_width, rel=0.002)
    assert coarse.metrics["axial_fwhm_um"] == pytest.approx(fine.metrics["axial_fwhm_um"], rel=0.002)
    # 完整 z 依赖使前后第一零点不严格对称，独立使用解析式检验。
    curvature = 2*config.wavelength/config.effective_radius**2
    left = (1/(1/config.focal_length+curvature)-config.focal_length)*1e6
    right = (1/(1/config.focal_length-curvature)-config.focal_length)*1e6
    assert fine.metrics["axial_first_min_left_um"] == pytest.approx(left, rel=0.002)
    assert fine.metrics["axial_first_min_right_um"] == pytest.approx(right, rel=0.002)
    assert fine.metrics["axial_rayleigh_um"] == pytest.approx((right-left)/2, rel=0.002)


def test_fixed_line_at_focal_plane_matches_same_fft_peak():
    config = OpticalConfig(aberrations=(ZernikeTerm(3, 1, 0.15), ZernikeTerm(2, 0, 0.1)))
    result = simulate_focus(config, Numerics(grid_size=128, padding_factor=4, axial_samples=65), aberrated=True)
    arrays = result.arrays
    at_focus = int(np.argmin(abs(arrays["axial_offset_m"])))
    assert arrays["axial_offset_m"][at_focus] == 0
    assert arrays["axial_intensity_W_m2"][at_focus] == pytest.approx(result.metrics["peak_intensity_W_m2"], rel=1e-10)
    assert result.metrics["axial_fixed_x_um"] == result.metrics["peak_x_um"]
    assert result.metrics["axial_fixed_y_um"] == result.metrics["peak_y_um"]
    assert result.metrics["observation_distance_mm"] == 50


def test_positive_zernike_defocus_shifts_axial_peak_without_refocusing_image():
    coefficient = 0.15
    config = OpticalConfig(aberrations=(ZernikeTerm(2, 0, coefficient),))
    result = simulate_focus(config, Numerics(grid_size=128, padding_factor=4, axial_samples=129), aberrated=True)
    shifted_focus = 1/(1/config.focal_length-4*np.sqrt(3)*coefficient*config.wavelength/config.beam_radius**2)
    # 几何相位抵消位置附近还有 1/z² 幅度引起的轻微主峰偏移。
    assert result.metrics["axial_peak_offset_um"] == pytest.approx((shifted_focus-config.focal_length)*1e6, abs=5)
    assert result.metrics["observation_distance_mm"] == config.focal_length_mm
    arrays = result.arrays
    at_focus = np.argmin(abs(arrays["axial_offset_m"]))
    assert arrays["axial_intensity_W_m2"][at_focus] < result.metrics["axial_peak_intensity_W_m2"]


def test_short_range_returns_missing_intersections_without_theory_substitution():
    result = simulate_focus(OpticalConfig(), Numerics(grid_size=128, padding_factor=2,
                            axial_samples=33, axial_half_range_um=5))
    assert result.metrics["axial_fwhm_um"] is None
    assert result.metrics["axial_rayleigh_um"] is None
    assert any("半高交点" in message for message in result.diagnostics)


def test_peak_outside_scan_range_is_not_reported_as_located_focus():
    config = OpticalConfig(aberrations=(ZernikeTerm(2, 0, 0.3),))
    result = simulate_focus(config, Numerics(grid_size=128, padding_factor=2,
                            axial_samples=33, axial_half_range_um=50), aberrated=True)
    assert result.metrics["axial_peak_offset_um"] is None
    assert result.metrics["axial_fwhm_um"] is None
    assert any("范围边界" in message for message in result.diagnostics)


@pytest.mark.parametrize("updates", [{"axial_samples": 32}, {"axial_samples": 64},
                                   {"axial_half_range_um": 0}, {"axial_half_range_um": float("nan")}])
def test_invalid_axial_sampling_is_rejected(updates):
    with pytest.raises(ValueError):
        replace(Numerics(), **updates)


def test_scan_cannot_cross_lens_plane():
    with pytest.raises(ValueError, match="小于"):
        simulate_focus(OpticalConfig(), Numerics(grid_size=128, padding_factor=2,
                       axial_half_range_um=50000))
