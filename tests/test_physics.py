"""以 Airy 解析解、Parseval 和解析相位验证物理模型。"""

from dataclasses import replace
import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest
from scipy.special import j1

from optics import Numerics, OpticalConfig, ZernikeTerm, calculate_resolution
from optics.metrics import first_dark_ring, fwhm
from optics.simulation import simulate_focus
from optics.zernike import zernike


@pytest.fixture(scope="module")
def plain():
    return simulate_focus(OpticalConfig(), Numerics(grid_size=256, padding_factor=8))


def test_air_tissue_and_underfilled_resolution():
    config = OpticalConfig()
    result = calculate_resolution(config)
    air, tissue = result["results"]["initial_medium"], result["results"]["sample_medium"]
    assert air["effective_na"] == pytest.approx(0.04)
    assert air["lateral_rayleigh_um"] == pytest.approx(8.3875)
    assert air["axial_resolution_um"] == pytest.approx(687.5)
    assert tissue["lateral_rayleigh_um"] == air["lateral_rayleigh_um"]
    assert tissue["axial_resolution_um"] == pytest.approx(941.875)
    underfilled = calculate_resolution(replace(config, beam_diameter_mm=2))
    assert underfilled["results"]["initial_medium"]["lateral_rayleigh_um"] == pytest.approx(2*8.3875)
    assert underfilled["results"]["initial_medium"]["axial_resolution_um"] == pytest.approx(4*687.5)
    assert replace(config, lens_diameter_mm=6).beam_radius == pytest.approx(3e-3)


def test_airy_profile_and_fwhm_in_physical_coordinates(plain):
    points = plain.arrays["focal_coordinate_m"]
    values = plain.arrays["x_intensity_W_m2"]
    argument = 2*np.pi*plain.config.effective_radius*points/(
        plain.config.wavelength*plain.config.focal_length
    )
    amplitude = np.ones_like(argument)
    np.divide(2*j1(argument), argument, out=amplitude, where=argument != 0)
    np.testing.assert_allclose(values/values.max(), amplitude**2, atol=0.001)
    expected_um = 0.514496985*plain.config.wavelength/plain.config.effective_na*1e6
    assert plain.metrics["lateral_fwhm_x_um"] == pytest.approx(expected_um, rel=0.004)
    assert plain.metrics["lateral_fwhm_y_um"] == pytest.approx(expected_um, rel=0.004)
    # 数值第一暗环由同一次 FFT 的场零点提取，不将理论公式当作测量值。
    expected_radius_um = 0.6098349456*plain.config.wavelength/plain.config.effective_na*1e6
    assert plain.metrics["lateral_rayleigh_um"] == pytest.approx(expected_radius_um, rel=0.001)
    assert plain.metrics["lateral_rayleigh_x_um"] == pytest.approx(plain.metrics["lateral_rayleigh_y_um"])


def test_absolute_power_and_fft_scale(plain):
    assert plain.metrics["transmitted_power_W"] == pytest.approx(1, abs=0.001)
    assert plain.metrics["full_fft_power_W"] == pytest.approx(plain.metrics["transmitted_power_W"], rel=1e-11)
    assert plain.metadata["focal_spacing_um"] == pytest.approx(0.6875)


def test_zernike_area_orthonormality():
    roots, weights = leggauss(60)
    rho = np.sqrt((roots[:, None]+1)/2)
    theta = np.arange(128)[None, :]*2*np.pi/128
    modes = [(n, m) for n in range(7) for m in range(-n, n+1, 2)]
    basis = np.array([zernike(n, m, rho, theta)+np.zeros_like(theta) for n, m in modes])
    gram = np.einsum("irk,jrk,r->ij", basis, basis, weights/2)/theta.size
    np.testing.assert_allclose(gram, np.eye(len(modes)), atol=5e-13)


def test_zero_phase_programs_match_exactly(plain):
    zero = simulate_focus(OpticalConfig(), plain.numerics, aberrated=True)
    np.testing.assert_array_equal(plain.arrays["focal_intensity_W_m2"], zero.arrays["focal_intensity_W_m2"])
    assert all(plain.metrics[key] == value for key, value in zero.metrics.items())
    assert zero.program == "zernike_focus"


def test_uniform_pupil_piston_cannot_change_intensity(plain):
    config = OpticalConfig(aberrations=(ZernikeTerm(0, 0, 0.31),))
    loaded = simulate_focus(config, plain.numerics, aberrated=True)
    np.testing.assert_allclose(loaded.arrays["focal_intensity_W_m2"],
                               plain.arrays["focal_intensity_W_m2"], rtol=1e-11, atol=1e-5)
    assert loaded.metadata["full_beam_wavefront_rms_waves"] == 0
    assert loaded.metadata["sampled_transmitted_wavefront_rms_waves"] < 1e-15


def test_defocus_retains_fixed_focal_plane_and_matches_on_axis_integral(plain):
    coefficient = 0.15
    loaded = simulate_focus(OpticalConfig(aberrations=(ZernikeTerm(2, 0, coefficient),)),
                            plain.numerics, aberrated=True)
    middle = len(loaded.arrays["focal_coordinate_m"])//2
    ratio = loaded.arrays["focal_intensity_W_m2"][middle, middle]/plain.metrics["peak_intensity_W_m2"]
    assert ratio == pytest.approx(np.sinc(2*np.sqrt(3)*coefficient)**2, rel=0.002)
    assert loaded.metrics["observation_distance_mm"] == 50


def test_overfill_clips_amplitude_without_rescaling_zernike():
    config = OpticalConfig(lens_diameter_mm=6, beam_diameter_mm=8,
                           aberrations=(ZernikeTerm(2, 0, 0.1),))
    loaded = simulate_focus(config, Numerics(grid_size=256, padding_factor=2), aberrated=True)
    arrays = loaded.arrays
    coordinate = arrays["pupil_coordinate_m"]
    rho = np.hypot(coordinate[:, None], coordinate[None, :])/4e-3
    support = arrays["pupil_support"]
    np.testing.assert_allclose(arrays["pupil_phase_unwrapped_rad"][support],
                               2*np.pi*0.1*np.sqrt(3)*(2*rho[support]**2-1))
    assert rho[support].max() == pytest.approx(0.75, abs=0.002)
    assert loaded.metrics["transmitted_power_W"] == pytest.approx((6/8)**2, abs=0.001)
    assert loaded.metadata["full_beam_wavefront_rms_waves"] == pytest.approx(0.1)
    assert loaded.metadata["sampled_transmitted_wavefront_rms_waves"] == pytest.approx(0.1*(6/8)**2, rel=0.002)


def test_underfill_doubles_focal_width(plain):
    result = simulate_focus(OpticalConfig(beam_diameter_mm=2), plain.numerics)
    assert result.metrics["lateral_fwhm_x_um"] == pytest.approx(2*plain.metrics["lateral_fwhm_x_um"])
    assert result.metrics["effective_na"] == pytest.approx(0.02)
    assert result.metrics["lateral_rayleigh_um"] == pytest.approx(2*plain.metrics["lateral_rayleigh_um"])


def test_zernike_tilt_moves_focus_by_analytic_amount(plain):
    config = OpticalConfig(aberrations=(ZernikeTerm(1, 1, 0.4),))
    result = simulate_focus(config, plain.numerics, aberrated=True)
    expected_um = 2*0.4*config.wavelength*config.focal_length/config.beam_radius*1e6
    assert result.metrics["peak_x_um"] == pytest.approx(expected_um, abs=0.35)
    assert result.metrics["peak_y_um"] == 0
    assert result.metrics["lateral_fwhm_x_um"] == pytest.approx(plain.metrics["lateral_fwhm_x_um"], rel=0.004)


def test_one_fft_per_focus_and_zero_phase_program_ignores_terms(monkeypatch):
    from optics import fourier
    original = fourier.fft2
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fourier, "fft2", counted)
    result = simulate_focus(OpticalConfig(aberrations=(ZernikeTerm(3, 1, 0.5),)),
                            Numerics(grid_size=128, padding_factor=2))
    assert len(calls) == 1
    assert result.config.aberrations == ()
    assert np.all(result.arrays["pupil_phase_unwrapped_rad"] == 0)
    assert "aberrations" not in result.summary()["parameters"]


def test_missing_fwhm_intersection_is_not_extrapolated():
    assert fwhm(np.arange(5), np.array([0, 0.2, 1, 0.9, 0.8]))["width"] is None


def test_first_dark_ring_uses_field_zero_without_optical_parameters():
    points = np.linspace(-3, 3, 49)
    # 独立的有符号场截面，零点为 ±sqrt(2)，附加整体相位不改变零点。
    field = (1-points**2/2)*np.exp(0.7j)
    # 只保留主瓣和零点附近，避免此测试多项式的远端大值被选为主峰。
    selected = abs(points) <= 1.75
    ring = first_dark_ring(points[selected], field[selected])
    assert ring["radius"] == pytest.approx(np.sqrt(2), abs=1e-10)
    assert ring["left"] == pytest.approx(-np.sqrt(2), abs=1e-10)
    assert ring["right"] == pytest.approx(np.sqrt(2), abs=1e-10)


def test_missing_first_dark_ring_is_not_replaced_with_theory():
    result = simulate_focus(OpticalConfig(), Numerics(grid_size=128, padding_factor=8, image_half_width_um=5))
    assert result.metrics["lateral_rayleigh_um"] is None
    assert result.summary()["results"]["lateral_rayleigh_um"] is None
    assert any("第一暗环" in message for message in result.diagnostics)


def test_requested_view_must_contain_global_peak():
    with pytest.raises(ValueError, match="全局主峰"):
        simulate_focus(OpticalConfig(aberrations=(ZernikeTerm(1, 1, 2.0),)),
                       Numerics(grid_size=128, padding_factor=4, image_half_width_um=5), aberrated=True)


@pytest.mark.parametrize("updates", [{"wavelength_nm": float("nan")}, {"focal_length_mm": 0},
                                   {"beam_diameter_mm": -1}, {"lens_diameter_mm": float("inf")}])
def test_invalid_physical_inputs_fail(updates):
    with pytest.raises(ValueError):
        OpticalConfig(**updates)
