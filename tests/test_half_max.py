"""多峰半高区间、整体跨度和最高峰切换，用独立径向积分验证球差案例。"""

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

from optics import Numerics, OpticalConfig, ZernikeTerm
from optics import axial
from optics.metrics import fwhm, half_max_regions
from optics.simulation import simulate_focus


def test_disconnected_half_height_regions_are_not_one_local_fwhm():
    x = np.arange(9, dtype=float)
    y = np.array([0, 1, 0, 0, 2, 0, 0, 1.5, 0])
    profile = half_max_regions(x, y)
    assert len(profile["intervals"]) == 3
    assert profile["intervals"][0] == (1, 1)  # 恰好触及 50% 的点也单独记录。
    assert profile["outer_left"] == 1
    assert profile["outer_right"] == pytest.approx(7+1/3)
    assert profile["outer_span"] == pytest.approx(6+1/3)
    local = fwhm(x, y)
    assert local["width"] == 1
    assert local["left"] == 3.5
    assert local["right"] == 4.5
    assert local["half_height_regions"] == 3


def test_clipped_outer_interval_does_not_hide_valid_local_width():
    x = np.arange(7, dtype=float)
    y = np.array([0.8, 0.8, 0, 0, 1, 0, 0])
    profile = half_max_regions(x, y)
    assert profile["outer_left"] is None
    assert profile["outer_span"] is None
    assert profile["intervals"][0] == (None, pytest.approx(1.375))
    assert fwhm(x, y)["width"] == 1
    assert half_max_regions(x, np.zeros_like(x))["intervals"] == []


def test_half_height_is_not_half_prominence_above_local_valley():
    x = np.arange(5, dtype=float)
    y = np.array([0.4, 0.4, 1, 0.4, 0.4])
    # 绝对半高为 0.5，而不是谷底 0.4 上方的半突出度高度 0.7。
    assert fwhm(x, y)["width"] == pytest.approx(5/3)


def test_all_detected_peak_heights_are_refined_before_global_selection(monkeypatch):
    def field_factory(*args, **kwargs):
        def evaluate(offset):
            x = offset*1e6
            value = 1/(1+((x+5)/0.8)**2) + 1.2/(1+((x-5.35)/0.15)**2)
            return complex(np.sqrt(value))
        return evaluate

    monkeypatch.setattr(axial, "normalized_point_field", field_factory)
    metrics, _, _, arrays = axial.scan_axial(
        np.ones((3, 3), dtype=complex), np.array([-1, 0, 1]),
        OpticalConfig(numerical_aperture=0.1),
        Numerics(axial_samples=33, axial_half_range_um=16), 0, 0,
    )
    # 整数位置的粗采样会误选 -5 μm 的宽峰，细化后真正最高峰在 +5.35 μm。
    assert metrics["axial_peak_offset_um"] == pytest.approx(5.35, abs=0.01)
    assert metrics["axial_peak_intensity_W_m2"] > 1.2
    assert np.max(arrays["axial_intensity_W_m2"]) == metrics["axial_peak_intensity_W_m2"]


@pytest.mark.parametrize("coefficient", [0.2, 0.4, 0.41, 0.6])
def test_spherical_aberration_half_max_intervals_match_independent_radial_integral(coefficient):
    # t=rho² 后面积积分退化成 [0,1] 上的积分，与二维离散瞳面实现独立。
    roots, weights = leggauss(160)
    t, weights = (roots+1)/2, weights/2
    phase = 2*np.pi*coefficient*np.sqrt(5)*(6*t*t-6*t+1)

    def intensity(z_um):
        return float(abs(np.sum(weights*np.exp(1j*(phase-np.pi*0.67**2*z_um/1.064*t))))**2)

    z = np.linspace(-45, 45, 3601)
    values = np.array([intensity(position) for position in z])
    peaks = find_peaks(values)[0]
    refined_heights = [-minimize_scalar(lambda p:-intensity(p), bounds=(z[i-1],z[i+1]),
                                        method="bounded").fun for i in peaks]
    half = max(refined_heights)/2
    indices = np.flatnonzero(np.diff(values >= half))
    crossings = [brentq(lambda p:intensity(p)-half,z[i],z[i+1]) for i in indices]
    expected_intervals = np.array(crossings).reshape(-1, 2)
    config = OpticalConfig(wavelength_nm=1064, numerical_aperture=0.67,
                           aberrations=(ZernikeTerm(4,0,coefficient),))
    result = simulate_focus(config, Numerics(grid_size=256, padding_factor=4,
                            axial_samples=161, section_samples=33), aberrated=True)
    actual = result.metrics
    assert actual["axial_half_height_regions"] == (3 if coefficient in (0.4, 0.41) else 1)
    np.testing.assert_allclose(actual["axial_half_max_intervals_um"], expected_intervals, atol=0.04)
    assert actual["axial_half_max_span_um"] == pytest.approx(crossings[-1]-crossings[0], abs=0.04)
    assert actual["axial_fwhm_um"] == actual["axial_half_max_span_um"]
    assert actual["axial_fwhm_um"] == pytest.approx(crossings[-1]-crossings[0], abs=0.04)
    if coefficient == 0.4:
        assert actual["axial_peak_offset_um"] == pytest.approx(0, abs=0.04)
        assert actual["axial_local_fwhm_um"] == pytest.approx(7.55, abs=0.04)
        assert actual["axial_fwhm_um"] > 2*actual["axial_local_fwhm_um"]
    elif coefficient == 0.41:
        assert abs(actual["axial_peak_offset_um"]) == pytest.approx(14.22, abs=0.04)
        assert actual["axial_local_fwhm_um"] == pytest.approx(14.83, abs=0.04)
        assert actual["axial_fwhm_um"] > 2*actual["axial_local_fwhm_um"]
    else:
        # 0.2、0.6 的内部谷底均高于半高；新旧宽度必须保持一致。
        assert actual["axial_fwhm_um"] == actual["axial_local_fwhm_um"]
