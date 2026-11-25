"""直接 NA 输入：尺寸独立性、近轴解析解、功率和两种焦区切面一致性。"""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import j1

from optics import Numerics, OpticalConfig, ResolutionConfig, ZernikeTerm, calculate_resolution
from optics.axial import normalized_point_field
from optics.cli import _parser
from optics.simulation import simulate_focus


@pytest.fixture(scope="module")
def na_focus():
    return simulate_focus(
        OpticalConfig(wavelength_nm=1064, numerical_aperture=0.67,
                      lens_diameter_mm=None, focal_length_mm=None),
        Numerics(grid_size=256, padding_factor=8, axial_samples=129, section_samples=65),
    )


@pytest.mark.parametrize("na", [0, -0.1, float("nan"), float("inf"), 1.0, 1.1])
def test_na_must_be_finite_positive_and_below_initial_index(na):
    with pytest.raises(ValueError, match="NA"):
        OpticalConfig(numerical_aperture=na)


def test_direct_na_is_not_multiplied_by_immersion_index_and_ignores_geometry():
    config = ResolutionConfig(wavelength_nm=1064, numerical_aperture=0.67,
                              immersion_medium="oil", sample_refractive_index=2.39,
                              lens_diameter_mm=None, focal_length_mm=0, beam_diameter_mm=float("nan"))
    result = calculate_resolution(config)
    initial, sample = result["results"].values()
    assert config.effective_na == 0.67
    assert initial["lateral_rayleigh_um"] == pytest.approx(0.61*1.064/0.67)
    assert initial["axial_resolution_um"] == pytest.approx(2*1.515*1.064/0.67**2)
    assert sample["axial_resolution_um"] == pytest.approx(2*2.39*1.064/0.67**2)
    assert sample["lateral_rayleigh_um"] == initial["lateral_rayleigh_um"]
    assert result["effective_diameter_mm"] is None
    assert result["parameters"]["input_mode"] == "numerical_aperture"
    for key in ("lens_diameter_mm", "focal_length_mm", "beam_diameter_mm"):
        assert result["parameters"][key] is None
    json.dumps(result, allow_nan=False)
    assert ResolutionConfig(numerical_aperture=1.2, immersion_medium="oil").effective_na == 1.2
    with pytest.raises(ValueError, match="NA"):
        ResolutionConfig(numerical_aperture=1.515, immersion_medium="oil")
    with pytest.raises(ValueError, match="全反射"):
        calculate_resolution(replace(config, sample_refractive_index=0.5))


def test_normalized_focal_airy_scale_and_power(na_focus):
    result = na_focus
    points = result.arrays["focal_coordinate_m"]
    argument = 2*np.pi*0.67*points/result.config.wavelength
    field = np.ones_like(argument)
    np.divide(2*j1(argument), argument, out=field, where=argument != 0)
    line = result.arrays["x_intensity_W_m2"]
    np.testing.assert_allclose(line/line.max(), field**2, atol=0.001)
    assert result.metrics["lateral_rayleigh_um"] == pytest.approx(0.6098349456*1.064/0.67, rel=0.001)
    assert result.metrics["lateral_fwhm_x_um"] == pytest.approx(0.514496985*1.064/0.67, rel=0.004)
    assert result.metrics["transmitted_power_W"] == pytest.approx(1, abs=0.001)
    assert result.metrics["full_fft_power_W"] == pytest.approx(result.metrics["transmitted_power_W"], rel=1e-11)
    assert result.metrics["geometric_transmission"] == 1
    assert result.metrics["observation_distance_mm"] is None
    assert result.metadata["zernike_normalization_radius_normalized"] == 1
    assert result.metadata["pupil_coordinate_units"] == "dimensionless"
    assert "pupil_coordinate_m" not in result.arrays
    assert "axial_z_m" not in result.arrays
    assert "no lens dimensions or focal length" in result.summary()["model"]
    assert any("近轴" in message for message in result.diagnostics)


def test_na_axial_sinc_fwhm_minima_and_sections(na_focus):
    result, arrays = na_focus, na_focus.arrays
    offsets = arrays["axial_offset_m"]
    peak = result.metrics["axial_peak_intensity_W_m2"]
    expected = np.sinc(0.67**2*offsets/(2*result.config.wavelength))**2
    np.testing.assert_allclose(arrays["axial_intensity_W_m2"]/peak, expected, atol=0.001)
    assert result.metrics["axial_peak_offset_um"] == pytest.approx(0, abs=1e-7)
    assert result.metrics["axial_rayleigh_um"] == pytest.approx(2*1.064/0.67**2, rel=0.002)
    assert result.metrics["axial_fwhm_um"] == pytest.approx(1.771785883*1.064/0.67**2, rel=0.002)
    center = len(arrays["xz_x_coordinate_m"])//2
    for plane in ("xz", "yz"):
        np.testing.assert_allclose(arrays[f"{plane}_intensity_W_m2"][:, center],
                                   arrays["axial_intensity_W_m2"], rtol=1e-10, atol=1e-2)
    at_focus = np.argmin(abs(offsets))
    assert offsets[at_focus] == 0
    assert arrays["axial_intensity_W_m2"][at_focus] == pytest.approx(result.metrics["peak_intensity_W_m2"], rel=1e-11)


def test_na_mode_is_independent_of_every_ignored_dimension(na_focus):
    config = replace(na_focus.config, lens_diameter_mm=0, focal_length_mm=float("nan"), beam_diameter_mm=-1)
    other = simulate_focus(config, na_focus.numerics)
    assert other.summary() == na_focus.summary()
    for key in other.arrays:
        np.testing.assert_array_equal(other.arrays[key], na_focus.arrays[key])


def test_na_zernike_defocus_and_independent_off_axis_section_integral():
    coefficient = 0.15
    config = OpticalConfig(wavelength_nm=1064, numerical_aperture=0.67,
                           aberrations=(ZernikeTerm(2, 0, coefficient), ZernikeTerm(1, 1, 0.1)))
    result = simulate_focus(config, Numerics(grid_size=128, padding_factor=8,
                            axial_samples=97, section_samples=65), aberrated=True)
    expected_shift_um = 4*np.sqrt(3)*coefficient*1.064/0.67**2
    assert result.metrics["axial_peak_offset_um"] == pytest.approx(expected_shift_um, rel=0.01)
    assert result.metrics["peak_x_um"] == pytest.approx(2*0.1*1.064/0.67, abs=0.08)
    expected_rms = np.hypot(coefficient, 0.1)
    assert result.metadata["full_beam_wavefront_rms_waves"] == pytest.approx(expected_rms)
    assert result.metadata["sampled_transmitted_wavefront_rms_waves"] == pytest.approx(expected_rms, rel=0.004)
    arrays = result.arrays
    pupil = arrays["pupil_support"]*np.exp(1j*arrays["pupil_phase_unwrapped_rad"])/np.sqrt(np.pi)
    for plane in ("xz", "yz"):
        row, column = 14, 29
        x = arrays["xz_x_coordinate_m"][column] if plane == "xz" else result.metrics["peak_x_um"]*1e-6
        y = arrays["yz_y_coordinate_m"][column] if plane == "yz" else result.metrics["peak_y_um"]*1e-6
        field = normalized_point_field(pupil, arrays["pupil_coordinate_normalized"],
                                       config.wavelength, config.effective_na, x, y)
        expected = abs(field(arrays["section_z_offset_m"][row]))**2
        assert arrays[f"{plane}_intensity_W_m2"][row, column] == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("program", ["resolution", "fourier_focus", "zernike_focus"])
def test_na_cli_and_long_alias_are_available_in_all_programs(program):
    for flag in ("--na", "--numerical-aperture"):
        assert _parser(program).parse_args([flag, "0.67"]).numerical_aperture == 0.67
    assert _parser(program).parse_args([]).numerical_aperture is None


def test_both_na_focus_programs_use_one_fft_and_zero_phase_matches(monkeypatch):
    from optics import fourier
    original = fourier.fft2
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fourier, "fft2", counted)
    config = OpticalConfig(numerical_aperture=0.1)
    numerics = Numerics(grid_size=128, padding_factor=2, axial_samples=33, section_samples=33)
    plain = simulate_focus(config, numerics)
    assert len(calls) == 1
    zero = simulate_focus(config, numerics, aberrated=True)
    assert len(calls) == 2
    for key in ("focal_intensity_W_m2", "axial_intensity_W_m2", "xz_intensity_W_m2", "yz_intensity_W_m2"):
        np.testing.assert_array_equal(plain.arrays[key], zero.arrays[key])


def test_na_theory_cli_ignores_geometry_without_loading_numeric_libraries(tmp_path):
    code = (
        "import runpy,sys; "
        f"sys.argv=['resolution.py','--na','0.67','--wavelength-nm','1064',"
        f"'--sample-refractive-index','2.39','--lens-diameter-mm','0','--focal-length-mm','0',"
        f"'--beam-diameter-mm','-1','--output',{str(tmp_path)!r}]; "
        "runpy.run_path('resolution.py',run_name='__main__'); "
        "assert not any(n in sys.modules for n in ('matplotlib','numpy','scipy'))"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    assert "忽略" in completed.stdout
    assert "工作焦距：" not in completed.stdout
    summary = json.loads((tmp_path/"summary.json").read_text(encoding="utf-8"))
    assert summary["parameters"]["input_mode"] == "numerical_aperture"
    assert summary["results"]["initial_medium"]["axial_resolution_um"] == pytest.approx(4.740476720873246)
    assert summary["results"]["sample_medium"]["axial_resolution_um"] == pytest.approx(11.329739362887056)
