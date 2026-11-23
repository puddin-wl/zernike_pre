"""用独立解析解验证传播及指标，而不是复制程序实现。"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from optics import Numerics, OpticalConfig, ZernikeTerm, simulate_diffraction, simulate_aberration
from optics.metrics import encircled_energy_radius, fwhm, peak_coordinate
from optics.propagation import FocusedField, fresnel_propagate, power
from optics.simulation import _prepare, simulate
from optics.zernike import wavefront, zernike


def grid(n=512, window=8e-3):
    coordinate = (np.arange(n)-n//2)*window/n
    radius = np.hypot(coordinate[:, None], coordinate[None, :])
    return coordinate, radius, window/n


def test_zernike_area_orthonormality():
    roots, weights = leggauss(60)
    rho = np.sqrt((roots[:, None]+1)/2)
    theta = np.arange(128)[None, :]*2*np.pi/128
    modes = [(n, m) for n in range(7) for m in range(-n, n+1, 2)]
    basis = np.array([zernike(n, m, rho, theta)+np.zeros_like(theta) for n, m in modes])
    gram = np.einsum("irk,jrk,r->ij", basis, basis, weights/2)/theta.size
    np.testing.assert_allclose(gram, np.eye(len(modes)), atol=5e-13)


def test_mask_support_does_not_clip_gaussian_tail():
    coordinate, radius, spacing = grid()
    amplitude = np.exp(-(radius/1.5e-3)**2)
    opd = wavefront(coordinate[None, :], coordinate[:, None], 0.5e-3,
                    (ZernikeTerm(4, 0, 300/550),), 550e-9)
    after = amplitude*np.exp(2j*np.pi*opd/550e-9)
    np.testing.assert_allclose(abs(after), amplitude, rtol=1e-14, atol=1e-15)
    assert np.all(opd[radius > 0.5e-3] == 0)
    assert power(after, spacing) == pytest.approx(power(amplitude, spacing), rel=1e-14)


def test_gaussian_fresnel_propagation_and_aperture_power():
    coordinate, radius, spacing = grid(n=768, window=4e-3)
    wavelength, distance, waist = 550e-9, 0.05, 0.3e-3
    source = np.exp(-(radius/waist)**2)
    propagated = fresnel_propagate(source, spacing, wavelength, distance)
    rayleigh = np.pi*waist**2/wavelength
    propagated_width = waist*np.sqrt(1+(distance/rayleigh)**2)
    analytic = (waist/propagated_width)**2*np.exp(-2*(radius/propagated_width)**2)
    np.testing.assert_allclose(abs(propagated)**2, analytic, atol=2e-10)
    assert power(propagated, spacing) == pytest.approx(power(source, spacing), rel=1e-13)
    aperture_radius = 0.4e-3
    transmitted = power(propagated*(radius <= aperture_radius), spacing)/power(propagated, spacing)
    expected = 1-np.exp(-2*(aperture_radius/propagated_width)**2)
    assert transmitted == pytest.approx(expected, abs=0.001)


def test_uniform_pupil_airy_and_axial_width():
    coordinate, radius, spacing = grid(n=768)
    wavelength, focal, pupil_radius = 550e-9, 0.05, 2e-3
    field = FocusedField((radius <= pupil_radius).astype(float), coordinate, wavelength, focal)
    na = pupil_radius/focal
    cross = np.linspace(-30e-6, 30e-6, 2001)
    intensity = field.line(focal, cross, 0, "x")
    measured = fwhm(cross, intensity)["width"]
    assert measured == pytest.approx(0.514496985*wavelength/na, rel=0.003)
    first_zero = 0.6098349456*wavelength/na
    assert field.point_intensity(focal, first_zero, 0)/field.point_intensity(focal, 0, 0) < 1e-6
    offsets = np.linspace(-1.5e-3, 1.5e-3, 401)
    axial = np.array([field.point_intensity(focal+offset, 0, 0) for offset in offsets])
    assert fwhm(offsets, axial)["width"] == pytest.approx(1.771785883*wavelength/na**2, rel=0.004)


def test_front_focal_gaussian_against_analytic_fourier_pair():
    coordinate, radius, spacing = grid(n=512, window=6e-3)
    wavelength, focal, input_waist = 550e-9, 0.05, 0.3e-3
    source = np.exp(-(radius/input_waist)**2)
    pupil = fresnel_propagate(source, spacing, wavelength, focal)
    field = FocusedField(pupil, coordinate, wavelength, focal)
    output_waist = wavelength*focal/(np.pi*input_waist)
    cross = np.linspace(-100e-6, 100e-6, 2001)
    intensity = field.line(focal, cross, 0, "x")
    np.testing.assert_allclose(intensity/intensity.max(), np.exp(-2*(cross/output_waist)**2), atol=1e-9)
    assert fwhm(cross, intensity)["width"] == pytest.approx(np.sqrt(2*np.log(2))*output_waist, rel=1e-4)
    image_coordinate = np.linspace(-100e-6, 100e-6, 257)
    image = field.image(focal, image_coordinate)
    observed_power = image.sum()*(image_coordinate[1]-image_coordinate[0])**2
    assert observed_power == pytest.approx(power(source, spacing), rel=1e-8)
    shifted_image = field.image(focal, image_coordinate+5e-6, image_coordinate-8e-6)
    assert shifted_image[128, 128] == pytest.approx(field.point_intensity(focal, 5e-6, -8e-6), rel=1e-10)
    z_rayleigh = np.pi*output_waist**2/wavelength
    for offset in (-z_rayleigh, z_rayleigh):
        ratio = field.point_intensity(focal+offset, 0, 0)/field.point_intensity(focal, 0, 0)
        assert ratio == pytest.approx(0.5, abs=2e-8)


def test_front_plane_tilt_moves_focus_and_modifies_pre_lens_intensity():
    coordinate, radius, spacing = grid(n=512)
    wavelength, focal, radius_z, coefficient = 550e-9, 0.05, 4e-3, 200e-9
    source = np.exp(-(radius/0.5e-3)**2)
    opd = wavefront(coordinate[None, :], coordinate[:, None], radius_z,
                    (ZernikeTerm(1, 1, coefficient/wavelength),), wavelength)
    original_pupil = fresnel_propagate(source, spacing, wavelength, focal)
    pupil = fresnel_propagate(source*np.exp(2j*np.pi*opd/wavelength), spacing, wavelength, focal)
    assert np.max(abs(abs(pupil)**2-abs(original_pupil)**2)) > 0.005
    field = FocusedField(pupil, coordinate, wavelength, focal)
    output = np.linspace(-80e-6, 80e-6, 257)
    x_peak, y_peak = peak_coordinate(output, field.image(focal, output))
    assert x_peak == pytest.approx(2*focal*coefficient/radius_z, abs=1e-10)
    assert abs(y_peak) < 1e-12


def test_full_path_zero_aberration_and_output_definitions():
    result = simulate(OpticalConfig(), Numerics(grid_size=256, image_samples=33, axial_samples=33))
    assert result.aberrated is result.baseline
    assert result.summary()["peak_intensity_ratio_same_input_power"] == 1.0
    # 默认的同直径高斯束只通过约 86.5%，验证没有把光束半径误当光强 FWHM。
    assert result.baseline.metrics["lens_transmission_fraction"] == pytest.approx(1-np.exp(-2), abs=0.002)
    assert result.baseline.metrics["lateral_fwhm_x_um"] > 0
    assert result.baseline.metrics["axial_fwhm_um"] > 0


def test_missing_half_height_crossings_are_not_invented():
    result = fwhm(np.linspace(-1, 1, 101), np.ones(101))
    assert result["width"] is None


def test_encircled_energy_uses_total_power_and_detects_missing_energy():
    waist = 10e-6
    coordinate = np.linspace(-2*waist, 2*waist, 513)
    radius_squared = coordinate[:, None]**2 + coordinate[None, :]**2
    image = 2/(np.pi*waist**2)*np.exp(-2*radius_squared/waist**2)
    radius, captured = encircled_energy_radius(coordinate, image, (0, 0), total_power=1)
    assert radius == pytest.approx(waist*np.sqrt(-np.log(0.2)/2), rel=0.003)
    assert captured == pytest.approx(1, abs=0.001)
    center = slice(240, 273)
    radius, captured = encircled_energy_radius(coordinate[center], image[center, center], (0, 0), total_power=1)
    assert radius is None
    assert captured < 0.8


@pytest.mark.parametrize("config,window_mm", [
    (OpticalConfig(), 12),
    (OpticalConfig(beam_type="uniform"), 12),
    (OpticalConfig(beam_diameter_mm=8), 24),
    (OpticalConfig(aberrations=(ZernikeTerm(3, 1, 150/550), ZernikeTerm(2, 0, 100/550))), 12),
])
def test_input_grid_refinement_converges(config, window_mm):
    measurements = []
    for size in (768, 1536):
        coordinate, incident, mask, opd, _, _, _ = _prepare(
            config, Numerics(grid_size=size, window_mm=window_mm, image_samples=33, axial_samples=33),
        )
        spacing = coordinate[1]-coordinate[0]
        pupil = fresnel_propagate(incident*np.exp(2j*np.pi*opd/config.wavelength),
                                  spacing, config.wavelength, config.focal_length)*mask
        field = FocusedField(pupil, coordinate, config.wavelength, config.focal_length)
        cross = np.linspace(-60e-6, 60e-6, 2001)
        intensity = field.line(config.focal_length, cross, 0, "x")
        measurements.append([power(pupil, spacing), fwhm(cross, intensity)["width"], intensity.max()])
    np.testing.assert_allclose(measurements[0], measurements[1], rtol=0.01)


@pytest.mark.parametrize("kwargs", [
    {"wavelength_nm": 0}, {"lens_diameter_mm": -1}, {"beam_type": "unknown"},
    {"focal_length_mm": float("nan")},
    {"aberrations": (ZernikeTerm(2, 0, 0.1), ZernikeTerm(2, 0, 0.2))},
])
def test_invalid_parameters_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        OpticalConfig(**kwargs)


def test_waves_coefficients_are_rms_cycles_and_have_wavelength_independent_phase():
    roots, weights = leggauss(60)
    rho = np.sqrt((roots[:, None]+1)/2)
    theta = np.arange(128)[None, :]*2*np.pi/128
    x, y = rho*np.cos(theta), rho*np.sin(theta)
    terms = (ZernikeTerm(3, 1, coefficient_waves=1.0),)
    opd_450 = wavefront(x, y, 1, terms, 450e-9)
    opd_650 = wavefront(x, y, 1, terms, 650e-9)
    phase_450 = 2*np.pi*opd_450/450e-9
    phase_650 = 2*np.pi*opd_650/650e-9
    rms = np.sqrt(np.sum(phase_450**2*weights[:, None]/2)/theta.size)
    assert rms == pytest.approx(2*np.pi, abs=1e-12)
    np.testing.assert_allclose(phase_450, phase_650, atol=3e-15)
    np.testing.assert_allclose(opd_650, opd_450*650/450, atol=1e-21)


def test_presentation_comparison_uses_common_coordinates_and_actual_phase():
    config = OpticalConfig(aberrations=(ZernikeTerm(3, 1, coefficient_waves=0.15),))
    result = simulate(config, Numerics(grid_size=256, image_samples=65, axial_samples=33))
    np.testing.assert_array_equal(result.baseline.arrays["presentation_coordinate_m"],
                                  result.aberrated.arrays["presentation_coordinate_m"])
    assert result.metadata["zernike_rms_excluding_piston_waves"] == pytest.approx(0.15)
    assert np.max(abs(result.aberrated.arrays["input_phase_rad"])) <= np.pi
    assert np.count_nonzero(result.aberrated.arrays["input_phase_rad"]) > 0
    assert np.all(result.baseline.arrays["input_phase_rad"] == 0)


def test_defocus_is_not_compensated_by_refocusing():
    config = OpticalConfig(beam_diameter_mm=0.6, zernike_radius_mm=1.0,
                           aberrations=(ZernikeTerm(2, 0, coefficient_waves=1.0),))
    result = simulate(config, Numerics(grid_size=512, window_mm=6,
                                      image_samples=129, axial_samples=61))
    # 无限孔径前后焦面的解析 Fourier 对：高斯输入叠加二次相位后，
    # 固定 z=f 处的光斑应按 sqrt(1+(beta*w²)²) 展宽，不能通过调焦消掉。
    beta = 4*np.pi*np.sqrt(3)/config.zernike_radius**2
    waist = config.wavelength*config.focal_length/(np.pi*config.beam_radius)
    expected_width = np.sqrt(2*np.log(2))*waist*np.sqrt(1+(beta*config.beam_radius**2)**2)
    assert result.aberrated.metrics["lateral_fwhm_x_um"] == pytest.approx(expected_width*1e6, rel=0.003)
    assert result.aberrated.metrics["lateral_fwhm_x_um"] > 2*result.baseline.metrics["lateral_fwhm_x_um"]
    for case in (result.baseline, result.aberrated):
        assert case.metrics["observation_distance_from_lens_mm"] == config.focal_length_mm
        assert case.metrics["observation_offset_um"] == 0
        assert not any("best" in key for key in case.metrics | case.arrays)
        assert "presentation_focal_image_W_m2" in case.arrays


@pytest.mark.parametrize("beam", ["gaussian", "uniform"])
def test_two_programs_match_when_aberrations_are_zero(beam):
    config = OpticalConfig(beam_type=beam, zernike_radius_mm=1.3,
                           aberrations=(ZernikeTerm(3, 1, 0.0),))
    numerics = Numerics(grid_size=256, image_samples=65, axial_samples=33)
    plain = simulate_diffraction(config, numerics)
    loaded = simulate_aberration(config, numerics)
    assert plain.case.metrics == loaded.case.metrics
    np.testing.assert_array_equal(plain.case.arrays["presentation_focal_image_W_m2"],
                                  loaded.case.arrays["presentation_focal_image_W_m2"])
    assert "aberrations" not in plain.summary()["parameters"]
    assert "zernike_radius_mm" not in plain.summary()["parameters"]
    assert "baseline" not in loaded.summary()
    assert "aberrated" not in loaded.summary()
    assert plain.program == "diffraction" and loaded.program == "aberration"


def test_diffraction_program_cannot_inherit_aberration_parameters():
    config = OpticalConfig(aberrations=(ZernikeTerm(3, 1, 0.15),))
    result = simulate_diffraction(config, Numerics(grid_size=256, image_samples=33, axial_samples=33))
    assert result.config.aberrations == ()
    assert np.all(result.case.arrays["input_phase_rad"] == 0)
    from optics.cli import _parser
    with pytest.raises(SystemExit):
        _parser("diffraction").parse_args(["--zernike", "3,1,0.15"])


def test_longitudinal_sections_match_direct_field_at_fixed_planes():
    coordinate, radius, spacing = grid(n=256, window=6e-3)
    x, y = coordinate[None, :], coordinate[:, None]
    pupil = np.exp(-(x/0.6e-3)**2-(y/0.4e-3)**2)*np.exp(1j*(300*x+700*y))
    field = FocusedField(pupil, coordinate, 550e-9, 0.05)
    distances = np.array([0.049, 0.05, 0.051])
    offsets = np.linspace(-30e-6, 30e-6, 65)
    fixed_x, fixed_y = 5e-6, -3e-6
    xz, yz = field.longitudinal_sections(distances, offsets, fixed_x, fixed_y)
    for index, distance in enumerate(distances):
        for column in (7, 32, 55):
            assert xz[index, column] == pytest.approx(
                field.point_intensity(distance, fixed_x+offsets[column], fixed_y), rel=1e-9)
            assert yz[index, column] == pytest.approx(
                field.point_intensity(distance, fixed_x, fixed_y+offsets[column]), rel=1e-9)


def test_program_sections_follow_same_fixed_line_as_axial_curve():
    numerics = Numerics(grid_size=256, image_samples=65, axial_samples=33)
    plain = simulate_diffraction(OpticalConfig(), numerics)
    loaded = simulate_aberration(OpticalConfig(aberrations=(ZernikeTerm(3, 1, 0.15),)), numerics)
    for result in (plain, loaded):
        arrays = result.case.arrays
        middle = len(arrays["section_offset_m"])//2
        np.testing.assert_allclose(arrays["xz_intensity_W_m2"][:, middle],
                                   arrays["axial_intensity_W_m2"], rtol=1e-8)
    assert "yz_intensity_W_m2" not in plain.case.arrays
    arrays = loaded.case.arrays
    np.testing.assert_allclose(arrays["yz_intensity_W_m2"][:, middle],
                               arrays["axial_intensity_W_m2"], rtol=1e-8)
    assert arrays["xz_fixed_y_m"] == pytest.approx(loaded.case.metrics["peak_y_um"]*1e-6)
    assert arrays["yz_fixed_x_m"] == pytest.approx(loaded.case.metrics["peak_x_um"]*1e-6)
