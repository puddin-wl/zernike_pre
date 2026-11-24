"""程序一的两介质输入和理想样品界面尺度。"""

import pytest

from optics import OpticalConfig, ResolutionConfig, calculate_resolution


@pytest.mark.parametrize("sample_index", [1.0, 1.2, 1.333, 1.37, 1.515, 1.7])
def test_sample_index_is_configurable_without_changing_incident_na(sample_index):
    result = calculate_resolution(ResolutionConfig(sample_refractive_index=sample_index))
    initial = result["results"]["initial_medium"]
    sample = result["results"]["sample_medium"]
    assert initial["effective_na"] == pytest.approx(0.04)
    assert initial["axial_resolution_um"] == pytest.approx(687.5)
    assert sample["lateral_rayleigh_um"] == initial["lateral_rayleigh_um"]
    assert sample["axial_resolution_um"]/initial["axial_resolution_um"] == pytest.approx(sample_index)
    assert result["parameters"]["sample_refractive_index"] == sample_index


def test_oil_changes_na_while_sample_to_initial_axial_ratio_uses_both_indices():
    config = ResolutionConfig(immersion_medium="oil", sample_refractive_index=1.40)
    result = calculate_resolution(config, 1.40)
    initial, sample = result["results"].values()
    assert config.immersion_refractive_index == 1.515
    assert config.effective_na == pytest.approx(0.0606)
    assert initial["lateral_rayleigh_um"] == pytest.approx(8.3875/1.515)
    assert initial["axial_resolution_um"] == pytest.approx(687.5/1.515)
    assert sample["axial_resolution_um"]/initial["axial_resolution_um"] == pytest.approx(1.40/1.515)
    assert initial["lateral_rayleigh_um"] == sample["lateral_rayleigh_um"]


def test_custom_initial_index_and_preset_name_resolution():
    config = ResolutionConfig(immersion_medium=1.42)
    assert config.immersion_label == "custom"
    assert calculate_resolution(config, 1.33)["parameters"]["immersion_medium"] == 1.42
    assert ResolutionConfig(immersion_medium=" WATER ").immersion_refractive_index == 1.333


@pytest.mark.parametrize("medium", [0, -1, float("nan"), float("inf"), "unknown"])
def test_invalid_initial_medium_fails(medium):
    with pytest.raises(ValueError):
        ResolutionConfig(immersion_medium=medium)


@pytest.mark.parametrize("index", [0, -1, float("nan"), float("inf")])
def test_invalid_sample_index_fails(index):
    with pytest.raises(ValueError, match="折射率"):
        calculate_resolution(OpticalConfig(), index)


def test_interface_with_lost_angular_spectrum_is_not_treated_as_conserved_na():
    with pytest.raises(ValueError, match="全反射"):
        calculate_resolution(OpticalConfig(), 0.01)


def test_sample_angle_has_its_own_paraxial_diagnostic():
    result = calculate_resolution(ResolutionConfig(sample_refractive_index=0.08))
    assert any("NA/n_s" in message for message in result["diagnostics"])
