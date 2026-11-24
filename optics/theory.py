"""仅用标准库计算理想近轴分辨率；不导入绘图库或 FFT。"""

from dataclasses import asdict
import math
from .config import OpticalConfig, ResolutionConfig


def optical_parameters(config: OpticalConfig):
    parameters = asdict(config)
    parameters["beam_diameter_mm"] = config.beam_radius * 2e3
    if isinstance(config, ResolutionConfig):
        parameters["immersion_refractive_index"] = config.immersion_refractive_index
    return parameters


def calculate_resolution(config: OpticalConfig, sample_refractive_index: float | None = None):
    if sample_refractive_index is None:
        sample_refractive_index = getattr(config, "sample_refractive_index", 1.37)
    if not math.isfinite(sample_refractive_index) or sample_refractive_index <= 0:
        raise ValueError("样品介质折射率必须是正的有限实数。")
    parameters = optical_parameters(config)
    parameters.pop("aberrations")
    na = config.effective_na
    if sample_refractive_index <= na:
        raise ValueError("样品折射率必须大于入射 NA，才能使用无角谱截断的平面界面模型；当前组合需处理全反射/角谱损失。")
    parameters["sample_refractive_index"] = sample_refractive_index
    lateral_um = 0.61 * config.wavelength / na * 1e6
    media = {}
    initial_index = getattr(config, "immersion_refractive_index", 1.0)
    for name, index in (("initial_medium", initial_index),
                        ("sample_medium", sample_refractive_index)):
        media[name] = {
            "refractive_index": index,
            "effective_na": na,
            "lateral_rayleigh_um": lateral_um,
            "axial_resolution_um": 2 * index * config.wavelength / na**2 * 1e6,
        }
    diagnostics = config.diagnostics()
    if na/sample_refractive_index > 0.2:
        diagnostics.append("样品中 NA/n_s > 0.2：组织侧的近轴轴向公式误差可能显著，请使用更完整的传播模型。")
    return {
        "program": "resolution",
        "model": "scalar paraxial; working focal length in initial medium; normal planar interface into sample; conserved n*sin(theta); no mismatch-induced aberrations",
        "parameters": parameters,
        "effective_diameter_mm": config.effective_radius * 2e3,
        "formulas": {
            "effective_na": "n_initial*min(D_beam, D_lens)/(2*f_working)",
            "lateral_rayleigh": "0.61*lambda_vacuum/NA",
            "axial_resolution": "2*n*lambda_vacuum/NA^2 (paraxial estimate; not FWHM)",
        },
        "results": media,
        "diagnostics": diagnostics,
    }
