"""程序一：初始介质与可配置样品介质的理论分辨率。"""

from optics import ResolutionConfig
from optics.cli import run_resolution


PARAMETERS = ResolutionConfig(
    wavelength_nm=550.0,
    lens_diameter_mm=4.0,
    focal_length_mm=50.0,       # 透镜在所选初始介质中的工作焦距
    beam_diameter_mm=None,     # None：等于透镜直径；也可填瞳面等效直径 mm
    immersion_medium="air",   # air / water / oil，也可直接填数值，如 1.40
    sample_refractive_index=1.37,  # 样品介质，可改为其他正的实数折射率
)


if __name__ == "__main__":
    run_resolution(PARAMETERS)
