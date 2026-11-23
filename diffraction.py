"""程序一：无像差衍射计算。修改参数后直接运行本文件。"""

from optics import OpticalConfig
from optics.cli import run_diffraction


PARAMETERS = OpticalConfig(
    wavelength_nm=550.0,
    lens_diameter_mm=4.0,
    focal_length_mm=50.0,
    beam_type="uniform",       # "gaussian" 或 "uniform"
    beam_diameter_mm=4.0,        # 前焦面：高斯 1/e² 光强直径／均匀圆光斑直径
)


if __name__ == "__main__":
    run_diffraction(PARAMETERS)
