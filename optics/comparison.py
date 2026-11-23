"""封装保留的对比程序：python -m optics.comparison。"""

from optics import OpticalConfig, ZernikeTerm
from optics.comparison_cli import run


PARAMETERS = OpticalConfig(
    wavelength_nm=550.0,
    lens_diameter_mm=4.0,
    focal_length_mm=50.0,
    beam_type="gaussian",       # "gaussian" 或 "uniform"
    beam_diameter_mm=4.0,        # 前焦面：高斯 1/e² 光强直径／均匀圆光斑直径
    zernike_radius_mm=2.0,       # 前焦面 Zernike 归一化圆半径，不是挡光孔径
    aberrations=(
        ZernikeTerm(n=3, m=1, coefficient_waves=0.15),  # x 彗差；1 wave = 2π rad RMS
        # ZernikeTerm(n=2, m=0, coefficient_waves=0.10),  # 离焦
        # ZernikeTerm(n=4, m=0, coefficient_waves=0.10),  # 初级球差
    ),
)


if __name__ == "__main__":
    run(PARAMETERS)
