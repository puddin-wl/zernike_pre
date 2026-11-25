"""程序二：均匀圆光束的后焦面 FFT 与固定纵线轴向扫描。"""

from optics import Numerics, OpticalConfig
from optics.cli import run_focus


PARAMETERS = OpticalConfig(
    wavelength_nm=1064,
    numerical_aperture=0.67,   # 如 0.67：均匀光充分填充；忽略孔径、焦距和光束直径
    lens_diameter_mm=4.0,
    focal_length_mm=50.0,
    beam_diameter_mm=None,     # None：等于透镜直径；所有尺寸均指瞳面等效尺寸
)
NUMERICS = Numerics(
    grid_size=512, padding_factor=8,
    axial_samples=241,
    axial_half_range_um=None,  # None：自动；也可输入相对后焦面的扫描半范围 μm
    section_samples=257,
    section_half_width_um=None,  # XZ/YZ 横向半宽，None：自动覆盖离焦光束
)


if __name__ == "__main__":
    run_focus(PARAMETERS, default_numerics=NUMERICS)
