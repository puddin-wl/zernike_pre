"""程序三：瞳面共轭面泽尼克相位、固定后焦面 FFT 与固定纵线轴向扫描。"""

from optics import Numerics, OpticalConfig, ZernikeTerm
from optics.cli import run_focus


PARAMETERS = OpticalConfig(
    wavelength_nm=550.0,
    lens_diameter_mm=4.0,
    focal_length_mm=50.0,
    beam_diameter_mm=None,     # None：等于透镜直径；泽尼克单位圆自动覆盖完整光束
    aberrations=(
        # 系数单位 waves；非活塞项为完整光束圆上的 waves RMS。
        # m>0：cos(mθ)，m<0：sin(|m|θ)。取消注释即可启用对应模式。
        # ZernikeTerm(0, 0, 0.10),   # 活塞：整体相位，不改变焦面光强
        # ZernikeTerm(1, 1, 0.10),   # x 倾斜
        # ZernikeTerm(1, -1, 0.10),  # y 倾斜
        ZernikeTerm(2, 0, 0.10),     # 离焦：仍观察固定后焦面，不重新调焦
        # ZernikeTerm(2, 2, 0.10),   # 0°/90° 像散
        # ZernikeTerm(2, -2, 0.10),  # 45°/135° 像散
        ZernikeTerm(3, 1, 0.15),     # x 彗差
        # ZernikeTerm(3, -1, 0.10),  # y 彗差
        # ZernikeTerm(3, 3, 0.10),   # 三叶，cos(3θ)
        # ZernikeTerm(3, -3, 0.10),  # 三叶，sin(3θ)
        ZernikeTerm(4, 0, 0.10),     # 初级球差
        # 也可加入其他满足 n<=20、|m|<=n、n-|m| 为偶数的模式。
    ),
)
NUMERICS = Numerics(
    grid_size=512, padding_factor=8,
    axial_samples=241,
    axial_half_range_um=None,  # None：自动；不通过扫描重新选择横向观察面
    section_samples=257,
    section_half_width_um=None,  # XZ/YZ 横向半宽，None：自动覆盖离焦光束
)


if __name__ == "__main__":
    run_focus(PARAMETERS, aberrated=True, default_numerics=NUMERICS)
