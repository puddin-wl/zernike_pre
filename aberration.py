"""程序二：前焦面加载 RMS 归一化 Zernike 像差。"""

from optics import OpticalConfig, ZernikeTerm
from optics.cli import run_aberration


PARAMETERS = OpticalConfig(
    wavelength_nm=550.0,
    lens_diameter_mm=4.0,
    focal_length_mm=50.0,
    beam_type="gaussian",       # "gaussian" 或 "uniform"
    beam_diameter_mm=4.0,
    zernike_radius_mm=2.0,       # 加载面归一化圆半径，不是光阑
    aberrations=(
        # 系数单位 waves RMS：1 wave = 2π rad RMS；正负号控制相位方向。
        # 取消行首注释即可启用，示例系数可自行修改；同一 (n,m) 只保留一项。
        # m>0 使用 cos(mθ)，m<0 使用 sin(|m|θ)。下面列出 n=0~6 的全部实模式。

        # n=0：活塞（只在归一化圆内施加，并非整个光场的整体相位）
        # ZernikeTerm(n=0, m=0, coefficient_waves=0.10),   # 活塞

        # n=1：倾斜
        # ZernikeTerm(n=1, m=1, coefficient_waves=0.10),   # x 方向倾斜
        # ZernikeTerm(n=1, m=-1, coefficient_waves=0.10),  # y 方向倾斜

        # n=2：离焦与初级像散
        ZernikeTerm(n=2, m=0, coefficient_waves=0.10),     # 离焦
        # ZernikeTerm(n=2, m=2, coefficient_waves=0.10),   # 0°/90° 初级像散
        # ZernikeTerm(n=2, m=-2, coefficient_waves=0.10),  # 45°/135° 初级像散

        # n=3：初级彗差与三叶像差
        ZernikeTerm(n=3, m=1, coefficient_waves=0.15),     # x 初级彗差
        # ZernikeTerm(n=3, m=-1, coefficient_waves=0.10),  # y 初级彗差
        # ZernikeTerm(n=3, m=3, coefficient_waves=0.10),   # 三叶像差，cos(3θ)
        # ZernikeTerm(n=3, m=-3, coefficient_waves=0.10),  # 三叶像差，sin(3θ)

        # n=4：初级球差、次级像散与四叶像差
        ZernikeTerm(n=4, m=0, coefficient_waves=0.10),     # 初级球差
        # ZernikeTerm(n=4, m=2, coefficient_waves=0.10),   # 次级像散，cos(2θ)
        # ZernikeTerm(n=4, m=-2, coefficient_waves=0.10),  # 次级像散，sin(2θ)
        # ZernikeTerm(n=4, m=4, coefficient_waves=0.10),   # 四叶像差，cos(4θ)
        # ZernikeTerm(n=4, m=-4, coefficient_waves=0.10),  # 四叶像差，sin(4θ)

        # n=5：次级彗差、次级三叶与五叶像差
        # ZernikeTerm(n=5, m=1, coefficient_waves=0.10),   # x 次级彗差
        # ZernikeTerm(n=5, m=-1, coefficient_waves=0.10),  # y 次级彗差
        # ZernikeTerm(n=5, m=3, coefficient_waves=0.10),   # 次级三叶像差，cos(3θ)
        # ZernikeTerm(n=5, m=-3, coefficient_waves=0.10),  # 次级三叶像差，sin(3θ)
        # ZernikeTerm(n=5, m=5, coefficient_waves=0.10),   # 五叶像差，cos(5θ)
        # ZernikeTerm(n=5, m=-5, coefficient_waves=0.10),  # 五叶像差，sin(5θ)

        # n=6：次级球差、三级像散、次级四叶与六叶像差
        # ZernikeTerm(n=6, m=0, coefficient_waves=0.10),   # 次级球差
        # ZernikeTerm(n=6, m=2, coefficient_waves=0.10),   # 三级像散，cos(2θ)
        # ZernikeTerm(n=6, m=-2, coefficient_waves=0.10),  # 三级像散，sin(2θ)
        # ZernikeTerm(n=6, m=4, coefficient_waves=0.10),   # 次级四叶像差，cos(4θ)
        # ZernikeTerm(n=6, m=-4, coefficient_waves=0.10),  # 次级四叶像差，sin(4θ)
        # ZernikeTerm(n=6, m=6, coefficient_waves=0.10),   # 六叶像差，cos(6θ)
        # ZernikeTerm(n=6, m=-6, coefficient_waves=0.10),  # 六叶像差，sin(6θ)
    ),
)


if __name__ == "__main__":
    run_aberration(PARAMETERS)
