"""三个程序的输入约定；内部长度统一使用 m。"""

from dataclasses import dataclass, field
import math


MEDIUM_INDICES = {"air": 1.0, "water": 1.333, "oil": 1.515}


def refractive_index(medium):
    """预设为演示用参考值，也允许直接填写任意正的实数折射率。"""
    if isinstance(medium, str):
        name = medium.strip().lower()
        if name in MEDIUM_INDICES:
            return MEDIUM_INDICES[name]
    try:
        value = float(medium)
    except (TypeError, ValueError) as error:
        raise ValueError("初始介质请填写 air、water、oil 或正的数值折射率。") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("初始介质折射率必须是正的有限实数。")
    return value


@dataclass(frozen=True)
class ZernikeTerm:
    """单位圆面积 RMS 基底；非活塞项系数单位 waves RMS。"""

    n: int
    m: int
    coefficient_waves: float

    def __post_init__(self):
        if type(self.n) is not int or type(self.m) is not int:
            raise ValueError("Zernike n、m 必须是整数。")
        if not (0 <= self.n <= 20 and abs(self.m) <= self.n):
            raise ValueError("要求 0 <= n <= 20，且 |m| <= n。")
        if (self.n - abs(self.m)) % 2:
            raise ValueError("Zernike 要求 n-|m| 为偶数。")
        if not math.isfinite(self.coefficient_waves):
            raise ValueError("像差系数必须是有限数。")


@dataclass(frozen=True)
class OpticalConfig:
    wavelength_nm: float = 550.0
    lens_diameter_mm: float = 4.0
    focal_length_mm: float = 50.0
    beam_diameter_mm: float | None = None  # None：自动等于透镜直径
    aberrations: tuple[ZernikeTerm, ...] = field(default_factory=tuple)

    def __post_init__(self):
        for name in ("wavelength_nm", "lens_diameter_mm", "focal_length_mm", "beam_diameter_mm"):
            value = getattr(self, name)
            if name == "beam_diameter_mm" and value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是正的有限数。")
        if not all(isinstance(term, ZernikeTerm) for term in self.aberrations):
            raise ValueError("aberrations 必须由 ZernikeTerm 组成。")
        modes = [(term.n, term.m) for term in self.aberrations]
        if len(modes) != len(set(modes)):
            raise ValueError("同一个 (n,m) 只能出现一次，请先合并系数。")

    @property
    def wavelength(self):
        return self.wavelength_nm * 1e-9

    @property
    def focal_length(self):
        return self.focal_length_mm * 1e-3

    @property
    def lens_radius(self):
        return self.lens_diameter_mm * 0.5e-3

    @property
    def beam_radius(self):
        diameter = self.lens_diameter_mm if self.beam_diameter_mm is None else self.beam_diameter_mm
        return diameter * 0.5e-3

    @property
    def effective_radius(self):
        return min(self.lens_radius, self.beam_radius)

    @property
    def effective_na(self):
        return self.effective_radius / self.focal_length

    def diagnostics(self):
        if self.effective_na > 0.2:
            return ["有效 NA > 0.2：近轴模型误差可能显著，本结果不是高 NA 精确解。"]
        return []


@dataclass(frozen=True)
class ResolutionConfig(OpticalConfig):
    """仅程序一使用的两介质参数；f 为初始介质中的工作焦距。"""

    immersion_medium: str | float = "air"
    sample_refractive_index: float = 1.37

    def __post_init__(self):
        super().__post_init__()
        refractive_index(self.immersion_medium)
        if not math.isfinite(self.sample_refractive_index) or self.sample_refractive_index <= 0:
            raise ValueError("样品介质折射率必须是正的有限实数。")
        if self.angular_aperture >= 1:
            raise ValueError("有效半径/工作焦距必须小于 1；当前几何超出可用近轴孔径范围。")

    @property
    def immersion_refractive_index(self):
        return refractive_index(self.immersion_medium)

    @property
    def immersion_label(self):
        name = str(self.immersion_medium).strip().lower()
        return name if name in MEDIUM_INDICES else "custom"

    @property
    def angular_aperture(self):
        return self.effective_radius / self.focal_length

    @property
    def effective_na(self):
        return self.immersion_refractive_index * self.angular_aperture

    def diagnostics(self):
        if self.angular_aperture > 0.2:
            return ["NA/n_i = 有效半径/工作焦距 > 0.2：近轴模型误差可能显著，本结果不是高 NA 精确解。"]
        return []


@dataclass(frozen=True)
class Numerics:
    grid_size: int = 512
    padding_factor: int = 8
    image_half_width_um: float | None = None
    axial_samples: int = 241
    axial_half_range_um: float | None = None
    section_samples: int = 257
    section_half_width_um: float | None = None

    def __post_init__(self):
        if type(self.grid_size) is not int or self.grid_size < 128 or self.grid_size % 2:
            raise ValueError("grid_size 必须是不小于 128 的偶数。")
        if type(self.padding_factor) is not int or self.padding_factor < 2:
            raise ValueError("padding_factor 必须是不小于 2 的整数。")
        if type(self.axial_samples) is not int or self.axial_samples < 33 or self.axial_samples % 2 != 1:
            raise ValueError("axial_samples 必须是不小于 33 的奇数。")
        if type(self.section_samples) is not int or self.section_samples < 33 or self.section_samples % 2 != 1:
            raise ValueError("section_samples 必须是不小于 33 的奇数。")
        for name in ("image_half_width_um", "axial_half_range_um", "section_half_width_um"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} 必须是正的有限数。")
