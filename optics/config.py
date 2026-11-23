from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class ZernikeTerm:
    """单位面积 RMS 基底，系数单位 waves；1 wave 对应 2π rad RMS。"""

    n: int
    m: int
    coefficient_waves: float

    def __post_init__(self):
        if not isinstance(self.n, int) or not isinstance(self.m, int):
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
    beam_type: str = "gaussian"
    beam_diameter_mm: float = 4.0
    zernike_radius_mm: float = 2.0
    aberrations: tuple[ZernikeTerm, ...] = field(default_factory=tuple)

    def __post_init__(self):
        for name in ("wavelength_nm", "lens_diameter_mm", "focal_length_mm",
                     "beam_diameter_mm", "zernike_radius_mm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是正的有限数。")
        if self.beam_type not in ("gaussian", "uniform"):
            raise ValueError('beam_type 只能为 "gaussian" 或 "uniform"。')
        if not all(isinstance(term, ZernikeTerm) for term in self.aberrations):
            raise ValueError("aberrations 必须由 ZernikeTerm 组成。")
        modes = [(term.n, term.m) for term in self.aberrations]
        if len(modes) != len(set(modes)):
            raise ValueError("同一个 (n,m) 只能出现一次；请先合并系数。")

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
        return self.beam_diameter_mm * 0.5e-3

    @property
    def zernike_radius(self):
        return self.zernike_radius_mm * 1e-3

    @property
    def paraxial_na(self):
        return self.lens_radius / self.focal_length


@dataclass(frozen=True)
class Numerics:
    grid_size: int = 768
    image_samples: int = 257
    axial_samples: int = 121
    window_mm: float | None = None
    image_half_width_um: float | None = None
    axial_half_range_um: float | None = None

    def __post_init__(self):
        if self.grid_size < 128 or self.grid_size % 2:
            raise ValueError("grid_size 要求为不小于 128 的偶数。")
        if self.image_samples < 33 or self.image_samples % 2 != 1:
            raise ValueError("image_samples 要求为不小于 33 的奇数。")
        if self.axial_samples < 33 or self.axial_samples % 2 != 1:
            raise ValueError("axial_samples 要求为不小于 33 的奇数。")
        for name in ("window_mm", "image_half_width_um", "axial_half_range_um"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} 必须是正的有限数。")
