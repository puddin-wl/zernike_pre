"""按物理 z 设置离焦相位，在固定纵线上求瞳面积分，无额外二维 FFT。"""

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import fwhm


def fixed_point_field(pupil, coordinate, wavelength, focal_length, x_fixed, y_fixed):
    """返回观察面距透镜 z>0 时该固定横向点的场计算器。

    同时合并透镜和 Fresnel 相位，保留 dx²/(λz) 幅度及 X/(λz)
    空间频率。只省略不影响单点光强的单位模传播相位。
    """
    x, y = np.meshgrid(coordinate, coordinate)
    support = np.abs(pupil) > 0
    radius_squared = (x*x+y*y)[support]
    projection = (x*x_fixed+y*y_fixed)[support]
    weights = pupil[support]
    spacing = coordinate[1]-coordinate[0]

    def evaluate(z):
        if z <= 0:
            raise ValueError("轴向观察位置必须在透镜后 z>0。")
        phase = np.pi/wavelength*((1/z-1/focal_length)*radius_squared-2*projection/z)
        return complex(np.sum(weights*np.exp(1j*phase))*spacing**2/(wavelength*z))

    return evaluate


def _minimum(evaluate, lo, hi):
    """以无量纲区间求极小值，避免 m 单位下的优化绝对容差问题。"""
    result = minimize_scalar(lambda t: evaluate(lo+t*(hi-lo)), bounds=(0, 1),
                             method="bounded", options={"xatol": 1e-10})
    position = lo+float(result.x)*(hi-lo)
    return position, float(evaluate(position))


def scan_axial(pupil, coordinate, config, numerics, x_fixed, y_fixed, *, measure_rayleigh=False):
    field = fixed_point_field(pupil, coordinate, config.wavelength, config.focal_length, x_fixed, y_fixed)
    focal = config.focal_length
    intensity = lambda offset: abs(field(focal+offset))**2
    support = np.abs(pupil) > 0
    # 原有相位保持不变；这里只为自动扫描范围估计像差幅度。
    from .zernike import phase_on_beam
    phase = phase_on_beam(coordinate[None, :], coordinate[:, None], config.beam_radius, config.aberrations)
    waves_span = float(np.ptp(phase[support])/(2*np.pi))
    natural_range = 2*config.wavelength/config.effective_na**2
    half_range = (3+2*waves_span)*natural_range
    messages = []
    if numerics.axial_half_range_um is not None:
        half_range = numerics.axial_half_range_um*1e-6
        if numerics.axial_half_range_um >= config.focal_length_mm*1000:
            raise ValueError("轴向扫描半范围必须小于空气中的透镜焦距，以保证所有观察位置 z>0。")
    elif half_range > 0.8*focal:
        half_range = 0.8*focal
        messages.append("自动轴向半范围已限制到 0.8f；请检查峰值是否被完整覆盖。")
    offsets = np.linspace(-half_range, half_range, numerics.axial_samples)
    values = np.array([intensity(offset) for offset in offsets])
    sampled_peak = int(np.argmax(values))
    boundary_peak = sampled_peak in (0, len(values)-1)
    if boundary_peak:
        peak_offset = None
        peak_intensity = float(values[sampled_peak])
        width = {"width": None, "left": None, "right": None, "half_height_regions": 0}
        messages.append("轴向扫描最大值位于范围边界，峰值偏移和 FWHM 返回 null，请扩大 --axial-half-range-um。")
    else:
        peak_offset, negative_peak = _minimum(lambda offset: -intensity(offset),
                                              offsets[sampled_peak-1], offsets[sampled_peak+1])
        peak_intensity = -negative_peak
        insertion = int(np.searchsorted(offsets, peak_offset))
        if insertion < len(offsets) and offsets[insertion] == peak_offset:
            values[insertion] = peak_intensity
        else:
            offsets = np.insert(offsets, insertion, peak_offset)
            values = np.insert(values, insertion, peak_intensity)
        width = fwhm(offsets, values, peak_index=insertion, evaluator=intensity)
        if width["width"] is None:
            messages.append("轴向主瓣缺少完整半高交点，FWHM 返回 null，请扩大 --axial-half-range-um。")
        elif width["width"] < 6*(2*half_range/(numerics.axial_samples-1)):
            messages.append("轴向主瓣不足 6 个扫描间隔，请增加 --axial-samples 检查收敛。")
        if width["half_height_regions"] > 1:
            messages.append("轴向曲线有多个半高区域，FWHM 仅描述所选最强主峰的连通半高区。")

    def um(value):
        return None if value is None else float(value*1e6)

    metrics = {
        "axial_fixed_x_um": x_fixed*1e6, "axial_fixed_y_um": y_fixed*1e6,
        "axial_peak_offset_um": um(peak_offset),
        "axial_peak_intensity_W_m2": peak_intensity,
        "axial_fwhm_um": um(width["width"]),
        "axial_half_max_left_um": um(width["left"]),
        "axial_half_max_right_um": um(width["right"]),
        "axial_half_height_regions": width["half_height_regions"],
    }
    if measure_rayleigh:
        # 离散圆孔径的轴向场未必有严格零点；定位近零的第一极小值。
        # 极小值强度超过主峰的 1e-4 时，不把它认定为轴向零点参考。
        candidates = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))+1
        left = right = None
        if peak_offset is not None:
            for side in ("left", "right"):
                indices = candidates[offsets[candidates] < peak_offset][::-1] if side == "left" else (
                    candidates[offsets[candidates] > peak_offset]
                )
                if len(indices):
                    index = int(indices[0])
                    position, minimum = _minimum(intensity, offsets[index-1], offsets[index+1])
                    if minimum <= peak_intensity*1e-4:
                        if side == "left":
                            left = position
                        else:
                            right = position
        metrics.update({
            "axial_first_min_left_um": um(left), "axial_first_min_right_um": um(right),
            "axial_rayleigh_um": None if left is None or right is None else um((right-left)/2),
        })
        if metrics["axial_rayleigh_um"] is None:
            messages.append("未定位到主峰两侧的近零第一轴向极小值，轴向瑞利距离返回 null；请扩大扫描范围或加密采样。")

    spacing = coordinate[1]-coordinate[0]
    max_curvature = max(abs(1/(focal-half_range)-1/focal), abs(1/(focal+half_range)-1/focal))
    scan_step_phase = np.pi/config.wavelength*(
        max_curvature*(2*config.effective_radius*spacing+spacing**2)
        + 2*max(abs(x_fixed), abs(y_fixed))*spacing/(focal-half_range)
    )
    if scan_step_phase > np.pi/2:
        messages.append("扫描端点的传播相位在相邻瞳面像素间变化超过 π/2，请增加 --grid-size 检查轴向收敛。")
    metadata = {
        "axial_half_range_um": half_range*1e6,
        "axial_base_step_um": 2*half_range/(numerics.axial_samples-1)*1e6,
        "axial_definition": "fixed transverse line through the z=f sampled peak; exact paraxial z-dependent pupil integral; no extra FFT; one common intensity normalization; no transverse peak tracking",
        "axial_peak_definition": "strongest peak on the fixed line within scan range, locally refined; boundary maximum is not accepted as a located focus",
    }
    if measure_rayleigh:
        metadata["axial_rayleigh_definition"] = "half separation of nearest near-zero intensity minima on both sides of the axial peak; minimum/peak <= 1e-4; numerical pupil integral, not theoretical substitution"
    return metrics, metadata, messages, {
        "axial_offset_m": offsets, "axial_z_m": focal+offsets,
        "axial_intensity_W_m2": values,
    }
