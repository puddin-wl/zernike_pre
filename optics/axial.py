"""按物理 z 设置离焦相位，在固定纵线上求瞳面积分，无额外二维 FFT。"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from .metrics import fwhm, half_max_regions


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


def normalized_point_field(pupil, coordinate, wavelength, numerical_aperture, x_fixed, y_fixed):
    """单位瞳面上的近轴焦区场；输入为相对参考焦面的 Δz（m）。

    pupil 的面积积分功率归一化于无量纲单位圆。以 NA/λ 映射到焦区，
    保留近轴离焦项 -π NA² Δz ρ²/λ；不引入未知的 f、物理瞳面尺寸或 1/z。
    """
    x, y = np.meshgrid(coordinate, coordinate)
    support = np.abs(pupil) > 0
    radius_squared = (x*x+y*y)[support]
    projection = (x*x_fixed+y*y_fixed)[support]
    weights = pupil[support]
    factor = (coordinate[1]-coordinate[0])**2*numerical_aperture/wavelength

    def evaluate(offset):
        phase = -np.pi/wavelength*(numerical_aperture**2*offset*radius_squared
                                  + 2*numerical_aperture*projection)
        return complex(np.sum(weights*np.exp(1j*phase))*factor)

    return evaluate


def _minimum(evaluate, lo, hi):
    """以无量纲区间求极小值，避免 m 单位下的优化绝对容差问题。"""
    result = minimize_scalar(lambda t: evaluate(lo+t*(hi-lo)), bounds=(0, 1),
                             method="bounded", options={"xatol": 1e-10})
    position = lo+float(result.x)*(hi-lo)
    return position, float(evaluate(position))


def scan_axial(pupil, coordinate, config, numerics, x_fixed, y_fixed, *, measure_rayleigh=False):
    if config.uses_na:
        field = normalized_point_field(pupil, coordinate, config.wavelength, config.effective_na, x_fixed, y_fixed)
        intensity = lambda offset: abs(field(offset))**2
    else:
        focal = config.focal_length
        field = fixed_point_field(pupil, coordinate, config.wavelength, focal, x_fixed, y_fixed)
        intensity = lambda offset: abs(field(focal+offset))**2
    support = np.abs(pupil) > 0
    # 原有相位保持不变；这里只为自动扫描范围估计像差幅度。
    from .zernike import phase_on_beam
    beam_radius = 1.0 if config.uses_na else config.beam_radius
    phase = phase_on_beam(coordinate[None, :], coordinate[:, None], beam_radius, config.aberrations)
    waves_span = float(np.ptp(phase[support])/(2*np.pi))
    natural_range = 2*config.wavelength/config.effective_na**2
    half_range = (3+2*waves_span)*natural_range
    messages = []
    if numerics.axial_half_range_um is not None:
        half_range = numerics.axial_half_range_um*1e-6
        if not config.uses_na and numerics.axial_half_range_um >= config.focal_length_mm*1000:
            raise ValueError("轴向扫描半范围必须小于空气中的透镜焦距，以保证所有观察位置 z>0。")
    elif not config.uses_na and half_range > 0.8*focal:
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
        local_width = width
        profile = {"intervals": [], "outer_left": None, "outer_right": None, "outer_span": None}
        messages.append("轴向扫描最大值位于范围边界，峰值偏移和 FWHM 返回 null，请扩大 --axial-half-range-um。")
    else:
        # 逐一细化采样中发现的局部峰，再比较真实峰高，避免仅细化一个采样最大值。
        candidates = find_peaks(values)[0]
        refined = [_minimum(lambda offset: -intensity(offset), offsets[index-1], offsets[index+1])
                   for index in candidates]
        peak_offset, negative_peak = min(refined, key=lambda item: item[1])
        peak_intensity = -negative_peak
        offsets = np.r_[offsets, [position for position, value in refined]]
        values = np.r_[values, [-value for position, value in refined]]
        order = np.argsort(offsets, kind="stable")
        offsets, values = offsets[order], values[order]
        unique = np.r_[True, np.diff(offsets) > 0]
        offsets, values = offsets[unique], values[unique]
        insertion = int(np.searchsorted(offsets, peak_offset))
        local_width = fwhm(offsets, values, peak_index=insertion, evaluator=intensity)
        profile = half_max_regions(offsets, values, peak_height=peak_intensity, evaluator=intensity)
        # 轴向测量整条光强分布的最外侧半高宽，不让内部凹槽或最高峰身份截断宽度。
        width = {"width": profile["outer_span"], "left": profile["outer_left"],
                 "right": profile["outer_right"], "half_height_regions": len(profile["intervals"])}
        if width["width"] is None:
            messages.append("轴向最外侧半高交点未完整覆盖，FWHM 返回 null，请扩大 --axial-half-range-um。")
        elif width["width"] < 6*(2*half_range/(numerics.axial_samples-1)):
            messages.append("轴向主瓣不足 6 个扫描间隔，请增加 --axial-samples 检查收敛。")
        if width["half_height_regions"] > 1:
            messages.append("轴向 FWHM 按整条曲线最外侧半高交点计算，包含中间低于半高的凹槽；最高光强位置的切换不会切换测量区间。")

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
        "axial_half_max_intervals_um": [[um(lo), um(hi)] for lo, hi in profile["intervals"]],
        "axial_half_max_outer_left_um": um(profile["outer_left"]),
        "axial_half_max_outer_right_um": um(profile["outer_right"]),
        "axial_half_max_span_um": um(profile["outer_span"]),
        "axial_local_fwhm_um": um(local_width["width"]),
        "axial_local_half_max_left_um": um(local_width["left"]),
        "axial_local_half_max_right_um": um(local_width["right"]),
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
    if config.uses_na:
        scan_step_phase = np.pi/config.wavelength*(
            config.effective_na**2*half_range*(2*spacing+spacing**2)
            + 2*config.effective_na*max(abs(x_fixed), abs(y_fixed))*spacing
        )
    else:
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
        "axial_definition": (
            "NA mode; fixed transverse line through the reference-focus peak; normalized pupil paraxial defocus integral in delta_z; no focal length; no extra FFT; no transverse peak tracking"
            if config.uses_na else
            "fixed transverse line through the z=f sampled peak; exact paraxial z-dependent pupil integral; no extra FFT; one common intensity normalization; no transverse peak tracking"
        ),
        "axial_peak_definition": "all sampled local peaks on the fixed line are refined before selecting the strongest; boundary maximum is not accepted; selected position can switch between lobes in a multi-peak profile",
        "axial_fwhm_definition": "outermost half-maximum width of the entire axial profile at 0.5 times the refined global maximum; includes internal valleys below half height; peak position does not select the width interval; zero intensity baseline; null if clipped; does not imply continuous depth of focus",
        "axial_local_fwhm_definition": "auxiliary connected half-height region containing the selected highest peak; may jump when the highest peak changes; not the reported axial FWHM",
        "axial_half_max_span_definition": "alias of axial_fwhm_um: distance between outermost global half-maximum crossings, including internal gaps; null if clipped",
    }
    if measure_rayleigh:
        metadata["axial_rayleigh_definition"] = "half separation of nearest near-zero intensity minima on both sides of the axial peak; minimum/peak <= 1e-4; numerical pupil integral, not theoretical substitution"
    arrays = {"axial_offset_m": offsets, "axial_intensity_W_m2": values}
    if not config.uses_na:
        arrays["axial_z_m"] = focal+offsets
    return metrics, metadata, messages, arrays
