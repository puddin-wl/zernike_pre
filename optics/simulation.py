"""光路编排、固定后焦面光强、轴向扫描与数值诊断。"""

from dataclasses import asdict, dataclass, replace
from typing import Callable
import numpy as np

from .config import Numerics, OpticalConfig
from .metrics import encircled_energy_radius, fwhm, peak_coordinate
from .propagation import FocusedField, fresnel_propagate, power
from .zernike import wavefront


@dataclass
class CaseResult:
    metrics: dict
    diagnostics: list[str]
    arrays: dict[str, np.ndarray]


@dataclass
class SingleResult:
    """单个程序的结果，不包含无像差/有像差对比。"""

    config: OpticalConfig
    numerics: Numerics
    case: CaseResult
    diagnostics: list[str]
    metadata: dict
    program: str

    def summary(self):
        parameters = asdict(self.config)
        metadata = dict(self.metadata)
        if self.program == "diffraction":
            parameters.pop("aberrations")
            parameters.pop("zernike_radius_mm")
            metadata = {key: value for key, value in metadata.items() if not key.startswith("zernike_")}
        return {
            "program": self.program,
            "model": "air; scalar paraxial Fresnel; finite-aperture thin lens; observation at z=f",
            "parameters": parameters, "numerics": asdict(self.numerics),
            "metadata": metadata, "results": self.case.metrics,
            "diagnostics": self.diagnostics+self.case.diagnostics,
        }


@dataclass
class SimulationResult:
    config: OpticalConfig
    numerics: Numerics
    baseline: CaseResult
    aberrated: CaseResult
    diagnostics: list[str]
    metadata: dict

    def summary(self):
        reference = self.baseline.metrics["peak_intensity_W_m2"]
        return {
            "model": "air; scalar paraxial Fresnel; finite-aperture thin lens; mask at front focal plane",
            "parameters": asdict(self.config),
            "numerics": asdict(self.numerics),
            "metadata": self.metadata,
            "baseline": self.baseline.metrics,
            "aberrated": self.aberrated.metrics,
            "peak_intensity_ratio_same_input_power": self.aberrated.metrics[
                "peak_intensity_W_m2"]/reference,
            "diagnostics": self.diagnostics,
            "baseline_diagnostics": self.baseline.diagnostics,
            "aberrated_diagnostics": self.aberrated.diagnostics,
        }


def _edge_fraction(field, pixels=12):
    intensity = np.abs(field)**2
    interior = intensity[pixels:-pixels, pixels:-pixels].sum()
    return float(max(0.0, 1-interior/intensity.sum()))


def _prepare(config, numerics):
    n = numerics.grid_size
    wavelength, focal = config.wavelength, config.focal_length
    window = max(5*config.lens_radius, 6*config.beam_radius,
                 8*wavelength*focal/config.beam_radius,
                 np.sqrt(1.2*wavelength*focal*n))
    if numerics.window_mm is not None:
        window = numerics.window_mm*1e-3
    spacing = window/n
    coordinate = (np.arange(n)-n//2)*spacing
    x, y = coordinate[None, :], coordinate[:, None]
    radius = np.hypot(x, y)
    amplitude = np.exp(-(radius/config.beam_radius)**2) if config.beam_type == "gaussian" else (
        (radius <= config.beam_radius).astype(float)
    )
    # 按无限平面解析入射功率归一到 1 W，不把窗口外丢失的尾部偷偷补回去。
    incident_integral = np.pi*config.beam_radius**2/(2 if config.beam_type == "gaussian" else 1)
    incident = amplitude/np.sqrt(incident_integral)
    pupil_mask = radius <= config.lens_radius
    opd = wavefront(x, y, config.zernike_radius, config.aberrations, wavelength)
    messages = []
    if not 380 <= config.wavelength_nm <= 780:
        messages.append("波长在本项目约定的可见光范围 380–780 nm 外；仍按输入波长计算。")
    if config.paraxial_na > 0.2:
        messages.append("D/(2f)>0.2：近轴薄透镜误差可能显著；本程序仍只报告近轴模型结果。")
    for diameter, name in ((2*config.lens_radius, "透镜"), (2*config.beam_radius, "入射光束")):
        if diameter/spacing < 128:
            messages.append(f"{name}直径仅覆盖 {diameter/spacing:.1f} 个输入像素，建议增加 --grid-size。")
    if max(config.lens_radius, config.beam_radius) > window*0.35:
        messages.append("输入计算窗口的余量偏小，可能发生 FFT 周期回绕；请扩大 --window-mm 并检查收敛。")
    if wavelength*focal*n/window**2 > 1:
        messages.append("前段传播传递函数采样不足；请扩大 --window-mm，并相应增加 --grid-size。")
    sampled_power = power(incident, spacing)
    if abs(sampled_power-1) > 0.005:
        messages.append(f"输入网格积分功率为 {sampled_power:.6f} W，偏离解析 1 W；请检查窗口及采样。")
    if _edge_fraction(incident) > 1e-6:
        messages.append("加载面计算窗口边缘存在可见能量，可能截掉输入光场。")
    active = (radius < config.zernike_radius-2*spacing) & (amplitude > 1e-4)
    derivatives = np.gradient(opd, spacing)
    gradient = max(float(np.max(abs(d[active]))) for d in derivatives) if np.any(active) else 0.0
    if gradient*spacing*2*np.pi/wavelength > np.pi/2:
        messages.append("像差相位在相邻输入像素间变化较快，请加密网格检查收敛。")
    effective_radius = min(config.lens_radius, config.beam_radius)
    half_width = 6*wavelength*focal/effective_radius + 1.5*focal*gradient
    if numerics.image_half_width_um is not None:
        half_width = numerics.image_half_width_um*1e-6
    half_range = 6*wavelength*focal**2/effective_radius**2
    largest_opd = max((abs(t.coefficient_waves)*wavelength for t in config.aberrations), default=0.0)
    half_range = max(half_range, 12*largest_opd*focal**2/config.zernike_radius**2)
    if numerics.axial_half_range_um is not None:
        half_range = numerics.axial_half_range_um*1e-6
    if half_range >= 0.8*focal:
        half_range = 0.8*focal
        messages.append("轴向扫描半范围限制为 0.8f，避免穿过透镜；可能无法覆盖全部半高宽。")
    image_coordinate = np.linspace(-half_width, half_width, numerics.image_samples)
    z_offsets = np.linspace(-half_range, half_range, numerics.axial_samples)
    return coordinate, incident, pupil_mask, opd, image_coordinate, z_offsets, messages


def _focus_metrics(field, image_coordinate, z_offsets, numerics, callback):
    """观察面严格固定 z=f；轴向扫描只用于曲线及 FWHM，不用于调焦。"""
    focal = field.focal_length
    fine_coordinate = np.linspace(image_coordinate[0], image_coordinate[-1],
                                  max(257, numerics.image_samples))
    image = field.image(focal, fine_coordinate)
    x_peak, y_peak = peak_coordinate(fine_coordinate, image)
    half_width = image_coordinate[-1]
    x_line = np.linspace(x_peak-half_width, x_peak+half_width, 2049)
    y_line = np.linspace(y_peak-half_width, y_peak+half_width, 2049)
    intensity_x = field.line(focal, x_line, y_peak, "x")
    intensity_y = field.line(focal, y_line, x_peak, "y")
    width_x = fwhm(x_line, intensity_x, peak=x_peak)
    width_y = fwhm(y_line, intensity_y, peak=y_peak)
    peak_intensity = field.point_intensity(focal, x_peak, y_peak)

    # 固定在 z=f 光斑的主峰横向位置扫描，不跟随各 z 平面的峰值。
    axial_coordinate = np.unique(np.r_[z_offsets, 0.0])
    evaluate_axial = lambda offset: field.point_intensity(focal+offset, x_peak, y_peak)
    axial_intensity = np.empty(len(axial_coordinate))
    for index, offset in enumerate(axial_coordinate):
        axial_intensity[index] = evaluate_axial(offset)
        if callback is not None and index in (0, len(axial_coordinate)//2, len(axial_coordinate)-1):
            callback(f"轴向光强扫描 {index+1}/{len(axial_coordinate)}")
    axial_width = fwhm(axial_coordinate, axial_intensity, evaluator=evaluate_axial)
    passed_power = power(field.pupil, field.spacing)
    energy_half_width = max(3*(width_x["width"] or half_width),
                            3*(width_y["width"] or half_width))
    ee80 = None
    for _ in range(4):
        energy_coordinate = np.linspace(-energy_half_width, energy_half_width, 513)
        energy_image = field.image(focal, energy_coordinate+x_peak, energy_coordinate+y_peak)
        ee80, energy_window_fraction = encircled_energy_radius(
            energy_coordinate, energy_image, (0.0, 0.0), passed_power,
        )
        if ee80 is not None:
            break
        energy_half_width *= 2
    image_power_fraction = float(np.sum(image)*(fine_coordinate[1]-fine_coordinate[0])**2/passed_power)

    def micrometers(value):
        return None if value is None else float(value*1e6)

    metrics = {
        "observation_distance_from_lens_mm": focal*1e3,
        "observation_offset_um": 0.0,
        "peak_x_um": x_peak*1e6, "peak_y_um": y_peak*1e6,
        "lateral_fwhm_x_um": micrometers(width_x["width"]),
        "lateral_fwhm_y_um": micrometers(width_y["width"]),
        "axial_fwhm_um": micrometers(axial_width["width"]),
        "axial_half_max_left_um": micrometers(axial_width["left"]),
        "axial_half_max_right_um": micrometers(axial_width["right"]),
        "peak_intensity_W_m2": peak_intensity,
        "encircled_energy_80_radius_um": micrometers(ee80),
        "image_window_power_fraction": image_power_fraction,
        "energy_window_power_fraction": energy_window_fraction,
        "axial_half_height_regions": axial_width["half_height_regions"],
    }
    diagnostics = []
    for name, width in (("横向 x", width_x), ("横向 y", width_y), ("轴向", axial_width)):
        if width["width"] is None:
            diagnostics.append(f"{name}未找到完整的半高交点；输出 null，请扩大对应扫描范围。")
        if width["half_height_regions"] > 1:
            diagnostics.append(f"{name}存在多个分离的半高区域；FWHM 仅描述所选主峰，需结合光强图判断。")
    if int(np.argmax(axial_intensity)) in (0, len(axial_coordinate)-1):
        diagnostics.append("轴向曲线的最大采样值位于扫描边界，请扩大 --axial-half-range-um。")
    if max(abs(x_peak), abs(y_peak)) > 0.85*half_width:
        diagnostics.append("z=f 平面的峰值接近横向视场边界，请扩大 --image-half-width-um。")
    if ee80 is None:
        diagnostics.append("观察面视场未收集到 80% 透射功率，EE80 返回 null，请扩大视场。")
    if image_power_fraction > 1.02:
        diagnostics.append("观察面数值积分功率超过透射功率 2%，输出采样可能不足，请增加 --image-samples。")
    image_step = fine_coordinate[1]-fine_coordinate[0]
    if any(width["width"] is not None and width["width"]/image_step < 3 for width in (width_x, width_y)):
        diagnostics.append("z=f 光斑每个主峰半高宽不足 3 个采样点，请增加 --image-samples 或缩小视场。")
    if axial_width["width"] is not None and axial_width["width"]/(z_offsets[1]-z_offsets[0]) < 4:
        diagnostics.append("轴向半高宽不足 4 个扫描间隔，请增加 --axial-samples。")
    arrays = {
        "image_coordinate_m": fine_coordinate, "focal_image_W_m2": image,
        "axial_coordinate_m": axial_coordinate, "axial_intensity_W_m2": axial_intensity,
        "x_line_m": x_line, "y_line_m": y_line,
        "x_intensity_W_m2": intensity_x, "y_intensity_W_m2": intensity_y,
    }
    return metrics, diagnostics, arrays


def _simulate(config, numerics, progress, program):
    numerics = numerics or Numerics()
    coordinate, incident, aperture, opd, image_coordinate, z_offsets, diagnostics = _prepare(config, numerics)
    spacing = coordinate[1]-coordinate[0]

    def calculate(phase, name):
        if progress is not None:
            progress(f"{name}：前焦面传播至透镜")
        before_lens = fresnel_propagate(incident*np.exp(2j*np.pi*phase/config.wavelength),
                                       spacing, config.wavelength, config.focal_length)
        pupil = before_lens*aperture
        if power(pupil, spacing) < 1e-12:
            raise ValueError("透过透镜的功率接近零，请检查参数或增加计算窗口／采样。")
        field = FocusedField(pupil, coordinate, config.wavelength, config.focal_length)
        callback = None if progress is None else lambda message: progress(f"{name}：{message}")
        metrics, messages, arrays = _focus_metrics(field, image_coordinate, z_offsets, numerics, callback)
        if program != "comparison":
            offsets = np.linspace(image_coordinate[0], image_coordinate[-1], max(257, numerics.image_samples))
            x_fixed, y_fixed = metrics["peak_x_um"]*1e-6, metrics["peak_y_um"]*1e-6
            xz, yz = field.longitudinal_sections(
                config.focal_length+arrays["axial_coordinate_m"], offsets, x_fixed, y_fixed,
                include_y=(program == "aberration"), progress=callback,
            )
            arrays.update({"section_z_offset_m": arrays["axial_coordinate_m"],
                           "section_offset_m": offsets,
                           "xz_x_coordinate_m": x_fixed+offsets,
                           "xz_fixed_y_m": np.array(y_fixed),
                           "xz_intensity_W_m2": xz})
            if yz is not None:
                arrays.update({"yz_y_coordinate_m": y_fixed+offsets,
                               "yz_fixed_x_m": np.array(x_fixed),
                               "yz_intensity_W_m2": yz})
        at_mask_power = power(incident, spacing)
        before_power = power(before_lens, spacing)
        passed_power = power(pupil, spacing)
        edge = _edge_fraction(before_lens)
        metrics.update({"analytic_input_power_W": 1.0,
                        "sampled_input_power_W": at_mask_power,
                        "power_before_lens_W": before_power,
                        "power_after_lens_W": passed_power,
                        "lens_transmission_fraction": passed_power/before_power,
                        "pre_lens_window_edge_power_fraction": edge})
        if edge > 1e-5:
            messages.append("透镜前计算窗口边缘存在能量；FFT 可能回绕，请扩大窗口并进行独立收敛检查。")
        arrays.update({"pupil_coordinate_m": coordinate,
                       "input_intensity_W_m2": np.abs(incident)**2,
                       "loaded_opd_m": phase,
                       "input_phase_rad": np.angle(np.exp(2j*np.pi*phase/config.wavelength)),
                       "pre_lens_intensity_W_m2": np.abs(before_lens)**2,
                       "pre_lens_phase_rad": np.angle(before_lens),
                       "pupil_intensity_W_m2": np.abs(pupil)**2})
        return CaseResult(metrics, messages, arrays)

    if program == "comparison":
        baseline = calculate(np.zeros_like(opd), "无像差参考")
        aberrated = calculate(opd, "加载像差") if np.any(opd) else baseline
    else:
        # 两个独立程序每次只传播自己的光场，不附带参考计算。
        baseline = aberrated = calculate(opd, "无像差" if program == "diffraction" else "加载像差")
    # 展示用的两张焦斑重新在同一密集网格求值，不放大原先的粗像素。
    cases = (baseline,) if aberrated is baseline else (baseline, aberrated)
    valid_widths = [case.metrics[f"lateral_fwhm_{direction}_um"]*1e-6
                    for case in cases for direction in ("x", "y")
                    if case.metrics[f"lateral_fwhm_{direction}_um"] is not None]
    width = max(valid_widths, default=2*config.wavelength/config.paraxial_na)
    displacement = max(abs(case.metrics[f"peak_{direction}_um"])*1e-6
                       for case in cases for direction in ("x", "y"))
    presentation_radius = displacement + 2.5*width
    presentation_coordinate = np.linspace(-presentation_radius, presentation_radius, 513)
    for case in cases:
        pupil = np.sqrt(case.arrays["pupil_intensity_W_m2"])*np.exp(1j*case.arrays["pre_lens_phase_rad"])
        field = FocusedField(pupil, coordinate, config.wavelength, config.focal_length)
        case.arrays["presentation_coordinate_m"] = presentation_coordinate
        case.arrays["presentation_focal_image_W_m2"] = field.image(
            config.focal_length, presentation_coordinate,
        )
    rms_waves = float(np.sqrt(sum(term.coefficient_waves**2 for term in config.aberrations if term.n != 0)))
    metadata = {
        "input_power_normalization_W": 1.0,
        "observation_plane": "fixed at z=f behind lens; no focus search or refocusing",
        "axial_curve": "fixed lateral position at the peak of the z=f image; used only for axial intensity FWHM",
        "longitudinal_sections": "XZ at fixed y=peak_y(z=f); YZ at fixed x=peak_x(z=f) for aberration program; linear intensity, common normalization for both sections",
        "initial_wavefront": "flat at front focal plane; Gaussian has its waist here",
        "zernike_convention": "unit disk area RMS; m>0 cos, m<0 sin; outside disk OPD=0; no amplitude stop",
        "zernike_coefficient_unit": "waves; phase = 2*pi*sum(coefficient_waves*Z); OPD = wavelength*sum(coefficient_waves*Z)",
        "zernike_rms_excluding_piston_waves": rms_waves,
        "zernike_phase_rms_excluding_piston_rad": 2*np.pi*rms_waves,
        "zernike_rms_excluding_piston_nm": rms_waves*config.wavelength_nm,
        "input_window_mm": float(spacing*numerics.grid_size*1e3),
        "input_spacing_um": float(spacing*1e6),
        "paraxial_na": config.paraxial_na,
        "uniform_pupil_analytic_reference": {
            "condition": "uniform flat wavefront AT LENS, not any finite front-plane beam",
            "rayleigh_lateral_um": 0.6098349456*config.wavelength/config.paraxial_na*1e6,
            "lateral_fwhm_um": 0.514496985*config.wavelength/config.paraxial_na*1e6,
            "axial_fwhm_um": 1.771785883*config.wavelength/config.paraxial_na**2*1e6,
        },
    }
    if program == "comparison":
        return SimulationResult(config, numerics, baseline, aberrated, diagnostics, metadata)
    return SingleResult(config, numerics, aberrated, diagnostics, metadata, program)


def simulate_diffraction(config: OpticalConfig, numerics: Numerics | None = None,
                         progress: Callable[[str], None] | None = None) -> SingleResult:
    """无像差程序；清空全部像差，仅计算无像差光场。"""
    return _simulate(replace(config, aberrations=()), numerics, progress, "diffraction")


def simulate_aberration(config: OpticalConfig, numerics: Numerics | None = None,
                        progress: Callable[[str], None] | None = None) -> SingleResult:
    """像差程序；仅计算指定相位屏及其后续传播。"""
    return _simulate(config, numerics, progress, "aberration")


def simulate(config: OpticalConfig, numerics: Numerics | None = None,
             progress: Callable[[str], None] | None = None) -> SimulationResult:
    """封装保留的对比接口；独立程序不调用此接口。"""
    return _simulate(config, numerics, progress, "comparison")
