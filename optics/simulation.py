"""均匀圆光束、固定后焦面 FFT 与固定纵线的物理离焦扫描。"""

from dataclasses import asdict, dataclass, replace
import numpy as np

from .config import Numerics, OpticalConfig
from .fourier import focus_fft
from .metrics import first_dark_ring, fwhm
from .theory import optical_parameters
from .zernike import phase_on_beam
from .axial import scan_axial
from .sections import longitudinal_sections


@dataclass
class FocusResult:
    config: OpticalConfig
    numerics: Numerics
    program: str
    metrics: dict
    metadata: dict
    diagnostics: list[str]
    arrays: dict[str, np.ndarray]

    def summary(self):
        parameters = optical_parameters(self.config)
        metrics = dict(self.metrics)
        if self.program == "fourier_focus":
            parameters.pop("aberrations")
        return {
            "program": self.program,
            "model": "air; scalar paraxial; ideal pupil-conjugate phase; one scaled 2D FFT at z=f plus axial fixed-point defocus integrals",
            "parameters": parameters,
            "numerics": asdict(self.numerics),
            "metadata": self.metadata,
            "results": metrics,
            "diagnostics": self.diagnostics,
        }


def simulate_focus(config: OpticalConfig, numerics: Numerics | None = None, *, aberrated=False):
    numerics = numerics or Numerics()
    if not aberrated:
        config = replace(config, aberrations=())
    # 只采样实际透射区域，过填充时被挡住的外围不必占据计算窗口。
    window = 2.5 * config.effective_radius
    spacing = window / numerics.grid_size
    coordinate = (np.arange(numerics.grid_size)-numerics.grid_size//2)*spacing
    x, y = coordinate[None, :], coordinate[:, None]
    radius = np.hypot(x, y)
    support = radius <= config.effective_radius
    phase = phase_on_beam(x, y, config.beam_radius, config.aberrations)
    # 入射光束解析功率为 1 W；截光后不重新归一化功率。
    amplitude = 1 / np.sqrt(np.pi*config.beam_radius**2)
    pupil = amplitude * support * np.exp(1j*phase)
    transmitted_power = float(np.sum(np.abs(pupil)**2)*spacing**2)
    analytic_transmission = (config.effective_radius/config.beam_radius)**2
    messages = config.diagnostics()
    if abs(transmitted_power/analytic_transmission-1) > 0.005:
        messages.append("离散圆孔径面积误差超过 0.5%，请增加 --grid-size。")
    phase_steps = []
    for axis in (0, 1):
        valid = np.take(support, range(support.shape[axis]-1), axis=axis) & np.take(
            support, range(1, support.shape[axis]), axis=axis
        )
        differences = np.abs(np.diff(phase, axis=axis))
        phase_steps.append(float(np.max(differences[valid])) if np.any(valid) else 0.0)
    maximum_step = max(phase_steps)
    if maximum_step > np.pi/2:
        messages.append("相邻瞳面像素相位变化超过 π/2，请增加 --grid-size 检查收敛，防止混叠。")

    focal_coordinate, full_image, origin_field_x, origin_field_y = focus_fft(
        pupil, spacing, config.wavelength, config.focal_length, numerics.padding_factor
    )
    focal_spacing = float(focal_coordinate[1]-focal_coordinate[0])
    full_power = float(full_image.sum()*focal_spacing**2)
    row, column = np.unravel_index(np.argmax(full_image), full_image.shape)
    peak_x, peak_y = float(focal_coordinate[column]), float(focal_coordinate[row])
    natural_scale = config.wavelength*config.focal_length/(2*config.effective_radius)
    half_width = max(4*natural_scale, max(abs(peak_x), abs(peak_y))+3*natural_scale)
    if numerics.image_half_width_um is not None:
        half_width = numerics.image_half_width_um*1e-6
    available = min(abs(float(focal_coordinate[0])), float(focal_coordinate[-1]))
    if half_width > available:
        half_width = available
        messages.append("显示视场达到 FFT 可用范围，请增加 --grid-size 以扩大焦面可用视场。")
    selected = np.flatnonzero(abs(focal_coordinate) <= half_width)
    if len(selected) < 5:
        raise ValueError("焦面视场不足 5 个采样点，请增大 --image-half-width-um 或 --padding-factor。")
    first, last = int(selected[0]), int(selected[-1])+1
    if not (first <= row < last and first <= column < last):
        raise ValueError("指定焦面视场未包含全局主峰，请增大 --image-half-width-um。")
    points = focal_coordinate[first:last].copy()
    image = full_image[first:last, first:last].copy()
    local_row, local_column = row-first, column-first
    line_x, line_y = image[local_row, :].copy(), image[:, local_column].copy()
    widths = {"x": fwhm(points, line_x, local_column), "y": fwhm(points, line_y, local_row)}
    metrics = {
        "effective_diameter_mm": config.effective_radius*2e3,
        "effective_na": config.effective_na,
        "observation_distance_mm": config.focal_length_mm,
        "peak_x_um": peak_x*1e6,
        "peak_y_um": peak_y*1e6,
        "peak_intensity_W_m2": float(full_image[row, column]),
        "incident_power_W": 1.0,
        "transmitted_power_W": transmitted_power,
        "geometric_transmission": analytic_transmission,
        "full_fft_power_W": full_power,
        "display_power_fraction": float(image.sum()*focal_spacing**2/transmitted_power),
    }
    for direction, width in widths.items():
        metrics[f"lateral_fwhm_{direction}_um"] = None if width["width"] is None else width["width"]*1e6
        metrics[f"half_height_regions_{direction}"] = width["half_height_regions"]
        if width["width"] is None:
            messages.append(f"{direction} 截面未覆盖完整半高交点，FWHM 返回 null，请扩大显示视场。")
        elif width["half_height_regions"] > 1:
            messages.append(f"{direction} 截面存在多个半高区域，FWHM 仅描述所选主峰。")
    field_sections = {}
    if not aberrated:
        radii = []
        for direction, field in (("x", origin_field_x), ("y", origin_field_y)):
            section = field[first:last].copy()
            field_sections[f"{direction}_reduced_field_sqrt_W_per_m"] = section
            ring = first_dark_ring(points, section)
            for name in ("left", "right"):
                metrics[f"first_dark_{name}_{direction}_um"] = (
                    None if ring[name] is None else ring[name]*1e6
                )
            radius_um = None if ring["radius"] is None else ring["radius"]*1e6
            metrics[f"lateral_rayleigh_{direction}_um"] = radius_um
            radii.append(radius_um)
        metrics["lateral_rayleigh_um"] = None if any(r is None for r in radii) else float(np.mean(radii))
        if metrics["lateral_rayleigh_um"] is None:
            messages.append("未能从数值焦斑定位完整第一暗环，瑞利距离返回 null，请扩大视场或细化 FFT 采样。")
    border = max(float(full_image[0, :].max()), float(full_image[-1, :].max()),
                 float(full_image[:, 0].max()), float(full_image[:, -1].max()))
    if border > metrics["peak_intensity_W_m2"]*0.01:
        messages.append("FFT 边界光强超过峰值的 1%，请增加 --grid-size 并检查相位采样。")
    transmitted_waves = phase[support]/(2*np.pi)
    metadata = {
        "pupil_spacing_um": spacing*1e6,
        "pupil_window_mm": window*1e3,
        "fft_size": numerics.grid_size*numerics.padding_factor,
        "focal_spacing_um": focal_spacing*1e6,
        "zernike_normalization_radius_mm": config.beam_radius*1e3,
        "full_beam_wavefront_rms_waves": float(np.sqrt(sum(
            t.coefficient_waves**2 for t in config.aberrations if t.n != 0
        ))),
        "sampled_transmitted_wavefront_rms_waves": float(np.std(transmitted_waves)),
        "maximum_neighbor_phase_step_rad": maximum_step,
        "fwhm_definition": "connected half-maximum width through sampled global peak at fixed z=f; not Rayleigh",
    }
    if not aberrated:
        metadata["rayleigh_definition"] = (
            "numerical first dark ring: first sign changes on both sides of the FFT field peak; "
            "cubic-spline zero crossings; mean of left/right radii and x/y sections; no 0.61*lambda/NA substitution"
        )
    axial_metrics, axial_metadata, axial_messages, axial_arrays = scan_axial(
        pupil, coordinate, config, numerics, peak_x, peak_y, measure_rayleigh=not aberrated
    )
    metrics.update(axial_metrics)
    metadata.update(axial_metadata)
    messages.extend(axial_messages)
    section_arrays = longitudinal_sections(
        pupil, coordinate, config, numerics, axial_arrays["axial_offset_m"], peak_x, peak_y
    )
    metadata["section_normalization_W_m2"] = max(
        float(section_arrays["xz_intensity_W_m2"].max()),
        float(section_arrays["yz_intensity_W_m2"].max()),
    )
    metadata["section_definition"] = "XZ at fixed y=peak_y(z=f); YZ at fixed x=peak_x(z=f); no peak tracking; one shared linear intensity normalization for both entire sections"
    return FocusResult(
        config, numerics, "zernike_focus" if aberrated else "fourier_focus",
        metrics, metadata, messages,
        {"pupil_coordinate_m": coordinate, "pupil_support": support,
         "pupil_phase_unwrapped_rad": np.where(support, phase, 0),
         "focal_coordinate_m": points, "focal_intensity_W_m2": image,
         "x_intensity_W_m2": line_x, "y_intensity_W_m2": line_y,
         **field_sections, **axial_arrays, **section_arrays},
    )
