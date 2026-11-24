"""居中双列布局：后焦面光斑、轴向曲线、XZ/YZ 和横向截面。"""

import json
from pathlib import Path
import numpy as np
from .display import font_families


def _number(value, digits=4):
    return "未覆盖" if value is None else f"{value:.{digits}g}"


def print_summary(result, verbose=False):
    metrics = result.metrics
    print(f"\n{result.program}：均匀圆光束，观察面 z=f={result.config.focal_length_mm:g} mm")
    print(f"  有效直径：{metrics['effective_diameter_mm']:g} mm；有效 NA：{metrics['effective_na']:.6g}")
    if result.program == "fourier_focus":
        print(f"  瑞利（Rayleigh）分辨距离：{_number(metrics['lateral_rayleigh_um'], 6)} μm（由数值焦斑第一暗环测得）")
    for direction in ("x", "y"):
        print(f"  横向 {direction} 光强 FWHM：{_number(metrics[f'lateral_fwhm_{direction}_um'])} μm")
    print(f"  主峰坐标：x={metrics['peak_x_um']:.4g} μm，y={metrics['peak_y_um']:.4g} μm")
    print(f"  轴向固定纵线：x={metrics['axial_fixed_x_um']:.4g} μm，y={metrics['axial_fixed_y_um']:.4g} μm")
    print(f"  轴向光强 FWHM：{_number(metrics['axial_fwhm_um'], 6)} μm")
    print(f"  轴向主峰相对 z=f 偏移：{_number(metrics['axial_peak_offset_um'], 6)} μm")
    if result.program == "fourier_focus":
        print(f"  轴向瑞利距离：{_number(metrics['axial_rayleigh_um'], 6)} μm（由两侧近零第一极小值测得）")
    for message in result.diagnostics:
        print("  提示："+message)
    if verbose:
        print(f"  FFT：{result.metadata['fft_size']} × {result.metadata['fft_size']}；"
              f"焦面采样：{result.metadata['focal_spacing_um']:.6g} μm")
        print(f"  透射功率：{metrics['transmitted_power_W']:.8g} W；"
              f"全 FFT 积分功率：{metrics['full_fft_power_W']:.8g} W")
        print(f"  轴向扫描：±{result.metadata['axial_half_range_um']:.6g} μm；"
              f"基础步长：{result.metadata['axial_base_step_um']:.6g} μm")


def plot_result(result, destination):
    import matplotlib.pyplot as plt
    cfg, arrays, metrics = result.config, result.arrays, result.metrics
    points = arrays["focal_coordinate_m"]*1e6
    image = arrays["focal_intensity_W_m2"]
    peak = metrics["peak_intensity_W_m2"]
    with plt.rc_context({"font.family": font_families(), "font.size": 10,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False}):
        figure, grid = plt.subplots(3, 2, figsize=(14, 13.5))
        axes = grid.ravel()
        figure.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.82, wspace=0.38, hspace=0.88)
        label = "泽尼克相位聚焦" if result.program == "zernike_focus" else "均匀圆孔径聚焦"
        figure.text(0.5, 0.965, label, fontsize=16, ha="center")
        figure.text(0.5, 0.93,
                    f"λ = {cfg.wavelength_nm:g} nm    f = {cfg.focal_length_mm:g} mm    "
                    f"透镜直径 = {cfg.lens_diameter_mm:g} mm    光束直径 = {cfg.beam_radius*2e3:g} mm    "
                    f"有效 NA = {cfg.effective_na:.4g}", fontsize=10, ha="center")
        terms = [t for t in cfg.aberrations if t.coefficient_waves]
        description = ";  ".join(f"Z({t.n},{t.m}): {t.coefficient_waves:g}" for t in terms[:4])
        if len(terms) > 4:
            description += f";  共 {len(terms)} 项，完整列表见 summary.json"
        phase_description = "系数（waves）："+description if terms else "相位为零"
        if result.program == "fourier_focus":
            phase_description += (f"    瑞利分辨距离 = {_number(metrics['lateral_rayleigh_um'], 6)} μm"
                                  "（数值焦斑第一暗环）")
        figure.text(0.5, 0.897, phase_description, fontsize=10, color="#555555", ha="center")
        dx = result.metadata["focal_spacing_um"]
        artist = axes[0].imshow(image/peak, origin="lower", cmap="magma", interpolation="bilinear",
                                extent=[points[0]-dx/2, points[-1]+dx/2]*2, vmin=0, vmax=1)
        axes[0].set(title="（A）后焦面光强（z=f）", xlabel="x (μm)", ylabel="y (μm)")
        # 色条使用独立的子坐标轴，避免 colorbar 挤压方形图并将其推到列右侧。
        bar = figure.colorbar(artist, cax=axes[0].inset_axes([1.04, 0, 0.035, 1]))
        bar.set_label("I/Imax")
        axial = axes[1]
        offsets = arrays["axial_offset_m"]*1e6
        axial.plot(offsets, arrays["axial_intensity_W_m2"]/metrics["axial_peak_intensity_W_m2"],
                   color="#2674B5", linewidth=1.6)
        axial.axvline(0, color="#777777", linestyle="--", linewidth=0.8, label="原后焦面 z=f")
        axial.axhline(0.5, color="#999999", linestyle=":", linewidth=0.8)
        if metrics["axial_peak_offset_um"] is not None:
            axial.plot(metrics["axial_peak_offset_um"], 1, "o", color="#C47A18", markersize=4)
        if metrics["axial_fwhm_um"] is not None:
            axial.axvspan(metrics["axial_half_max_left_um"], metrics["axial_half_max_right_um"],
                          color="#5BAA85", alpha=0.13)
        axial.set(title=f"（B）轴向光强曲线 I(z)\n固定 x={metrics['axial_fixed_x_um']:.3g}，y={metrics['axial_fixed_y_um']:.3g} μm",
                  xlabel="相对后焦面 Δz (μm)", ylabel="归一化光强 I/Imax", ylim=(-0.025, 1.08))
        axial.legend(loc="upper right", fontsize=8, frameon=False)
        axial_label = (f"轴向 FWHM = {_number(metrics['axial_fwhm_um'], 6)} μm；"
                       f"主峰偏移 = {_number(metrics['axial_peak_offset_um'], 5)} μm")
        if result.program == "fourier_focus":
            axial_label += f"\n轴向瑞利距离 = {_number(metrics['axial_rayleigh_um'], 6)} μm"
            for side in ("left", "right"):
                position = metrics[f"axial_first_min_{side}_um"]
                if position is not None:
                    axial.axvline(position, color="#C47A18", linewidth=0.9, linestyle="--")
        axial.text(0.5, -0.32, axial_label, transform=axial.transAxes,
                   ha="center", va="top", fontsize=10, linespacing=1.6)
        section_norm = result.metadata["section_normalization_W_m2"]
        for axis, plane, direction, fixed, panel in (
            (axes[2], "xz", "x", "y", "C"), (axes[3], "yz", "y", "x", "D")
        ):
            transverse = arrays[f"{plane}_{direction}_coordinate_m"]*1e6
            section = axis.pcolormesh(transverse, arrays["section_z_offset_m"]*1e6,
                                     arrays[f"{plane}_intensity_W_m2"]/section_norm,
                                     shading="auto", cmap="magma", vmin=0, vmax=1, rasterized=True)
            axis.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.8)
            axis.axvline(metrics[f"peak_{direction}_um"], color="white", linewidth=0.6, linestyle=":", alpha=0.5)
            axis.set(title=f"（{panel}）{plane.upper()} 二维光强切面\n固定 {fixed}={metrics[f'peak_{fixed}_um']:.3g} μm",
                     xlabel=f"{direction} (μm)", ylabel="相对后焦面 Δz (μm)")
        section_bar = figure.colorbar(section, cax=axes[3].inset_axes([1.04, 0, 0.035, 1]))
        section_bar.set_label("I/Imax（两切面共用）", fontsize=9)
        for axis, direction, fixed, panel in ((axes[4], "x", "y", "E"), (axes[5], "y", "x", "F")):
            values = arrays[f"{direction}_intensity_W_m2"]/peak
            axis.plot(points, values, color="#2674B5", linewidth=1.6)
            axis.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":")
            if result.program == "fourier_focus":
                for side in ("left", "right"):
                    position = metrics[f"first_dark_{side}_{direction}_um"]
                    if position is not None:
                        axis.axvline(position, color="#C47A18", linewidth=0.9, linestyle="--")
            axis.set(xlabel=f"{direction} (μm)", ylabel="I/Imax", ylim=(-0.025, 1.08),
                     title=f"（{panel}）横向 {direction} 截面（{fixed}={metrics[f'peak_{fixed}_um']:.3g} μm）")
            axis.text(0.5, -0.30, f"FWHM = {_number(metrics[f'lateral_fwhm_{direction}_um'])} μm",
                      transform=axis.transAxes, ha="center", fontsize=11)
        note = ("橙色虚线：横向第一暗环、轴向近零第一极小值；绿色区域：轴向半高全宽。横向光斑固定在 z=f。"
                if result.program == "fourier_focus" else
                "轴向沿固定纵线扫描，宽度与主峰偏移分别报告；横向光斑保持在原后焦面 z=f，不随扫描重新调焦。")
        figure.text(0.5, 0.034, note, fontsize=9, color="#555555", ha="center")
        figure.text(0.5, 0.015, "XZ/YZ 共用一个线性色标，不逐层归一化；白色虚线为 z=f，白色点线为轴向曲线所在纵线。",
                    fontsize=9, color="#555555", ha="center")
        figure.savefig(destination, dpi=160, facecolor="white")
        figure.savefig(Path(destination).with_suffix(".svg"), facecolor="white")
    return figure


def save_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"summary.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(directory/"fields.npz", **result.arrays)
    return plot_result(result, directory/"overview.png")
