"""两个独立程序的精简输出；对比绘图位于 comparison_report.py。"""

import json
from pathlib import Path
import numpy as np
from .display import font_families


def _number(value):
    if value is None:
        return "未覆盖"
    return f"{value:.2f}" if abs(value) < 100 else (f"{value:.1f}" if abs(value) < 1000 else f"{value:.0f}")


def print_summary(result, verbose=False):
    label = "无像差" if result.program == "diffraction" else "加载像差"
    beam = "高斯光" if result.config.beam_type == "gaussian" else "均匀光"
    print(f"\n{label} / {beam}，观察面 z=f={result.config.focal_length_mm:g} mm")
    for label, key in (("横向 x", "lateral_fwhm_x_um"),
                       ("横向 y", "lateral_fwhm_y_um"), ("轴向", "axial_fwhm_um")):
        print(f"  {label} FWHM：{_number(result.case.metrics[key])} μm")
    messages = result.diagnostics+result.case.diagnostics
    if messages:
        print(f"数值诊断 {len(messages)} 项，详见 summary.json。")
        if verbose:
            for message in messages:
                print("  - "+message)


def _image(figure, rect, coordinate, data, title, unit, *, phase=False, limit=None):
    import matplotlib.pyplot as plt
    axis = figure.add_axes(rect)
    scale = 1e3 if unit == "mm" else 1e6
    points = coordinate*scale
    if phase:
        palette = plt.get_cmap("twilight_shifted").copy()
        palette.set_bad("#EEEEEE")
        values, low, high = data, -np.pi, np.pi
    else:
        palette = "magma"
        values, low, high = data/max(float(np.max(data)), 1e-300), 0, 1
    artist = axis.imshow(values, origin="lower", extent=[points[0], points[-1]]*2,
                         cmap=palette, vmin=low, vmax=high, interpolation="nearest")
    axis.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})")
    axis.set_title(title, fontsize=11, pad=12, linespacing=1.6)
    if limit is not None:
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
    bar_rect = [rect[0]+rect[2]+0.017, rect[1], 0.007, rect[3]]
    bar = figure.colorbar(artist, cax=figure.add_axes(bar_rect))
    if phase:
        bar.set_ticks([-np.pi, 0, np.pi], labels=["−π", "0", "π"])
    else:
        bar.set_ticks([0, 0.5, 1])
    bar.ax.set_title("rad" if phase else "I/Imax", fontsize=9, pad=10)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9)
    return axis


def _curve(figure, rect, result, direction, title, metric_y):
    axis = figure.add_axes(rect)
    arrays, metrics = result.case.arrays, result.case.metrics
    if direction == "z":
        positions = arrays["axial_coordinate_m"]*1e6
        values = arrays["axial_intensity_W_m2"]
        width = metrics["axial_fwhm_um"]
        if width is not None:
            axis.set_xlim(max(positions[0], metrics["axial_half_max_left_um"]-0.6*width),
                          min(positions[-1], metrics["axial_half_max_right_um"]+0.6*width))
        xlabel = "相对后焦面 Δz (μm)"
    else:
        positions = arrays[f"{direction}_line_m"]*1e6-metrics[f"peak_{direction}_um"]
        values = arrays[f"{direction}_intensity_W_m2"]
        width = metrics[f"lateral_fwhm_{direction}_um"]
        if width is not None:
            axis.set_xlim(-2.1*width, 2.1*width)
        xlabel = "相对主峰位置 (μm)"
    axis.plot(positions, values/max(float(values.max()), 1e-300), color="#3576C9", linewidth=1.8)
    axis.axhline(0.5, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=0)
    axis.set(ylim=(-0.035, 1.08), yticks=[0, 0.5, 1])
    axis.set_title(title, fontsize=11, pad=12)
    axis.set_xlabel(xlabel, fontsize=10, labelpad=7)
    axis.set_ylabel("I/Imax", fontsize=10, labelpad=7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9, length=3)
    figure.text(rect[0]+rect[2]/2, metric_y, f"FWHM = {_number(width)} μm", ha="center", fontsize=11)


def _section(figure, rect, result, direction, normalization, show_y=True):
    arrays = result.case.arrays
    plane, transverse, fixed_name = ("xz", "x", "y") if direction == "x" else ("yz", "y", "x")
    coordinate = arrays[f"{plane}_{transverse}_coordinate_m"]*1e6
    z_coordinate = arrays["section_z_offset_m"]*1e6
    values = arrays[f"{plane}_intensity_W_m2"]/normalization
    fixed = float(arrays[f"{plane}_fixed_{fixed_name}_m"])*1e6
    if abs(fixed) < 0.005:
        fixed = 0.0
    axis = figure.add_axes(rect)
    artist = axis.pcolormesh(coordinate, z_coordinate, values, shading="auto",
                             cmap="magma", vmin=0, vmax=1, rasterized=True)
    axis.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
    axis.set(xlim=(coordinate[0], coordinate[-1]), ylim=(z_coordinate[0], z_coordinate[-1]),
             xlabel=f"{transverse} (μm)", ylabel="Δz = z−f (μm)")
    axis.set_title(f"{plane.upper()} 光强切面\n{fixed_name} = {fixed:.2f} μm",
                   fontsize=11, pad=12, linespacing=1.6)
    if not show_y:
        # 两张切面共用同一 z 轴范围，右图省略重复的纵轴标注。
        axis.set_ylabel("")
        axis.tick_params(axis="y", left=False, labelleft=False)
    bar = figure.colorbar(artist, cax=figure.add_axes([rect[0]+rect[2]+0.017, rect[1], 0.007, rect[3]]))
    bar.set_ticks([0, 0.5, 1])
    bar.ax.set_title("I/Imax", fontsize=9, pad=10)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9)
    return axis


def plot_result(result, destination):
    import matplotlib.pyplot as plt

    cfg, arrays = result.config, result.case.arrays
    beam = "高斯入射" if cfg.beam_type == "gaussian" else "均匀入射"
    with plt.rc_context({"font.family": font_families(), "font.size": 10,
                         "axes.unicode_minus": False, "text.color": "#222222",
                         "axes.labelcolor": "#444444", "xtick.color": "#555555", "ytick.color": "#555555",
                         "axes.edgecolor": "#AAAAAA", "axes.linewidth": 0.7}):
        figure = plt.figure(figsize=(16, 9), facecolor="white")
        figure.text(0.06, 0.968,
                    f"{beam}    λ = {cfg.wavelength_nm:g} nm    f = {cfg.focal_length_mm:g} mm    "
                    f"口径 = {cfg.lens_diameter_mm:g} mm    光束直径 = {cfg.beam_diameter_mm:g} mm",
                    fontsize=10)
        source_limit = 1.25*cfg.beam_diameter_mm/2
        focal_title = f"焦平面光斑\nz = f = {cfg.focal_length_mm:g} mm"
        section_max = max(float(arrays["xz_intensity_W_m2"].max()),
                          float(arrays.get("yz_intensity_W_m2", arrays["xz_intensity_W_m2"]).max()), 1e-300)
        if result.program == "diffraction":
            _image(figure, [0.075, 0.49, 0.21, 0.37333], arrays["pupil_coordinate_m"],
                   arrays["input_intensity_W_m2"], "输入光强（前焦面）", "mm", limit=source_limit)
            _image(figure, [0.390, 0.49, 0.21, 0.37333], arrays["presentation_coordinate_m"],
                   arrays["presentation_focal_image_W_m2"], focal_title, "μm")
            _section(figure, [0.705, 0.49, 0.21, 0.37333], result, "x", section_max)
            _curve(figure, [0.10, 0.17, 0.34, 0.21], result, "x", "横向截面（圆对称，x=y）", 0.065)
            _curve(figure, [0.60, 0.17, 0.34, 0.21], result, "z", "轴向光强", 0.065)
        else:
            terms = [term for term in cfg.aberrations if term.coefficient_waves]
            modes = ";  ".join(f"Z({term.n},{term.m}) = {term.coefficient_waves:g} waves" for term in terms[:3])
            if len(terms) > 3:
                modes += f";  共 {len(terms)} 项"
            figure.text(0.06, 0.938, modes or "像差系数为 0", color="#666666", fontsize=9)
            _image(figure, [0.060, 0.56, 0.16, 0.28444], arrays["pupil_coordinate_m"],
                   arrays["input_intensity_W_m2"], "输入光强（前焦面）", "mm", limit=source_limit)
            visible = arrays["input_intensity_W_m2"] > arrays["input_intensity_W_m2"].max()*1e-4
            phase = np.ma.array(arrays["input_phase_rad"], mask=~visible)
            _image(figure, [0.300, 0.56, 0.16, 0.28444], arrays["pupil_coordinate_m"], phase,
                   f"输入相位（加载后）\nRMS = {result.metadata['zernike_rms_excluding_piston_waves']:.3g} waves",
                   "mm", phase=True, limit=1.13*max(cfg.zernike_radius_mm, cfg.beam_diameter_mm/2))
            _image(figure, [0.540, 0.56, 0.16, 0.28444], arrays["presentation_coordinate_m"],
                   arrays["presentation_focal_image_W_m2"], focal_title, "μm")
            _curve(figure, [0.800, 0.56, 0.17, 0.28444], result, "z", "轴向光强", 0.477)
            _section(figure, [0.060, 0.13, 0.16, 0.30222], result, "x", section_max)
            _section(figure, [0.300, 0.13, 0.16, 0.30222], result, "y", section_max, show_y=False)
            _curve(figure, [0.565, 0.13, 0.17, 0.30222], result, "x", "横向 x 截面", 0.038)
            _curve(figure, [0.805, 0.13, 0.17, 0.30222], result, "y", "横向 y 截面", 0.038)
        figure.text(0.06, 0.010, "观察面固定 z=f；切面共用线性色标，虚线为 z=f；轴向扫描不用于重新调焦。",
                    color="#666666", fontsize=8)
        figure.savefig(destination, dpi=160, facecolor="white")
        figure.savefig(Path(destination).with_suffix(".svg"), facecolor="white")
    return figure


def save_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"summary.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    np.savez_compressed(directory/"fields.npz", **result.case.arrays)
    return plot_result(result, directory/"overview.png")
