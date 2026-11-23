"""精简控制台输出与 16:9 展示页；完整计算结果保存在数据文件。"""

import json
from pathlib import Path
import numpy as np

BLUE, ORANGE = "#3576C9", "#DF813A"


def _number(value):
    if value is None:
        return "未覆盖"
    magnitude = abs(value)
    return f"{value:.2f}" if magnitude < 100 else (f"{value:.1f}" if magnitude < 1000 else f"{value:.0f}")


def print_summary(result, verbose=False):
    beam = "高斯光" if result.config.beam_type == "gaussian" else "均匀光"
    print(f"\n{beam} | λ={result.config.wavelength_nm:g} nm | "
          f"D={result.config.lens_diameter_mm:g} mm | f={result.config.focal_length_mm:g} mm")
    print("FWHM (μm)                 无像差 → 有像差")
    for label, key in (("横向 x", "lateral_fwhm_x_um"),
                       ("横向 y", "lateral_fwhm_y_um"), ("轴向", "axial_fwhm_um")):
        print(f"  {label}                  {_number(result.baseline.metrics[key])} → "
              f"{_number(result.aberrated.metrics[key])}")
    messages = list(result.diagnostics)
    messages += ["无像差："+message for message in result.baseline.diagnostics]
    if result.aberrated is not result.baseline:
        messages += ["有像差："+message for message in result.aberrated.diagnostics]
    if messages:
        print(f"数值诊断 {len(messages)} 项，详见 summary.json。")
        if verbose:
            for message in messages:
                print("  - "+message)


def plot_result(result, destination):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cases, cfg = (result.baseline, result.aberrated), result.config
    with plt.rc_context({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10,
        "axes.unicode_minus": False, "text.color": "#222222",
        "axes.labelcolor": "#444444", "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.edgecolor": "#AAAAAA", "axes.linewidth": 0.7,
    }):
        figure = plt.figure(figsize=(16, 9), facecolor="white")
        beam = "高斯入射" if cfg.beam_type == "gaussian" else "均匀入射"
        figure.text(0.06, 0.968,
                    f"{beam}    λ = {cfg.wavelength_nm:g} nm    f = {cfg.focal_length_mm:g} mm    "
                    f"口径 = {cfg.lens_diameter_mm:g} mm    光束直径 = {cfg.beam_diameter_mm:g} mm",
                    fontsize=10)
        active = [term for term in cfg.aberrations if term.coefficient_waves]
        modes = ";  ".join(f"Z({term.n},{term.m}) = {term.coefficient_waves:g} waves" for term in active[:3])
        if len(active) > 3:
            modes += f";  共 {len(active)} 项"
        figure.text(0.06, 0.938, modes or "像差系数为 0", color="#666666", fontsize=9)
        figure.legend(handles=[Line2D([], [], color=BLUE, linestyle="--", linewidth=1.8, label="无像差"),
                               Line2D([], [], color=ORANGE, linewidth=1.8, label="有像差")],
                      loc="upper right", bbox_to_anchor=(0.968, 0.992), ncol=2,
                      frameon=False, fontsize=10, handlelength=2.4)

        # 所有说明放在轴标题中，不在坐标标签附近叠加 figure.text。
        phase_axis = figure.add_axes([0.075, 0.490, 0.21, 0.37333])
        arrays = result.aberrated.arrays
        coordinate = arrays["pupil_coordinate_m"]*1e3
        visible = arrays["input_intensity_W_m2"] > arrays["input_intensity_W_m2"].max()*1e-4
        phase = np.ma.array(arrays["input_phase_rad"], mask=~visible)
        palette = plt.get_cmap("twilight_shifted").copy()
        palette.set_bad("#EEEEEE")
        phase_image = phase_axis.imshow(phase, origin="lower", extent=[coordinate[0], coordinate[-1]]*2,
                                        cmap=palette, vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        phase_limit = 1.13*max(cfg.zernike_radius_mm, cfg.beam_diameter_mm/2)
        phase_axis.set(xlim=(-phase_limit, phase_limit), ylim=(-phase_limit, phase_limit),
                       xlabel="x (mm)", ylabel="y (mm)")
        phase_axis.set_title("输入相位（加载后）\n"
                             f"RMS = {result.metadata['zernike_rms_excluding_piston_waves']:.3g} waves",
                             fontsize=11, pad=12, linespacing=1.6)
        phase_bar = figure.colorbar(phase_image, cax=figure.add_axes([0.302, 0.490, 0.007, 0.37333]))
        phase_bar.set_ticks([-np.pi, 0, np.pi], labels=["−π", "0", "π"])
        phase_bar.ax.set_title("rad", fontsize=9, pad=10)
        phase_bar.outline.set_visible(False)
        phase_bar.ax.tick_params(length=0, labelsize=9)

        reference_peak = result.baseline.metrics["peak_intensity_W_m2"]
        shared_max = max(1.0, *(case.metrics["peak_intensity_W_m2"]/reference_peak for case in cases))
        focal_artists = []
        for index, case in enumerate(cases):
            axis = figure.add_axes([0.390+index*0.315, 0.490, 0.21, 0.37333])
            point = case.arrays["presentation_coordinate_m"]*1e6
            image = case.arrays["presentation_focal_image_W_m2"]/reference_peak
            focal_artists.append(axis.imshow(image, origin="lower", extent=[point[0], point[-1]]*2,
                                              cmap="magma", vmin=0, vmax=shared_max, interpolation="nearest"))
            axis.set(xlabel="x (μm)", ylabel="y (μm)")
            axis.set_title(("无像差焦斑", "有像差焦斑")[index]+"\n"
                           f"z = f = {cfg.focal_length_mm:g} mm",
                           fontsize=11, pad=12, linespacing=1.6)
        intensity_bar = figure.colorbar(focal_artists[1], cax=figure.add_axes([0.932, 0.490, 0.007, 0.37333]))
        intensity_bar.set_ticks([0, shared_max/2, shared_max])
        intensity_bar.ax.set_title("I / I₀", fontsize=9, pad=10)
        intensity_bar.outline.set_visible(False)
        intensity_bar.ax.tick_params(length=0, labelsize=9)

        for index, direction in enumerate(("x", "y", "z")):
            axis = figure.add_axes([0.075+index*0.315, 0.175, 0.245, 0.210])
            axis.set_title(("横向 x 截面", "横向 y 截面", "轴向光强")[index], fontsize=11, pad=12)
            for case, color, style in zip(cases, (BLUE, ORANGE), ("--", "-")):
                if direction == "z":
                    positions = case.arrays["axial_coordinate_m"]*1e6
                    values = case.arrays["axial_intensity_W_m2"]
                else:
                    positions = case.arrays[f"{direction}_line_m"]*1e6-case.metrics[f"peak_{direction}_um"]
                    values = case.arrays[f"{direction}_intensity_W_m2"]
                axis.plot(positions, values/max(float(values.max()), 1e-300),
                          color=color, linestyle=style, linewidth=1.8)
            axis.axhline(0.5, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=0)
            if direction != "z":
                widths = [case.metrics[f"lateral_fwhm_{direction}_um"] for case in cases]
                if all(width is not None for width in widths):
                    limit = 2.1*max(widths)
                    axis.set_xlim(-limit, limit)
                axis.set_xlabel("相对各自主峰 (μm)", fontsize=10, labelpad=7)
                metric_key = f"lateral_fwhm_{direction}_um"
            else:
                widths = [case.metrics["axial_fwhm_um"] for case in cases]
                if all(width is not None for width in widths):
                    left = min(case.metrics["axial_half_max_left_um"]-0.6*width for case, width in zip(cases, widths))
                    right = max(case.metrics["axial_half_max_right_um"]+0.6*width for case, width in zip(cases, widths))
                    axis.set_xlim(max(left, positions[0]), min(right, positions[-1]))
                axis.set_xlabel("相对后焦面 Δz (μm)", fontsize=10, labelpad=7)
                metric_key = "axial_fwhm_um"
            axis.set(ylim=(-0.035, 1.08), yticks=[0, 0.5, 1])
            axis.set_ylabel("I / 各自峰值", fontsize=10, labelpad=7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=9, length=3)
            figure.text(0.1975+index*0.315, 0.065,
                        f"FWHM:  {_number(cases[0].metrics[metric_key])} → {_number(cases[1].metrics[metric_key])} μm",
                        ha="center", fontsize=11)
        figure.text(0.06, 0.018, "焦斑共用无像差峰值 I₀ 与空间坐标；曲线分别归一化；相位范围 [−π, π]。",
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
    arrays = {"baseline_"+name: value for name, value in result.baseline.arrays.items()}
    if result.aberrated is not result.baseline:
        arrays.update({"aberrated_"+name: value for name, value in result.aberrated.arrays.items()})
    arrays["aberrated_identical_to_baseline"] = np.array(result.aberrated is result.baseline)
    np.savez_compressed(directory/"fields.npz", **arrays)
    return plot_result(result, directory/"overview.png")
