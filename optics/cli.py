"""三个入口的命令行接口；理论程序不加载图形和数值模拟依赖。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .config import Numerics, ZernikeTerm
from .theory import calculate_resolution


def _term(text):
    try:
        n, m, coefficient = text.split(",")
        return ZernikeTerm(int(n), int(m), float(coefficient))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(f"像差格式 n,m,系数waves，例如 3,1,0.15：{error}") from error


def _parser(program):
    labels = {"resolution": "初始介质与样品介质的理想近轴分辨率（纯数值）",
              "fourier_focus": "均匀圆光束的后焦面 Fourier 聚焦",
              "zernike_focus": "瞳面共轭面加载 Zernike 相位后的 Fourier 聚焦"}
    parser = argparse.ArgumentParser(description=labels[program])
    parser.add_argument("--na", "--numerical-aperture", dest="numerical_aperture", type=float,
                        help="直接输入有效 NA，默认均匀光充分填充；忽略透镜直径、焦距和光束直径")
    if program == "resolution":
        medium = parser.add_mutually_exclusive_group()
        medium.add_argument("--immersion-medium", help="初始介质：air、water、oil 或数值折射率；默认 air")
        medium.add_argument("--immersion-refractive-index", dest="immersion_medium", type=float,
                            help="直接指定初始介质折射率，与 --immersion-medium 二选一")
        parser.add_argument("--sample-refractive-index", type=float, help="样品介质折射率，默认 1.37")
    for name in ("wavelength-nm", "lens-diameter-mm", "focal-length-mm", "beam-diameter-mm"):
        parser.add_argument("--"+name, type=float,
                            help="瞳面等效光束直径，默认随透镜直径变化" if name == "beam-diameter-mm" else None)
    if program == "zernike_focus":
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--zernike", type=_term, action="append", help="n,m,系数waves；可重复输入")
        group.add_argument("--no-aberrations", action="store_true", help="将相位清零，验证与程序二一致")
    if program != "resolution":
        parser.add_argument("--grid-size", type=int, help="瞳面采样边长，默认 512")
        parser.add_argument("--padding-factor", type=int, help="FFT 零填充倍率，默认 8")
        parser.add_argument("--image-half-width-um", type=float, help="后焦面显示半宽，默认自动覆盖主峰")
        parser.add_argument("--axial-samples", type=int, help="轴向扫描点数，不小于 33 的奇数，默认 241")
        parser.add_argument("--axial-half-range-um", type=float, help="相对参考焦面的轴向扫描半范围，默认自动设置")
        parser.add_argument("--section-samples", type=int, help="XZ/YZ 横向采样点数，默认 257，要求奇数")
        parser.add_argument("--section-half-width-um", type=float, help="XZ/YZ 相对固定纵线的横向半宽，默认自动设置")
        parser.add_argument("--no-show", action="store_true", help="仅保存结果，不弹出图窗")
        parser.add_argument("--verbose", action="store_true", help="打印采样与功率诊断")
    parser.add_argument("--output", type=Path, help="输出目录，默认 results/程序名")
    return parser


def _arguments(default_config, program):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser(program)
    args = parser.parse_args()
    names = ("wavelength_nm", "lens_diameter_mm", "focal_length_mm", "beam_diameter_mm", "numerical_aperture")
    if program == "resolution":
        names += ("immersion_medium", "sample_refractive_index")
    updates = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if program != "zernike_focus":
        updates["aberrations"] = ()
    elif args.zernike is not None:
        updates["aberrations"] = tuple(args.zernike)
    elif args.no_aberrations:
        updates["aberrations"] = ()
    try:
        config = replace(default_config, **updates)
    except ValueError as error:
        parser.error(str(error))
    directory = args.output or Path(__file__).resolve().parents[1]/"results"/program
    return parser, args, config, directory


def _save_json(data, destination):
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_resolution(default_config):
    parser, args, config, directory = _arguments(default_config, "resolution")
    sample_refractive_index = config.sample_refractive_index
    try:
        result = calculate_resolution(config, sample_refractive_index)
    except ValueError as error:
        parser.error(str(error))
    lines = ["均匀圆光束：理想近轴分辨率"]
    if config.uses_na:
        lines.extend([
            f"真空波长：{config.wavelength_nm:g} nm；直接输入有效 NA：{config.effective_na:.6g}",
            "NA 模式：均匀光充分填充；忽略透镜直径、焦距和光束直径。",
        ])
    else:
        lines.extend([
            f"真空波长：{config.wavelength_nm:g} nm；初始介质中的工作焦距：{config.focal_length_mm:g} mm",
            f"透镜直径：{config.lens_diameter_mm:g} mm；光束直径：{config.beam_radius*2e3:g} mm",
            f"有效直径：{config.effective_radius*2e3:g} mm；有效 NA：{config.effective_na:.6g}",
        ])
    lines.append("介质                 横向 Rayleigh (μm)    纵向估计 (μm，非 FWHM)")
    for key, label in (("initial_medium", f"{config.immersion_label} n={config.immersion_refractive_index:g}"),
                       ("sample_medium", f"样品 n={sample_refractive_index:g}")):
        values = result["results"][key]
        lines.append(f"{label:<18} {values['lateral_rayleigh_um']:>12.6g}          {values['axial_resolution_um']:>12.6g}")
    ratio = sample_refractive_index/config.immersion_refractive_index
    lines.append(f"进入样品后：横向不变，轴向数值为初始介质的 {ratio:.6g} 倍（固定入射 NA 的理想近轴模型）。")
    lines.extend("提示："+message for message in result["diagnostics"])
    directory.mkdir(parents=True, exist_ok=True)
    _save_json(result, directory/"summary.json")
    (directory/"summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"数值结果：{(directory/'summary.txt').resolve()}")


def run_focus(default_config, *, aberrated=False, default_numerics=None):
    program = "zernike_focus" if aberrated else "fourier_focus"
    parser, args, config, directory = _arguments(default_config, program)
    try:
        numerics = replace(default_numerics or Numerics(), **{
            name: getattr(args, name) for name in (
                "grid_size", "padding_factor", "image_half_width_um", "axial_samples", "axial_half_range_um",
                "section_samples", "section_half_width_um")
            if getattr(args, name) is not None
        })
    except ValueError as error:
        parser.error(str(error))
    from .display import configure_display, prepare_interactive_figure
    configure_display(no_show=args.no_show)
    from .simulation import simulate_focus
    from .report import print_summary, save_result
    import matplotlib.pyplot as plt
    try:
        result = simulate_focus(config, numerics, aberrated=aberrated)
    except ValueError as error:
        parser.error(str(error))
    print_summary(result, verbose=args.verbose)
    figure = save_result(result, directory)
    print(f"图片：{(directory/'overview.png').resolve()}")
    if args.no_show:
        plt.close(figure)
    else:
        prepare_interactive_figure(figure)
        plt.show(block=True)
