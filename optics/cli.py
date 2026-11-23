"""两个独立程序的公共输入与启动逻辑；对比程序单独封装。"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .config import Numerics, ZernikeTerm
from .display import configure_display
from .simulation import simulate_aberration, simulate_diffraction


def _term(text):
    try:
        n, m, coefficient = text.split(",")
        return ZernikeTerm(int(n), int(m), float(coefficient))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(f"像差格式 n,m,系数waves，例如 3,1,0.15：{error}") from error


def _parser(program):
    label = "无像差衍射计算" if program == "diffraction" else "前焦面 Zernike 像差计算"
    parser = argparse.ArgumentParser(description=f"{label}；观察面固定在透镜后 z=f。")
    for name in ("wavelength-nm", "lens-diameter-mm", "focal-length-mm", "beam-diameter-mm"):
        parser.add_argument("--"+name, type=float)
    parser.add_argument("--beam", choices=("gaussian", "uniform", "both"))
    if program == "aberration":
        parser.add_argument("--zernike-radius-mm", type=float)
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--zernike", type=_term, action="append", help="n,m,系数waves；可重复，1=2π rad RMS")
        group.add_argument("--no-aberrations", action="store_true", help="清空全部系数，验证与程序一一致")
    parser.add_argument("--grid-size", type=int, default=768)
    parser.add_argument("--image-samples", type=int, default=257)
    parser.add_argument("--axial-samples", type=int, default=121)
    for name in ("window-mm", "image-half-width-um", "axial-half-range-um"):
        parser.add_argument("--"+name, type=float)
    parser.add_argument("--output", type=Path, help="指定输出目录")
    parser.add_argument("--no-show", action="store_true", help="只保存结果，不显示图窗")
    parser.add_argument("--verbose", action="store_true", help="显示计算进度及数值诊断")
    return parser


def _run(default_config, program):
    parser = _parser(program)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    configure_display(no_show=args.no_show)
    from .report import print_summary, save_result
    import matplotlib.pyplot as plt

    names = ["wavelength_nm", "lens_diameter_mm", "focal_length_mm", "beam_diameter_mm"]
    if program == "aberration":
        names.append("zernike_radius_mm")
    updates = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if program == "diffraction":
        updates["aberrations"] = ()
    elif args.zernike is not None:
        updates["aberrations"] = tuple(args.zernike)
    elif args.no_aberrations:
        updates["aberrations"] = ()
    try:
        config = replace(default_config, **updates)
        numerics = Numerics(**{name: getattr(args, name) for name in (
            "grid_size", "image_samples", "axial_samples", "window_mm",
            "image_half_width_um", "axial_half_range_um")})
    except ValueError as error:
        parser.error(str(error))
    beam_types = ("gaussian", "uniform") if args.beam == "both" else (args.beam or config.beam_type,)
    root = args.output or Path(__file__).resolve().parents[1]/"results"/program
    calculate = simulate_diffraction if program == "diffraction" else simulate_aberration
    for beam in beam_types:
        callback = (lambda message: print(message, flush=True)) if args.verbose else None
        result = calculate(replace(config, beam_type=beam), numerics, progress=callback)
        print_summary(result, verbose=args.verbose)
        # 默认每种程序/光束只保留最新结果，避免再次积累运行目录。
        directory = root/beam if args.output is None or len(beam_types) > 1 else root
        save_result(result, directory)
        print(f"图片：{(directory/'overview.png').resolve()}\n", flush=True)
        if args.no_show:
            plt.close("all")
    if not args.no_show:
        plt.show(block=True)


def run_diffraction(default_config):
    _run(default_config, "diffraction")


def run_aberration(default_config):
    _run(default_config, "aberration")
