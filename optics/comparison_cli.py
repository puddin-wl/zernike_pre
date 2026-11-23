"""参数解析封装在这里，主程序只保留物理参数。"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .config import Numerics, ZernikeTerm
from .display import configure_display
from .simulation import simulate


def _term(text):
    try:
        n, m, coefficient = text.split(",")
        return ZernikeTerm(int(n), int(m), float(coefficient))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(f"像差格式为 n,m,系数waves，例如 3,1,0.15：{error}") from error


def run(default_config):
    parser = argparse.ArgumentParser(description="前焦面相位加载、空气传播与有限口径薄透镜聚焦。")
    parser.add_argument("--wavelength-nm", type=float)
    parser.add_argument("--lens-diameter-mm", type=float)
    parser.add_argument("--focal-length-mm", type=float)
    parser.add_argument("--beam", choices=("gaussian", "uniform", "both"))
    parser.add_argument("--beam-diameter-mm", type=float)
    parser.add_argument("--zernike-radius-mm", type=float)
    aberrations = parser.add_mutually_exclusive_group()
    aberrations.add_argument("--zernike", type=_term, action="append", help="n,m,系数waves；1=2π rad RMS，可重复")
    aberrations.add_argument("--no-aberrations", action="store_true", help="清空主程序中的所有像差")
    parser.add_argument("--verbose", action="store_true", help="显示计算进度及完整数值诊断")
    parser.add_argument("--grid-size", type=int, default=768)
    parser.add_argument("--image-samples", type=int, default=257)
    parser.add_argument("--axial-samples", type=int, default=121)
    parser.add_argument("--window-mm", type=float)
    parser.add_argument("--image-half-width-um", type=float)
    parser.add_argument("--axial-half-range-um", type=float)
    parser.add_argument("--output", type=Path, help="输出目录；默认在项目 results 下创建独立运行目录")
    parser.add_argument("--no-show", action="store_true", help="仅保存图，不弹出绘图窗口")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    configure_display(no_show=args.no_show)
    from .comparison_report import print_summary, save_result
    import matplotlib.pyplot as plt

    updates = {name: getattr(args, name) for name in (
        "wavelength_nm", "lens_diameter_mm", "focal_length_mm", "beam_diameter_mm",
        "zernike_radius_mm") if getattr(args, name) is not None}
    if args.zernike is not None:
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
    modes = ("gaussian", "uniform") if args.beam == "both" else (args.beam or config.beam_type,)
    root = args.output or Path(__file__).resolve().parents[1]/"results"/"comparison"
    for mode in modes:
        callback = (lambda message: print(message, flush=True)) if args.verbose else None
        result = simulate(replace(config, beam_type=mode), numerics, progress=callback)
        print_summary(result, verbose=args.verbose)
        directory = root/mode if len(modes) > 1 else root
        save_result(result, directory)
        print(f"\n结果已保存：{directory.resolve()}\n", flush=True)
        if args.no_show:
            plt.close("all")
    if not args.no_show:
        plt.show(block=True)
