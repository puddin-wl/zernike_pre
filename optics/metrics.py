"""固定后焦面主峰截面的半高全宽，不将其等同于 Rayleigh 分辨率。"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


def first_dark_ring(coordinate, field):
    """从零相位、中心对称圆孔径的 FFT 场截面测量第一暗环半径。

    去除整体相位后，场为实数。分别寻找主峰两侧第一次变号，
    在 FFT 场样本的三次样条上求零点，半径取两侧距离的平均。
    不使用波长、NA 或 Airy 解析系数；视场缺任一侧零点时返回 None。
    此方法不能直接套用于任意像差光斑。
    """
    missing = {"radius": None, "left": None, "right": None}
    field = np.asarray(field, dtype=complex)
    if len(coordinate) < 4:
        return missing
    peak = int(np.argmax(np.abs(field)))
    amplitude = abs(field[peak])
    if amplitude == 0:
        return missing
    reduced = field / field[peak]
    if np.max(np.abs(reduced.imag)) > 1e-8:
        return missing
    values = reduced.real
    left = peak
    right = peak
    while left > 0 and values[left] > 0:
        left -= 1
    while right < len(values)-1 and values[right] > 0:
        right += 1
    if values[left] > 0 or values[right] > 0:
        return missing
    # 在像素索引上求根，避免以 m 为变量时默认绝对容差影响精度。
    interpolant = CubicSpline(np.arange(len(values)), values)

    def root(i, j):
        index = brentq(interpolant, i, j, xtol=1e-12)
        return float(np.interp(index, np.arange(len(coordinate)), coordinate))

    lo = root(left, left+1)
    hi = root(right-1, right)
    return {"radius": (hi-lo)/2, "left": lo, "right": hi}


def half_max_regions(coordinate, intensity, *, peak_height=None, evaluator=None):
    """以绝对峰高的 50% 求全部连通区及最外侧交点跨度；不扣除局部谷底。

    多个区间之间可能低于半高，因此 outer_span 不是某个单峰的 FWHM。
    扫描边界截断的交点记为 None，整体跨度也返回 None，不外推。
    """
    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(intensity, dtype=float)
    half = float(np.max(values) if peak_height is None else peak_height)/2
    missing = {"intervals": [], "outer_left": None, "outer_right": None, "outer_span": None}
    if half <= 0:
        return missing
    above = values >= half
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    ends = np.flatnonzero(above & ~np.r_[above[1:], False])

    def crossing(i, j):
        if evaluator is not None:
            span = coordinate[j]-coordinate[i]
            fraction = brentq(lambda t: evaluator(coordinate[i]+t*span)-half,
                              0, 1, xtol=1e-10)
            return float(coordinate[i]+fraction*span)
        return float(coordinate[i] + (half-values[i]) * (
            coordinate[j]-coordinate[i]) / (values[j]-values[i]))

    intervals = []
    for start, end in zip(starts, ends):
        lo = crossing(start-1, start) if start > 0 else (
            float(coordinate[0]) if values[0] == half else None
        )
        hi = crossing(end, end+1) if end < len(values)-1 else (
            float(coordinate[-1]) if values[-1] == half else None
        )
        intervals.append((lo, hi))
    if not intervals:
        return missing
    outer_left, outer_right = intervals[0][0], intervals[-1][1]
    span = None if outer_left is None or outer_right is None else outer_right-outer_left
    return {"intervals": intervals, "outer_left": outer_left, "outer_right": outer_right, "outer_span": span}


def fwhm(coordinate, intensity, peak_index=None, *, evaluator=None):
    """所选主峰的连通半高区；缺交点返回 None，不跨越低于半高的间隙。"""
    values = np.asarray(intensity, dtype=float)
    index = int(np.argmax(values)) if peak_index is None else peak_index
    profile = half_max_regions(coordinate, values, peak_height=values[index], evaluator=evaluator)
    regions = len(profile["intervals"])
    position = coordinate[index]
    selected = next(((lo, hi) for lo, hi in profile["intervals"]
                     if (lo is None or lo <= position) and (hi is None or position <= hi)), (None, None))
    lo, hi = selected
    if lo is None or hi is None:
        return {"width": None, "left": None, "right": None, "half_height_regions": regions}
    return {"width": hi-lo, "left": lo, "right": hi, "half_height_regions": regions}
