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


def fwhm(coordinate, intensity, peak_index=None, *, evaluator=None):
    """所选主峰的连通半高区；缺交点返回 None，不外推。"""
    values = np.asarray(intensity, dtype=float)
    index = int(np.argmax(values)) if peak_index is None else peak_index
    half = float(values[index]) / 2
    above = values >= half
    regions = int(np.count_nonzero(above & ~np.r_[False, above[:-1]]))
    left = right = index
    while left > 0 and values[left] >= half:
        left -= 1
    while right < len(values)-1 and values[right] >= half:
        right += 1
    if half <= 0 or values[left] >= half or values[right] >= half:
        return {"width": None, "left": None, "right": None, "half_height_regions": regions}

    def crossing(i, j):
        if evaluator is not None:
            span = coordinate[j]-coordinate[i]
            fraction = brentq(lambda t: evaluator(coordinate[i]+t*span)-half,
                              0, 1, xtol=1e-10)
            return float(coordinate[i]+fraction*span)
        return float(coordinate[i] + (half-values[i]) * (
            coordinate[j]-coordinate[i]) / (values[j]-values[i]))

    lo, hi = crossing(left, left+1), crossing(right-1, right)
    return {"width": hi-lo, "left": lo, "right": hi, "half_height_regions": regions}
