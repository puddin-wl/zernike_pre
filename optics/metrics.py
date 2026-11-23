import numpy as np
from scipy.optimize import brentq


def peak_coordinate(coordinate, image):
    """最大像素附近的对数抛物线插值；多峰时选最强峰。"""
    row, column = np.unravel_index(np.argmax(image), image.shape)

    def refine(index, values):
        if index == 0 or index == coordinate.size - 1:
            return float(coordinate[index])
        left, center, right = np.log(np.maximum(values[index-1:index+2], 1e-300))
        denominator = left - 2*center + right
        offset = 0.0 if abs(denominator) < 1e-15 else 0.5*(left-right)/denominator
        return float(coordinate[index] + np.clip(offset, -0.5, 0.5)*(coordinate[1]-coordinate[0]))

    return refine(column, image[row, :]), refine(row, image[:, column])


def fwhm(coordinate, intensity, peak=None, evaluator=None):
    """选定主峰所在连通半高区的全宽；缺交点时返回 None，绝不外推。

    可提供直接场计算 evaluator，用求根代替采样间的线性插值。
    同时返回半高以上区域数量，提示单个 FWHM 是否难以描述多峰。
    """
    values = np.asarray(intensity, dtype=float)
    index = int(np.argmax(values)) if peak is None else int(np.argmin(abs(coordinate-peak)))
    maximum = float(values[index] if evaluator is None else evaluator(coordinate[index]))
    half = maximum/2
    above = values >= half
    regions = int(np.count_nonzero(above & ~np.r_[False, above[:-1]]))
    left = index
    right = index
    while left > 0 and values[left] >= half:
        left -= 1
    while right < len(values)-1 and values[right] >= half:
        right += 1
    if values[left] >= half or values[right] >= half or maximum <= 0:
        return {"width": None, "left": None, "right": None, "half_height_regions": regions}

    def crossing(i, j):
        if evaluator is not None:
            # 改用区间内局部坐标求根，避免 m 单位下默认绝对容差过大。
            span = coordinate[j]-coordinate[i]
            fraction = brentq(lambda t: evaluator(coordinate[i]+t*span)-half, 0, 1,
                              xtol=1e-10)
            return float(coordinate[i]+fraction*span)
        return float(coordinate[i] + (half-values[i]) * (
            coordinate[j]-coordinate[i])/(values[j]-values[i]))

    left_crossing = crossing(left, left+1)
    right_crossing = crossing(right-1, right)
    return {"width": right_crossing-left_crossing, "left": left_crossing,
            "right": right_crossing, "half_height_regions": regions}


def encircled_energy_radius(coordinate, image, center, total_power, fraction=0.8):
    radius = np.hypot(coordinate[None, :]-center[0], coordinate[:, None]-center[1])
    order = np.argsort(radius.ravel())
    pixel_area = (coordinate[1]-coordinate[0])**2
    cumulative = np.cumsum(image.ravel()[order])*pixel_area
    index = int(np.searchsorted(cumulative, fraction*total_power))
    radius_value = None if index == len(order) else float(radius.ravel()[order[index]])
    return radius_value, float(cumulative[-1]/total_power)
