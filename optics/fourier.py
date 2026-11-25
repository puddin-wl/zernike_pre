"""瞳面至后焦面的一次二维 FFT；保留积分幅度及实际长度尺度。"""

import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifftshift


def focus_fft(pupil, spacing, wavelength, focal_length, padding_factor, *, numerical_aperture=None):
    """零填充只细化焦面采样，不提高真实分辨能力或瞳面采样精度。

    省略不影响强度的单位模相位因子。返回 x=y 坐标、W/m² 光强，
    以及穿过原点的 x/y 复振幅截面，用于从数值场定位第一暗环。
    NA 模式下 pupil 坐标为无量纲单位圆，尺度为 λ/NA，不使用焦距。
    """
    size = pupil.shape[0]
    padded_size = size * padding_factor
    pad = (padded_size-size)//2
    padded = np.pad(pupil, ((pad, pad), (pad, pad)))
    transformed = fftshift(fft2(ifftshift(padded), overwrite_x=True))
    frequency_scale = (numerical_aperture/wavelength if numerical_aperture is not None else
                       1/(wavelength*focal_length))
    transformed *= spacing**2 * frequency_scale
    intensity = np.abs(transformed)**2
    coordinate = fftshift(fftfreq(padded_size, d=spacing)) / frequency_scale
    center = padded_size//2
    return coordinate, intensity, transformed[center, :].copy(), transformed[:, center].copy()
