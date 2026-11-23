"""只包含光场操作；内部长度全部使用 m。"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy.signal import CZT


def power(field, spacing):
    return float(np.sum(np.abs(field)**2) * spacing**2)


def fresnel_propagate(field, spacing, wavelength, distance):
    """同采样间距的 Fresnel 角谱传播，省略不影响光强的整体相位。

    H = exp[-i*pi*lambda*z*(fx²+fy²)]。FFT 隐含周期边界；调用者
    必须提供足够的计算窗口并检查采样、边缘能量及收敛。
    """
    frequency = fftfreq(field.shape[0], d=spacing)
    transfer = np.exp(-1j*np.pi*wavelength*distance * (
        frequency[:, None]**2 + frequency[None, :]**2
    ))
    return fftshift(ifft2(fft2(ifftshift(field))*transfer))


def _transform(field, coordinate, frequency_start, frequency_step, samples, axis):
    """在指定等间距空间频率处求离散 Fourier 积分。

    CZT 仅改变 Fourier 和的求值位置，不增加输入场分辨率。
    使用中心坐标时，显式恢复坐标原点带来的线性相位。
    """
    spacing = coordinate[1] - coordinate[0]
    operator = CZT(
        n=coordinate.size, m=samples,
        a=np.exp(2j*np.pi*frequency_start*spacing),
        w=np.exp(-2j*np.pi*frequency_step*spacing),
    )
    transformed = operator(field, axis=axis)
    frequencies = frequency_start + frequency_step*np.arange(samples)
    shape = [1]*transformed.ndim
    shape[axis] = samples
    return transformed * np.exp(-2j*np.pi*frequencies*coordinate[0]).reshape(shape)


class FocusedField:
    """从透镜前复光场计算透镜后任意 z>0 的 Fresnel 场。

    将薄透镜二次相位与传播积分的二次相位解析合并，避免在
    毫米口径网格上直接欠采样透镜相位；这不是省略透镜相位。
    观察面用独立的微米坐标网格，保留 dx²/(lambda*z) 绝对标度。
    """

    def __init__(self, pupil_field, coordinate, wavelength, focal_length):
        self.pupil = pupil_field
        self.coordinate = coordinate
        self.spacing = coordinate[1] - coordinate[0]
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.radius_squared = coordinate[:, None]**2 + coordinate[None, :]**2

    def _integrand(self, distance):
        if distance <= 0:
            raise ValueError("观察面必须位于透镜后方。")
        curvature = np.pi/self.wavelength * (1/distance - 1/self.focal_length)
        return self.pupil*np.exp(1j*curvature*self.radius_squared)

    def image(self, distance, output_coordinate, y_coordinate=None):
        if y_coordinate is None:
            y_coordinate = output_coordinate
        scale = self.wavelength*distance
        start = output_coordinate[0]/scale
        step = (output_coordinate[1] - output_coordinate[0])/scale
        result = _transform(self._integrand(distance), self.coordinate,
                            start, step, output_coordinate.size, axis=1)
        result = _transform(result, self.coordinate,
                            y_coordinate[0]/scale,
                            (y_coordinate[1]-y_coordinate[0])/scale,
                            y_coordinate.size, axis=0)
        # 输出面二次相位与整体相位不影响本程序的强度指标。
        return np.abs(result * (self.spacing**2/scale))**2

    def point_intensity(self, distance, x, y):
        phase_x = np.exp(-2j*np.pi*x*self.coordinate/(self.wavelength*distance))
        phase_y = np.exp(-2j*np.pi*y*self.coordinate/(self.wavelength*distance))
        value = np.sum(self._integrand(distance)*phase_y[:, None]*phase_x[None, :])
        return float(abs(value*self.spacing**2/(self.wavelength*distance))**2)

    def line(self, distance, output_coordinate, fixed, direction):
        integrand = self._integrand(distance)
        phase = np.exp(-2j*np.pi*fixed*self.coordinate/(self.wavelength*distance))
        collapsed = np.sum(integrand*phase[:, None], axis=0) if direction == "x" else (
            np.sum(integrand*phase[None, :], axis=1)
        )
        result = _transform(
            collapsed, self.coordinate, output_coordinate[0]/(self.wavelength*distance),
            (output_coordinate[1] - output_coordinate[0])/(self.wavelength*distance),
            output_coordinate.size, axis=0,
        )
        return np.abs(result*self.spacing**2/(self.wavelength*distance))**2

    def longitudinal_sections(self, distances, offsets, x_fixed, y_fixed, include_y=True,
                              progress=None):
        """固定平面上的 XZ/YZ 强度；共享每个 z 的积分核，避免计算完整体数据。

        XZ: y=y_fixed，横坐标 x=x_fixed+offsets。
        YZ: x=x_fixed，横坐标 y=y_fixed+offsets。不随 z 跟踪峰值。
        """
        xz = np.empty((len(distances), len(offsets)))
        yz = np.empty_like(xz) if include_y else None
        for index, distance in enumerate(distances):
            integrand = self._integrand(distance)
            scale = self.wavelength*distance
            y_phase = np.exp(-2j*np.pi*y_fixed*self.coordinate/scale)
            collapsed_x = np.sum(integrand*y_phase[:, None], axis=0)
            result_x = _transform(collapsed_x, self.coordinate, (x_fixed+offsets[0])/scale,
                                  (offsets[1]-offsets[0])/scale, len(offsets), axis=0)
            xz[index] = np.abs(result_x*self.spacing**2/scale)**2
            if include_y:
                x_phase = np.exp(-2j*np.pi*x_fixed*self.coordinate/scale)
                collapsed_y = np.sum(integrand*x_phase[None, :], axis=1)
                result_y = _transform(collapsed_y, self.coordinate, (y_fixed+offsets[0])/scale,
                                      (offsets[1]-offsets[0])/scale, len(offsets), axis=0)
                yz[index] = np.abs(result_y*self.spacing**2/scale)**2
            if progress is not None and index in (0, len(distances)//2, len(distances)-1):
                progress(f"纵向切面 {index+1}/{len(distances)}")
        return xz, yz
