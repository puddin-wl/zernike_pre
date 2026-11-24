"""固定 XZ/YZ 平面的离焦光强切面；不生成三维体数据或额外二维 FFT。"""

import numpy as np


def longitudinal_sections(pupil, coordinate, config, numerics, z_offsets, x_fixed, y_fixed):
    half_width = (config.effective_na*float(np.max(np.abs(z_offsets)))
                  + 3*config.wavelength/config.effective_na)
    if numerics.section_half_width_um is not None:
        half_width = numerics.section_half_width_um*1e-6
    transverse_offsets = np.linspace(-half_width, half_width, numerics.section_samples)
    x, y = coordinate[None, :], coordinate[:, None]
    radius_squared = x*x+y*y
    projection = x*x_fixed+y*y_fixed
    spacing = coordinate[1]-coordinate[0]
    transform_argument = np.outer(transverse_offsets, coordinate)
    xz = np.empty((len(z_offsets), len(transverse_offsets)))
    yz = np.empty_like(xz)
    for index, offset in enumerate(z_offsets):
        z = config.focal_length+offset
        phase = np.pi/config.wavelength*((1/z-1/config.focal_length)*radius_squared-2*projection/z)
        plane = pupil*np.exp(1j*phase)
        # 先沿固定坐标方向求和，再在另一方向求一维 Fourier 积分。
        # 横向坐标 = 固定主峰坐标 + transverse_offsets，两切面共用同一计算核。
        kernel = np.exp(-2j*np.pi*transform_argument/(config.wavelength*z))
        factor = spacing**2/(config.wavelength*z)
        xz[index] = np.abs(np.einsum("ij,j->i", kernel, plane.sum(axis=0))*factor)**2
        yz[index] = np.abs(np.einsum("ij,j->i", kernel, plane.sum(axis=1))*factor)**2
    return {
        "section_z_offset_m": np.asarray(z_offsets),
        "xz_x_coordinate_m": x_fixed+transverse_offsets,
        "yz_y_coordinate_m": y_fixed+transverse_offsets,
        "xz_intensity_W_m2": xz,
        "yz_intensity_W_m2": yz,
    }
