import numpy as np
from scipy.special import eval_jacobi


def zernike(n, m, rho, theta):
    """实 Zernike；单位圆面积平均的平方为 1，活塞以外均值为 0。"""
    magnitude = abs(m)
    p = (n - magnitude) // 2
    radial = (-1.0)**p * rho**magnitude * eval_jacobi(
        p, magnitude, 0, 1 - 2*rho*rho
    )
    if m == 0:
        return np.sqrt(n + 1) * radial
    angular = np.cos(m*theta) if m > 0 else np.sin(magnitude*theta)
    return np.sqrt(2*(n + 1)) * radial * angular


def wavefront(x, y, radius, terms, wavelength):
    """圆内按指定基底加载 OPD，圆外 OPD=0；不修改振幅。

    系数输入为 waves，OPD = wavelength * sum(c_j * Z_j)。
    这是有限圆形相位调制区，不是圆形挡光孔径。边界的相位跳变
    属于所指定相位屏的一部分，会参与传播和衍射。
    """
    rho = np.hypot(x, y) / radius
    inside = rho <= 1
    theta_inside = np.arctan2(y + np.zeros_like(x), x + np.zeros_like(y))[inside]
    opd = np.zeros_like(rho)
    rho_inside = rho[inside]
    for term in terms:
        if term.coefficient_waves:
            opd[inside] += term.coefficient_waves * wavelength * zernike(
                term.n, term.m, rho_inside, theta_inside
            )
    return opd
