"""单位圆面积正交归一化实 Zernike；m>0 为 cos，m<0 为 sin。"""

import numpy as np
from scipy.special import eval_jacobi


def zernike(n, m, rho, theta):
    magnitude = abs(m)
    p = (n - magnitude) // 2
    radial = (-1.0)**p * rho**magnitude * eval_jacobi(
        p, magnitude, 0, 1 - 2*rho*rho
    )
    if m == 0:
        return np.sqrt(n + 1) * radial
    angular = np.cos(m*theta) if m > 0 else np.sin(magnitude*theta)
    return np.sqrt(2*(n + 1)) * radial * angular


def phase_on_beam(x, y, beam_radius, terms):
    """以完整光束圆为单位圆计算未包裹相位；圆外无入射光。"""
    rho = np.hypot(x, y) / beam_radius
    inside = rho <= 1
    theta = np.broadcast_to(np.arctan2(y, x), rho.shape)
    phase = np.zeros_like(rho)
    for term in terms:
        if term.coefficient_waves:
            phase[inside] += 2*np.pi*term.coefficient_waves*zernike(
                term.n, term.m, rho[inside], theta[inside]
            )
    return phase
