"""
Attitude kinematics utilities (WP 1100).

Conventions (same as Basilisk):
- Modified Rodrigues Parameters (MRP) sigma, kept in the short-rotation set |sigma| <= 1.
- Quaternions are scalar-first: q = [q0, q1, q2, q3].
- DCM [BN] maps vectors from frame N to frame B: v_B = [BN] v_N.
"""

import numpy as np


def tilde(v):
    """Skew-symmetric cross-product matrix [v~] so that [v~] w = v x w."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def mrp_shadow(sigma):
    """Map an MRP to the short-rotation set (|sigma| <= 1)."""
    sigma = np.asarray(sigma, dtype=float)
    s2 = sigma @ sigma
    if s2 > 1.0:
        return -sigma / s2
    return sigma


def mrp_to_quat(sigma):
    """MRP -> quaternion (scalar first, q0 >= 0 for the short-rotation set)."""
    sigma = np.asarray(sigma, dtype=float)
    s2 = sigma @ sigma
    return np.concatenate(([1.0 - s2], 2.0 * sigma)) / (1.0 + s2)


def quat_to_mrp(q):
    """Quaternion (scalar first) -> MRP in the short-rotation set."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q[1:] / (1.0 + q[0])


def mrp_to_dcm(sigma):
    """MRP -> DCM, e.g. sigma_BN -> [BN]."""
    sigma = np.asarray(sigma, dtype=float)
    s2 = sigma @ sigma
    st = tilde(sigma)
    return np.eye(3) + (8.0 * st @ st - 4.0 * (1.0 - s2) * st) / (1.0 + s2) ** 2


def dcm_to_quat(C):
    """DCM -> quaternion (scalar first) using Sheppard's method."""
    tr = np.trace(C)
    b2 = np.array([1.0 + tr,
                   1.0 + 2.0 * C[0, 0] - tr,
                   1.0 + 2.0 * C[1, 1] - tr,
                   1.0 + 2.0 * C[2, 2] - tr]) / 4.0
    i = int(np.argmax(b2))
    q = np.zeros(4)
    q[i] = np.sqrt(b2[i])
    # Off-diagonal products b_i * b_j
    p01 = (C[1, 2] - C[2, 1]) / 4.0
    p02 = (C[2, 0] - C[0, 2]) / 4.0
    p03 = (C[0, 1] - C[1, 0]) / 4.0
    p12 = (C[0, 1] + C[1, 0]) / 4.0
    p13 = (C[2, 0] + C[0, 2]) / 4.0
    p23 = (C[1, 2] + C[2, 1]) / 4.0
    if i == 0:
        q[1], q[2], q[3] = p01 / q[0], p02 / q[0], p03 / q[0]
    elif i == 1:
        q[0], q[2], q[3] = p01 / q[1], p12 / q[1], p13 / q[1]
    elif i == 2:
        q[0], q[1], q[3] = p02 / q[2], p12 / q[2], p23 / q[2]
    else:
        q[0], q[1], q[2] = p03 / q[3], p13 / q[3], p23 / q[3]
    if q[0] < 0.0:
        q = -q
    return q


def dcm_to_mrp(C):
    """DCM -> MRP in the short-rotation set."""
    return quat_to_mrp(dcm_to_quat(C))


def mrp_error(sigma_BN, sigma_RN):
    """Attitude error sigma_BR of body frame B relative to reference frame R."""
    return dcm_to_mrp(mrp_to_dcm(sigma_BN) @ mrp_to_dcm(sigma_RN).T)


def mrp_add(sigma_BR, sigma_RN):
    """Compose rotations: sigma_BN from sigma_BR and sigma_RN ([BN] = [BR][RN])."""
    return dcm_to_mrp(mrp_to_dcm(sigma_BR) @ mrp_to_dcm(sigma_RN))


def principal_angle(sigma):
    """Principal rotation angle [rad] in [0, pi] of an MRP."""
    sigma = mrp_shadow(sigma)
    return 4.0 * np.arctan(np.linalg.norm(sigma))


def random_unit_vector(rng):
    """Uniformly distributed unit vector on the sphere."""
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def random_mrp_with_angle(angle, rng):
    """MRP with principal angle `angle` [rad] about a uniformly random axis."""
    return random_unit_vector(rng) * np.tan(angle / 4.0)


def random_mrp(rng):
    """Uniformly distributed random attitude as an MRP (short-rotation set)."""
    q = rng.standard_normal(4)
    return quat_to_mrp(q / np.linalg.norm(q))
