"""
Attitude math helpers.

Same conventions as Basilisk:
- MRPs (sigma) are kept in the short rotation set, |sigma| <= 1
- quaternions are scalar first: q = [q0, q1, q2, q3]
- the DCM [BN] turns a vector from frame N into frame B: v_B = [BN] v_N
"""

import numpy as np


def tilde(v):
    """Cross product matrix, so that tilde(v) @ w is the same as np.cross(v, w)"""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def mrp_shadow(sigma):
    """Switch to the shadow set if |sigma| > 1, so sigma always describes the short rotation"""
    sigma = np.asarray(sigma, dtype=float)
    s2 = sigma @ sigma
    if s2 > 1.0:
        return -sigma / s2
    return sigma


def mrp_to_quat(sigma):
    """MRP to quaternion. For |sigma| <= 1 we automatically get q0 >= 0"""
    sigma = np.asarray(sigma, dtype=float)
    s2 = sigma @ sigma
    return np.concatenate(([1.0 - s2], 2.0 * sigma)) / (1.0 + s2)


def quat_to_mrp(q):
    """Quaternion to MRP (short rotation)"""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q  # q and -q are the same rotation, take the one with q0 >= 0
    return q[1:] / (1.0 + q[0])


def mrp_to_dcm(sigma):
    """MRP to DCM, e.g. sigma_BN -> [BN]"""
    sigma = np.asarray(sigma, dtype=float)
    s2 = sigma @ sigma
    st = tilde(sigma)
    return np.eye(3) + (8.0 * st @ st - 4.0 * (1.0 - s2) * st) / (1.0 + s2) ** 2


def dcm_to_quat(C):
    """DCM to quaternion with Sheppard's method (works for every angle, no division by ~0)"""
    tr = np.trace(C)
    b2 = np.array([1.0 + tr,
                   1.0 + 2.0 * C[0, 0] - tr,
                   1.0 + 2.0 * C[1, 1] - tr,
                   1.0 + 2.0 * C[2, 2] - tr]) / 4.0

    # start with the biggest component, then get the other three from it
    i = int(np.argmax(b2))
    q = np.zeros(4)
    q[i] = np.sqrt(b2[i])

    # products of two components, b_i * b_j
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
    """DCM to MRP (short rotation)"""
    return quat_to_mrp(dcm_to_quat(C))


def mrp_error(sigma_BN, sigma_RN):
    """Attitude error sigma_BR: where the body is, relative to the reference (target)"""
    # [BR] = [BN] [RN]^T
    return dcm_to_mrp(mrp_to_dcm(sigma_BN) @ mrp_to_dcm(sigma_RN).T)


def mrp_add(sigma_BR, sigma_RN):
    """Opposite of mrp_error: sigma_BN from sigma_BR and sigma_RN"""
    # [BN] = [BR] [RN]
    return dcm_to_mrp(mrp_to_dcm(sigma_BR) @ mrp_to_dcm(sigma_RN))


def principal_angle(sigma):
    """Rotation angle of an MRP in rad, between 0 and pi"""
    sigma = mrp_shadow(sigma)
    return 4.0 * np.arctan(np.linalg.norm(sigma))


def random_unit_vector(rng):
    """Random direction, uniform on the sphere"""
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def random_mrp_with_angle(angle, rng):
    """MRP that rotates by `angle` [rad] about a random axis"""
    return random_unit_vector(rng) * np.tan(angle / 4.0)


def random_mrp(rng):
    """Random attitude, uniform over all orientations"""
    q = rng.standard_normal(4)  # a normalized 4D gaussian gives a uniform random quaternion
    return quat_to_mrp(q / np.linalg.norm(q))
