"""
Splits a body torque across the reaction wheels.

In the Basilisk simulation this is done by the rwMotorTorque module. This numpy version does the same,
we use it in the test simulation and for the PD controller in wheel_torque mode.

Sign convention: the wheels push on the hub with -Gs u (action = reaction).
So to get the body torque L we need Gs u = -L. With more wheels than axes there are many solutions,
we take the one with the smallest wheel torques: u = -pinv(Gs) L (same as rwMotorTorque).
"""

import numpy as np


def rw_axes_matrix(spin_axes_B):
    """Spin axes (one row per wheel) -> Gs matrix (one column per wheel), normalized to length 1"""
    Gs = np.asarray(spin_axes_B, dtype=float).T.copy()
    return Gs / np.linalg.norm(Gs, axis=0, keepdims=True)


def allocate_body_torque(L_B, Gs):
    """Wheel motor torques u [Nm] that create the body torque L_B [Nm] on the hub"""
    return -np.linalg.pinv(Gs) @ np.asarray(L_B, dtype=float)
