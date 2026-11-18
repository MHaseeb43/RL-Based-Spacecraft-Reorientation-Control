"""
Reaction wheel torque allocation (WP 2200 / WP 3100).

In the Basilisk simulation the commanded body torque is split across the wheels by the FSW module
`rwMotorTorque` (supervisor decision). This numpy version reproduces it for the stand-in test
simulator and for the PD baseline in wheel-torque mode.

Convention: the wheels act on the hub with -Gs u. For a desired body torque L_B, rwMotorTorque is
given the "required" torque Lr = -L_B and solves Gs u = Lr with the minimum-norm solution
u = Gs^T (Gs Gs^T)^-1 Lr (all three body axes controlled), i.e. u = -pinv(Gs) L_B.
"""

import numpy as np


def rw_axes_matrix(spin_axes_B):
    """(N, 3) spin axes -> Gs (3, N) with unit-length columns."""
    Gs = np.asarray(spin_axes_B, dtype=float).T.copy()
    return Gs / np.linalg.norm(Gs, axis=0, keepdims=True)


def allocate_body_torque(L_B, Gs):
    """Minimum-norm RW motor torques u (N,) [N m] that produce the body torque L_B (3,) on the hub."""
    return -np.linalg.pinv(Gs) @ np.asarray(L_B, dtype=float)
