"""
Classical PD (MRP feedback) attitude controller baseline (WP 3400 / WP 5300).

Desired body torque:   L_B = -K sigma_BR - P omega_BR_B
Wheel motor torques:   the reaction torque of wheel motor torques u on the hub is -Gs u,
                       so u = -pinv(Gs) L_B (minimum-norm distribution over N wheels).

Basilisk also provides this chain as FSW modules (mrpFeedback + rwMotorTorque); this
numpy version keeps the baseline independent of the simulation backend.
"""

import numpy as np


class PDController:
    def __init__(self, K, P, spin_axes_B, u_max):
        """
        Args:
            K: proportional gain [N m]
            P: derivative gain [N m s]
            spin_axes_B: (N, 3) RW spin axes in body frame (nominal, as known to the controller)
            u_max: max motor torque per wheel [N m]
        """
        self.K = float(K)
        self.P = float(P)
        Gs = np.asarray(spin_axes_B, dtype=float).T  # (3, N)
        Gs /= np.linalg.norm(Gs, axis=0, keepdims=True)
        self.Gs_pinv = np.linalg.pinv(Gs)            # (N, 3)
        self.u_max = float(u_max)

    def wheel_torques(self, sigma_BR, omega_BR_B):
        """RW motor torques [N m]; scaled uniformly (direction preserving) if any wheel saturates."""
        L_B = -self.K * np.asarray(sigma_BR) - self.P * np.asarray(omega_BR_B)
        u = -self.Gs_pinv @ L_B
        peak = np.max(np.abs(u))
        if peak > self.u_max:
            u *= self.u_max / peak
        return u

    def __call__(self, info):
        """Normalized action in [-1, 1] from the env `info` dict (same interface as an RL policy)."""
        return self.wheel_torques(info["sigma_BR"], info["omega_BR_B"]) / self.u_max
