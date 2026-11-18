"""
Classical PD (MRP feedback) attitude controller baseline (WP 3400 / WP 5300).

Desired body torque:   L_B = -K sigma_BR - P omega_BR_B   (same law as Basilisk mrpFeedback without integral term)

The PD controller uses the same action interface as the RL agent, so both are compared on identical terms:
- body_torque mode (default): outputs L_B; the simulator splits it across the wheels (rwMotorTorque).
- wheel_torque mode: splits L_B itself with the same minimum-norm allocation.
If the command exceeds the actuator range, it is scaled down uniformly (direction preserving).
"""

import numpy as np

from rl_reorientation.sim.allocation import allocate_body_torque, rw_axes_matrix


class PDController:
    def __init__(self, K, P, action_scale, spin_axes_B=None):
        """
        Args:
            K: proportional gain [N m]
            P: derivative gain [N m s]
            action_scale: physical value of action 1.0 (L_max [N m] for body torque, u_max [N m] for wheel torque)
            spin_axes_B: (N, 3) nominal RW spin axes -> wheel-torque output; None -> body-torque output
        """
        self.K = float(K)
        self.P = float(P)
        self.action_scale = float(action_scale)
        self.Gs = None if spin_axes_B is None else rw_axes_matrix(spin_axes_B)

    @classmethod
    def for_env(cls, env, K, P):
        """PD controller matching the action type of a ReorientationEnv."""
        axes = env.sc_cfg["reaction_wheels"]["spin_axes_B"] if env.action_type == "wheel_torque" else None
        return cls(K, P, env.action_scale, axes)

    def body_torque(self, sigma_BR, omega_BR_B):
        """Desired body torque L_B [N m]."""
        return -self.K * np.asarray(sigma_BR) - self.P * np.asarray(omega_BR_B)

    def __call__(self, info):
        """Normalized action in [-1, 1] from the env `info` dict (same interface as an RL policy)."""
        cmd = self.body_torque(info["sigma_BR"], info["omega_BR_B"])
        if self.Gs is not None:
            cmd = allocate_body_torque(cmd, self.Gs)
        a = cmd / self.action_scale
        peak = np.max(np.abs(a))
        return a / peak if peak > 1.0 else a
