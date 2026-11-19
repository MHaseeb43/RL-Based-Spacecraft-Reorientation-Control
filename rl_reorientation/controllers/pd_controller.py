"""
PD controller, the classical baseline we compare the RL agent against.

Control law (same as Basilisk's mrpFeedback, just without the integral part):
    L = -K * sigma_BR - P * omega_BR

It gets the same input and gives the same output as the RL agent, so the comparison is fair.
"""

import numpy as np

from rl_reorientation.sim.allocation import allocate_body_torque, rw_axes_matrix


class PDController:
    def __init__(self, K, P, action_scale, spin_axes_B=None):
        """
        :param K:  proportional gain [Nm]
        :param P:  derivative gain [Nms]
        :param action_scale:  torque that an action of 1.0 stands for [Nm]
        :param spin_axes_B:  wheel spin axes, only needed if we command the wheels directly (wheel_torque mode).
                             None -> we output a body torque like the agent
        """
        self.K = float(K)
        self.P = float(P)
        self.action_scale = float(action_scale)
        if spin_axes_B is None:
            self.Gs = None
        else:
            self.Gs = rw_axes_matrix(spin_axes_B)

    @classmethod
    def for_env(cls, env, K, P):
        """Create a PD controller that fits the action type of the environment"""
        axes = None
        if env.action_type == "wheel_torque":
            axes = env.sc_cfg["reaction_wheels"]["spin_axes_B"]
        return cls(K, P, env.action_scale, axes)

    def body_torque(self, sigma_BR, omega_BR_B):
        """Desired torque on the satellite body [Nm]"""
        return -self.K * np.asarray(sigma_BR) - self.P * np.asarray(omega_BR_B)

    def __call__(self, info):
        """Returns the normalized action in [-1, 1], same as the RL agent.
        Reads the true state from the info dict of the environment"""
        torque = self.body_torque(info["sigma_BR"], info["omega_BR_B"])
        if self.Gs is not None:
            torque = allocate_body_torque(torque, self.Gs)  # split it across the wheels ourselves
        action = torque / self.action_scale

        # if the command is too big, scale all axes down by the same factor so the direction stays the same
        biggest = np.max(np.abs(action))
        if biggest > 1.0:
            action = action / biggest
        return action
