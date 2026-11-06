"""
Basilisk spacecraft & reaction wheel simulation (WP 3100).

The environment only talks to this class through `reset()` and `step()`, so the dynamics
backend stays exchangeable (e.g. for a fast pure-numpy model for debugging).

Planned Basilisk setup (one SimulationBaseClass per simulator instance):
    - spacecraft.Spacecraft()                       rigid hub, inertia from Perturbation
    - reactionWheelStateEffector + simIncludeRW     N wheels (spin axes, Js, u_max, Omega_max)
    - extForceTorque.ExtForceTorque()               external disturbance torque
    - simpleNav.SimpleNav()                         attitude / rate measurement (optional noise)
    - inertial3D.inertial3D()                       reference attitude sigma_RN
    - attTrackingError.attTrackingError()           sigma_BR, omega_BR_B
    - messaging.ArrayMotorTorqueMsg                 RW motor torque command written by the agent

Each `step()` writes the commanded wheel torques to the motor torque message and advances
the simulation by `dt` with `ExecuteSimulation()` (`ConfigureStopTime` + integration_dt).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimState:
    time: float                # [s]
    sigma_BN: np.ndarray       # (3,) true inertial attitude (MRP)
    omega_BN_B: np.ndarray     # (3,) true body rate [rad/s]
    sigma_BR: np.ndarray       # (3,) attitude error w.r.t. reference (MRP)
    omega_BR_B: np.ndarray     # (3,) rate error [rad/s]
    wheel_speeds: np.ndarray   # (N,) RW speeds [rad/s]
    wheel_torques: np.ndarray  # (N,) applied RW motor torques [N m]


class BasiliskReorientationSim:
    """Spacecraft with reaction wheels simulated in Basilisk."""

    def __init__(self, sc_cfg, dt, integration_dt):
        self.sc_cfg = sc_cfg
        self.dt = dt
        self.integration_dt = integration_dt
        rw = sc_cfg["reaction_wheels"]
        self.n_wheels = len(rw["spin_axes_B"])
        self.u_max = float(rw["u_max"])
        self.omega_max = float(rw["omega_max_rpm"]) * 2.0 * np.pi / 60.0
        self._sim = None

    def reset(self, sigma_BN, omega_BN_B, sigma_RN, perturbation, wheel_speeds=None):
        """
        Build (or re-initialize) the Basilisk simulation for a new episode.

        Args:
            sigma_BN: (3,) initial attitude (MRP)
            omega_BN_B: (3,) initial body rate [rad/s]
            sigma_RN: (3,) target attitude (MRP)
            perturbation: Perturbation (true inertia, RW axes, disturbances)
            wheel_speeds: (N,) initial RW speeds [rad/s], defaults to config value
        Returns:
            SimState
        """
        # TODO(WP 3100): create the Basilisk task/process and the modules listed in the module docstring
        raise NotImplementedError("Basilisk simulation is implemented in WP 3100")

    def step(self, wheel_torque_cmd):
        """
        Apply RW motor torques [N m] (shape (N,), clipped to +-u_max) for one control step dt.

        Returns:
            SimState
        """
        # TODO(WP 3100): write ArrayMotorTorqueMsg, add disturbance torque, ExecuteSimulation
        #   disturbance torque (constant + periodic) in body frame: perturbation.external_torque(t)
        raise NotImplementedError("Basilisk simulation is implemented in WP 3100")
