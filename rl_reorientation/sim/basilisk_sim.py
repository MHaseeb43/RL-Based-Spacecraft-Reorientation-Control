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
    - rwMotorTorque.rwMotorTorque()                 splits the body torque command across the wheels
                                                    (control="body_torque", supervisor decision)
    - messaging.CmdTorqueBodyMsg                    body torque command written by the agent -> rwMotorTorque
      or messaging.ArrayMotorTorqueMsg              wheel torque command (control="wheel_torque")

Each `step()` writes the command message and advances the simulation by `dt` with
`ExecuteSimulation()` (`ConfigureStopTime` + integration_dt).

Interface contract for `step(cmd)` (checked by the tests with a numpy stand-in, tests/test_env_pd.py):
    control="body_torque":  cmd (3,) = desired torque on the hub in body frame [N m].
        rwMotorTorque expects the "required" torque Lr, and the wheels then act on the hub with -Lr,
        so write Lr = -cmd into the CmdTorqueBodyMsg (please verify the sign with a one-step test:
        cmd = [+L, 0, 0] must increase omega_BN_B[0]). Use controlAxes_B = identity (3-axis control)
        and the NOMINAL wheel axes in the rwMotorTorque RW config message.
        Reference implementation: rl_reorientation/sim/allocation.py
    control="wheel_torque": cmd (N,) = RW motor torques [N m].
    In both cases each wheel torque is limited to +-u_max (RW effector), and SimState.wheel_torques
    returns the torques actually applied.
"""

from dataclasses import dataclass

import numpy as np

CONTROL_MODES = ("body_torque", "wheel_torque")


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

    def __init__(self, sc_cfg, dt, integration_dt, control="body_torque"):
        if control not in CONTROL_MODES:
            raise ValueError(f"control must be one of {CONTROL_MODES}")
        self.sc_cfg = sc_cfg
        self.dt = dt
        self.integration_dt = integration_dt
        self.control = control
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

    def step(self, cmd):
        """
        Apply the command for one control step dt (see the interface contract in the module docstring).

        Args:
            cmd: (3,) body torque [N m] if control == "body_torque", (N,) RW motor torques [N m] otherwise
        Returns:
            SimState
        """
        # TODO(WP 3100): write CmdTorqueBodyMsg (Lr = -cmd) or ArrayMotorTorqueMsg, add disturbance torque,
        #   ExecuteSimulation. Disturbance torque (constant + periodic) in body frame: perturbation.external_torque(t)
        raise NotImplementedError("Basilisk simulation is implemented in WP 3100")
