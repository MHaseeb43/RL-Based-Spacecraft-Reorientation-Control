"""
Basilisk simulation of the satellite with reaction wheels.

The environment only uses reset() and step() of this class, so we can swap the simulation
(the tests use a simple numpy version instead of Basilisk).

Planned Basilisk modules (one SimulationBaseClass per simulation object):
    spacecraft.Spacecraft()                       the satellite body, inertia from the Perturbation
    reactionWheelStateEffector + simIncludeRW     the wheels (spin axes, Js, u_max, Omega_max)
    extForceTorque.ExtForceTorque()               disturbance torque
    simpleNav.SimpleNav()                         attitude / rate measurement (noise optional)
    inertial3D.inertial3D()                       target attitude sigma_RN
    attTrackingError.attTrackingError()           sigma_BR and omega_BR_B
    rwMotorTorque.rwMotorTorque()                 splits the body torque across the wheels
    messaging.CmdTorqueBodyMsg                    body torque from the agent -> rwMotorTorque
    or messaging.ArrayMotorTorqueMsg              wheel torques directly (control="wheel_torque")

Every step() writes the command message and runs the simulation for dt seconds
(ConfigureStopTime + ExecuteSimulation, integration step integration_dt).

What step(cmd) gets (tests/test_env_pd.py checks this with the numpy version):
    control="body_torque":  cmd = 3 values, the torque we want on the satellite body [Nm].
        rwMotorTorque wants the "required" torque Lr and the wheels then push on the hub with -Lr,
        so we have to write Lr = -cmd into the CmdTorqueBodyMsg.
        Please check the sign with a quick test: cmd = [+L, 0, 0] must make omega_BN_B[0] bigger.
        Use controlAxes_B = identity (control all 3 axes) and the NOMINAL wheel axes in the
        rwMotorTorque config message. allocation.py does the same thing in numpy.
    control="wheel_torque": cmd = one motor torque per wheel [Nm].
    In both cases every wheel torque is limited to +-u_max (the RW effector does that), and
    SimState.wheel_torques returns the torques that were really applied.
"""

from dataclasses import dataclass

import numpy as np

CONTROL_MODES = ("body_torque", "wheel_torque")


@dataclass
class SimState:
    time: float                # [s]
    sigma_BN: np.ndarray       # (3,) true attitude (MRP)
    omega_BN_B: np.ndarray     # (3,) true angular velocity [rad/s]
    sigma_BR: np.ndarray       # (3,) attitude error relative to the target (MRP)
    omega_BR_B: np.ndarray     # (3,) angular velocity error [rad/s]
    wheel_speeds: np.ndarray   # (N,) wheel speeds [rad/s]
    wheel_torques: np.ndarray  # (N,) applied wheel motor torques [Nm]


class BasiliskReorientationSim:
    """Satellite with reaction wheels, simulated in Basilisk"""

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
        self.omega_max = float(rw["omega_max_rpm"]) * 2.0 * np.pi / 60.0  # rpm -> rad/s
        self._sim = None

    def reset(self, sigma_BN, omega_BN_B, sigma_RN, perturbation, wheel_speeds=None):
        """
        Build (or reset) the Basilisk simulation for a new episode.

        :param sigma_BN:  start attitude (MRP)
        :param omega_BN_B:  start angular velocity [rad/s]
        :param sigma_RN:  target attitude (MRP)
        :param perturbation:  Perturbation of this episode (true inertia, wheel axes, disturbances)
        :param wheel_speeds:  start wheel speeds [rad/s], value from the config if None
        :return: SimState
        """
        # TODO: create the Basilisk process/task and the modules listed at the top of this file
        raise NotImplementedError("Basilisk simulation is not implemented yet (WP 3100)")

    def step(self, cmd):
        """
        Apply the command for one control step dt (see the top of this file for what cmd is).

        :param cmd:  body torque [Nm] (3 values) if control == "body_torque", otherwise one wheel torque per wheel [Nm]
        :return: SimState
        """
        # TODO: write the CmdTorqueBodyMsg (Lr = -cmd) or the ArrayMotorTorqueMsg, add the disturbance torque
        #   from perturbation.external_torque(t) (constant + periodic, body frame), then ExecuteSimulation
        raise NotImplementedError("Basilisk simulation is not implemented yet (WP 3100)")
