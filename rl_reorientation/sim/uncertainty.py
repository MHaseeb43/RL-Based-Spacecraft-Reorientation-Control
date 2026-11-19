"""
Uncertainties and disturbances.

At the start of every episode we draw a new Perturbation (random inertia, disturbance torque, ...).
The simulation uses the physical parts (inertia, wheel axes, wheel torque errors and external_torque(t)),
the environment adds the sensor noise to the observation.
"""

from dataclasses import dataclass

import numpy as np

from rl_reorientation.attitude import random_unit_vector

MAX_INERTIA_RESAMPLES = 100


@dataclass
class Perturbation:
    inertia: np.ndarray                # (3, 3) true inertia of the hub [kg m^2]
    spin_axes_B: np.ndarray            # (N, 3) true wheel spin axes
    torque_scale: np.ndarray           # (N,) scale factor of each wheel torque
    torque_noise_std: float            # [Nm] wheel torque noise per step
    ext_torque_const_B: np.ndarray     # (3,) constant disturbance torque [Nm]
    ext_torque_periodic_B: np.ndarray  # (3,) amplitude of the periodic disturbance torque [Nm]
    ext_torque_period: float           # [s] period of the periodic disturbance
    ext_torque_phase: float            # [rad] phase of the periodic disturbance
    ext_torque_std: float              # [Nm] random (white noise) disturbance torque per step
    attitude_noise_std: float          # [rad] attitude sensor noise
    rate_noise_std: float              # [rad/s] gyro noise

    @classmethod
    def nominal(cls, sc_cfg):
        """No uncertainty at all, the satellite is exactly like in the config"""
        return sample_perturbation(sc_cfg, {}, scale=0.0, rng=np.random.default_rng(0))

    def external_torque(self, t, rng=None):
        """Disturbance torque in the body frame [Nm] at time t [s]. The simulation calls this every step.
        constant + periodic part, plus white noise if an rng is given"""
        tau = self.ext_torque_const_B \
            + self.ext_torque_periodic_B * np.sin(2.0 * np.pi * t / self.ext_torque_period + self.ext_torque_phase)
        if rng is not None and self.ext_torque_std > 0.0:
            tau = tau + rng.normal(0.0, self.ext_torque_std, 3)
        return tau


def is_valid_inertia(J, tol=1e-12):
    """Checks if an inertia matrix is physically possible:
    symmetric, positive definite, and the principal moments fulfill the triangle inequality"""
    if not np.allclose(J, J.T):
        return False
    eig = np.linalg.eigvalsh(J)  # sorted, smallest first
    if eig[0] <= 0.0:
        return False
    return eig[0] + eig[1] >= eig[2] - tol


def _misalign(axis, max_angle, rng):
    """Tilt a unit vector by a random angle (up to max_angle) in a random direction"""
    perp = np.cross(axis, random_unit_vector(rng))
    perp /= np.linalg.norm(perp)
    angle = rng.uniform(0.0, max_angle)
    return np.cos(angle) * axis + np.sin(angle) * perp


def _sample_inertia(J_nominal, b, scale, rng):
    """Principal moments J_ii * (1 + random value in +-relative_max), plus optional products of inertia.
    We draw again until the result is physically valid"""
    rel_max = scale * b["relative_max"]
    poi_std = scale * b.get("products_of_inertia_std", 0.0)
    for _ in range(MAX_INERTIA_RESAMPLES):
        J = np.diag(np.diag(J_nominal) * (1.0 + rng.uniform(-rel_max, rel_max, 3)))
        off = poi_std * rng.standard_normal(3)
        J[0, 1] = J[1, 0] = J_nominal[0, 1] + off[0]
        J[0, 2] = J[2, 0] = J_nominal[0, 2] + off[1]
        J[1, 2] = J[2, 1] = J_nominal[1, 2] + off[2]
        if is_valid_inertia(J):
            return J
    raise ValueError(f"No physically valid inertia found in {MAX_INERTIA_RESAMPLES} tries, "
                     "make inertia.relative_max or products_of_inertia_std smaller")


def _get_block(unc_cfg, name, scale):
    """Returns one block of uncertainty.yaml, or None if it is disabled (or scale is 0)"""
    b = unc_cfg.get(name, {})
    if b.get("enabled", False) and scale > 0.0:
        return b
    return None


def sample_perturbation(sc_cfg, unc_cfg, scale, rng):
    """
    Draw one random set of uncertainties and disturbances.

    :param sc_cfg:  spacecraft config (spacecraft.yaml)
    :param unc_cfg:  uncertainty config (uncertainty.yaml)
    :param scale:  multiplies all uncertainty sizes (0 = nominal model, 1 = full uncertainty)
    :param rng:  numpy random generator
    """
    inertia = np.array(sc_cfg["hub"]["inertia"], dtype=float)
    axes = np.array(sc_cfg["reaction_wheels"]["spin_axes_B"], dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    n = axes.shape[0]

    # moment of inertia
    b = _get_block(unc_cfg, "inertia", scale)
    if b is not None:
        inertia = _sample_inertia(inertia, b, scale, rng)

    # wheel misalignment
    b = _get_block(unc_cfg, "rw_misalignment", scale)
    if b is not None:
        max_angle = np.deg2rad(scale * b["max_angle_deg"])
        axes = np.array([_misalign(a, max_angle, rng) for a in axes])

    # wheel torque errors
    torque_scale = np.ones(n)
    torque_noise_std = 0.0
    b = _get_block(unc_cfg, "rw_torque", scale)
    if b is not None:
        torque_scale = 1.0 + scale * b["scale_factor_std"] * rng.standard_normal(n)
        torque_noise_std = scale * b["noise_std"]

    # external disturbance torque: random size and direction for the constant and the periodic part
    ext_const = np.zeros(3)
    ext_periodic = np.zeros(3)
    period = 1.0
    phase = 0.0
    ext_std = 0.0
    b = _get_block(unc_cfg, "external_torque", scale)
    if b is not None:
        ext_const = rng.uniform(0.0, scale * b.get("constant_max", 0.0)) * random_unit_vector(rng)
        ext_periodic = rng.uniform(0.0, scale * b.get("periodic_max", 0.0)) * random_unit_vector(rng)
        period = rng.uniform(*b.get("period_range_s", [1.0, 1.0]))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        ext_std = scale * b.get("random_std", 0.0)

    # sensor noise
    att_std = 0.0
    rate_std = 0.0
    b = _get_block(unc_cfg, "sensor_noise", scale)
    if b is not None:
        att_std = np.deg2rad(scale * b["attitude_std_deg"])
        rate_std = scale * b["rate_std"]

    return Perturbation(
        inertia=inertia,
        spin_axes_B=axes,
        torque_scale=torque_scale,
        torque_noise_std=torque_noise_std,
        ext_torque_const_B=ext_const,
        ext_torque_periodic_B=ext_periodic,
        ext_torque_period=period,
        ext_torque_phase=phase,
        ext_torque_std=ext_std,
        attitude_noise_std=att_std,
        rate_noise_std=rate_std,
    )
