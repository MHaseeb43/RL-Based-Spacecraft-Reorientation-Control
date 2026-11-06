"""
Uncertainty & disturbance modelling (WP 3300).

A `Perturbation` is sampled once per episode (domain randomization) and handed to the
simulator, which applies the physical parts (inertia, misalignment, wheel torque errors and
the external torque from `external_torque(t)`). Sensor noise is applied by the environment
to the observation.
"""

from dataclasses import dataclass

import numpy as np

from rl_reorientation.attitude import random_unit_vector

MAX_INERTIA_RESAMPLES = 100


@dataclass
class Perturbation:
    inertia: np.ndarray                # (3, 3) true hub inertia [kg m^2]
    spin_axes_B: np.ndarray            # (N, 3) true RW spin axes
    torque_scale: np.ndarray           # (N,) RW torque scale factors
    torque_noise_std: float            # [N m] additive RW torque noise per step
    ext_torque_const_B: np.ndarray     # (3,) constant external disturbance torque [N m]
    ext_torque_periodic_B: np.ndarray  # (3,) amplitude vector of the periodic disturbance torque [N m]
    ext_torque_period: float           # [s] period of the periodic disturbance
    ext_torque_phase: float            # [rad] phase of the periodic disturbance
    ext_torque_std: float              # [N m] white-noise external torque per step
    attitude_noise_std: float          # [rad] attitude knowledge error
    rate_noise_std: float              # [rad/s] gyro noise

    @classmethod
    def nominal(cls, sc_cfg):
        """Perturbation-free model."""
        return sample_perturbation(sc_cfg, {}, scale=0.0, rng=np.random.default_rng(0))

    def external_torque(self, t, rng=None):
        """
        External disturbance torque in body frame [N m] at simulation time t [s]:
        constant + periodic (+ white noise if `rng` is given). Called by the simulator every step.
        """
        tau = self.ext_torque_const_B \
            + self.ext_torque_periodic_B * np.sin(2.0 * np.pi * t / self.ext_torque_period + self.ext_torque_phase)
        if rng is not None and self.ext_torque_std > 0.0:
            tau = tau + rng.normal(0.0, self.ext_torque_std, 3)
        return tau


def is_valid_inertia(J, tol=1e-12):
    """Symmetric, positive definite and principal moments satisfy the triangle inequality."""
    if not np.allclose(J, J.T):
        return False
    eig = np.linalg.eigvalsh(J)
    if eig[0] <= 0.0:
        return False
    return eig[0] + eig[1] >= eig[2] - tol


def _misalign(axis, max_angle, rng):
    """Tilt a unit vector by a random angle in [0, max_angle] in a random direction."""
    perp = np.cross(axis, random_unit_vector(rng))
    perp /= np.linalg.norm(perp)
    angle = rng.uniform(0.0, max_angle)
    return np.cos(angle) * axis + np.sin(angle) * perp


def _sample_inertia(J_nominal, b, scale, rng):
    """Principal MOI J_ii * (1 + U(-max, max) * scale) plus optional products of inertia, redrawn until valid."""
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
    raise ValueError(f"No physically valid inertia found in {MAX_INERTIA_RESAMPLES} samples; "
                     "reduce inertia.relative_max / products_of_inertia_std")


def sample_perturbation(sc_cfg, unc_cfg, scale, rng):
    """
    Sample one set of model uncertainties and disturbances.

    Args:
        sc_cfg: spacecraft config (configs/spacecraft.yaml)
        unc_cfg: uncertainty config (configs/uncertainty.yaml)
        scale: multiplier for all uncertainty magnitudes (0 = nominal model)
        rng: numpy Generator
    """
    def block(name):
        b = unc_cfg.get(name, {})
        return b if (b.get("enabled", False) and scale > 0.0) else None

    inertia = np.array(sc_cfg["hub"]["inertia"], dtype=float)
    axes = np.array(sc_cfg["reaction_wheels"]["spin_axes_B"], dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    n = axes.shape[0]

    if (b := block("inertia")) is not None:
        inertia = _sample_inertia(inertia, b, scale, rng)

    if (b := block("rw_misalignment")) is not None:
        max_angle = np.deg2rad(scale * b["max_angle_deg"])
        axes = np.array([_misalign(a, max_angle, rng) for a in axes])

    torque_scale = np.ones(n)
    torque_noise_std = 0.0
    if (b := block("rw_torque")) is not None:
        torque_scale = 1.0 + scale * b["scale_factor_std"] * rng.standard_normal(n)
        torque_noise_std = scale * b["noise_std"]

    ext_const = np.zeros(3)
    ext_periodic = np.zeros(3)
    period, phase = 1.0, 0.0
    ext_std = 0.0
    if (b := block("external_torque")) is not None:
        ext_const = rng.uniform(0.0, scale * b.get("constant_max", 0.0)) * random_unit_vector(rng)
        ext_periodic = rng.uniform(0.0, scale * b.get("periodic_max", 0.0)) * random_unit_vector(rng)
        period = rng.uniform(*b.get("period_range_s", [1.0, 1.0]))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        ext_std = scale * b.get("random_std", 0.0)

    att_std = 0.0
    rate_std = 0.0
    if (b := block("sensor_noise")) is not None:
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
