"""
Gymnasium environment wrapper around the Basilisk simulation (WP 3200).

State space  (WP 2100): [q_BR (4), omega_BR_B / rate_scale (3), Omega / Omega_max (N), a_prev (N)]
                        q_BR is the error quaternion (scalar first) with q0 >= 0, i.e. the short rotation.
                        Basilisk works with MRPs; sigma_BR is converted here (same as RigidBodyKinematics.MRP2EP).
                        Optionally the last `stack_frames` observations are stacked.
Action space (WP 2200): normalized RW motor torques a in [-1, 1]^N, u = a * u_max      (action.type: wheel_torque)
                        or normalized body torque a in [-1, 1]^3, u = -pinv(Gs) a L_max (action.type: body_torque)
Reward       (WP 2300): see rl_reorientation/envs/reward.py
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_reorientation.attitude import (mrp_add, mrp_shadow, mrp_to_quat, principal_angle, random_mrp,
                                       random_mrp_with_angle, random_unit_vector)
from rl_reorientation.envs.reward import compute_reward
from rl_reorientation.sim import BasiliskReorientationSim, Perturbation, sample_perturbation

# Phase used when no curriculum phase is given (large-angle rest-to-rest)
DEFAULT_PHASE = {
    "name": "default",
    "initial_angle_deg": [0.0, 180.0],
    "initial_rate_deg_s": [0.0, 0.0],
    "max_steps": 3000,
    "uncertainty_scale": 0.0,
}

OBS_BOUND = 2.0
ATTITUDE_DIMS = {"quaternion": 4, "mrp": 3}


class ReorientationEnv(gym.Env):
    """Spacecraft reorientation with reaction wheels."""

    metadata = {"render_modes": []}

    def __init__(self, config, phase=None, sim=None):
        """
        Args:
            config: dict from rl_reorientation.config.load_project_config()
            phase: curriculum phase dict (see configs/curriculum/*.yaml), defaults to DEFAULT_PHASE
            sim: optional simulator instance with the BasiliskReorientationSim interface
        """
        super().__init__()
        self.sc_cfg = config["spacecraft"]
        self.env_cfg = config["env"]
        self.unc_cfg = config["uncertainty"]
        sim_cfg = self.env_cfg["simulation"]
        self.sim = sim or BasiliskReorientationSim(self.sc_cfg, sim_cfg["dt"], sim_cfg["integration_dt"])
        self.n_wheels = self.sim.n_wheels
        self.set_phase(phase or DEFAULT_PHASE)

        obs_cfg = self.env_cfg["observation"]
        if obs_cfg["attitude"] not in ATTITUDE_DIMS:
            raise ValueError(f"observation.attitude must be one of {list(ATTITUDE_DIMS)}")
        act_cfg = self.env_cfg["action"]
        self.action_type = act_cfg["type"]
        if self.action_type == "wheel_torque":
            self.action_dim = self.n_wheels
        elif self.action_type == "body_torque":
            # The controller only knows the nominal wheel geometry (true axes may be misaligned)
            Gs = np.array(self.sc_cfg["reaction_wheels"]["spin_axes_B"], dtype=float).T
            Gs /= np.linalg.norm(Gs, axis=0, keepdims=True)
            self._Gs_pinv = np.linalg.pinv(Gs)
            self._L_max = float(act_cfg.get("body_torque_max") or self.sim.u_max)
            self.action_dim = 3
        else:
            raise ValueError("action.type must be 'wheel_torque' or 'body_torque'")

        self.stack_frames = int(obs_cfg.get("stack_frames", 1))
        frame_dim = ATTITUDE_DIMS[obs_cfg["attitude"]] + 3 * obs_cfg["include_rate"] \
            + self.n_wheels * obs_cfg["include_wheel_speeds"] + self.action_dim * obs_cfg["include_prev_action"]
        # All entries are normalized to ~[-1, 1] and clipped to the bounds
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(frame_dim * self.stack_frames,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

        self.state = None
        self.perturbation = None
        self.prev_action = np.zeros(self.action_dim)
        self.prev_angle_deg = 0.0
        self.steps = 0
        self._frames = deque(maxlen=self.stack_frames)

    def set_phase(self, phase):
        """Switch the curriculum phase (call via VecEnv.env_method('set_phase', phase))."""
        self.phase = {**DEFAULT_PHASE, **phase}
        self.max_steps = int(self.phase["max_steps"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Random target attitude and initial attitude error within the phase range
        sigma_RN = random_mrp(rng)
        angle = np.deg2rad(rng.uniform(*self.phase["initial_angle_deg"]))
        sigma_BN = mrp_add(random_mrp_with_angle(angle, rng), sigma_RN)
        rate = np.deg2rad(rng.uniform(*self.phase["initial_rate_deg_s"]))
        omega_BN_B = rate * random_unit_vector(rng)

        scale = float(self.phase["uncertainty_scale"])
        self.perturbation = sample_perturbation(self.sc_cfg, self.unc_cfg, scale, rng) if scale > 0.0 \
            else Perturbation.nominal(self.sc_cfg)

        self.state = self.sim.reset(sigma_BN, omega_BN_B, sigma_RN, self.perturbation)
        self.prev_action = np.zeros(self.action_dim)
        self.prev_angle_deg = np.rad2deg(principal_angle(self.state.sigma_BR))
        self.steps = 0

        frame = self._frame()
        self._frames.clear()
        for _ in range(self.stack_frames):
            self._frames.append(frame)
        return self._observe(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.state = self.sim.step(self.wheel_torques(action))
        self.steps += 1

        angle_deg = np.rad2deg(principal_angle(self.state.sigma_BR))
        reward, terms, terminated = compute_reward(
            angle_deg, self.prev_angle_deg, action, self.state.omega_BR_B, self.state.wheel_speeds,
            self.sim.omega_max, self.env_cfg["reward"], self.env_cfg.get("termination"))
        self.prev_angle_deg = angle_deg
        self.prev_action = action

        truncated = (not terminated) and self.steps >= self.max_steps
        self._frames.append(self._frame())
        info = self._info()
        info["reward_terms"] = terms
        return self._observe(), reward, terminated, truncated, info

    def wheel_torques(self, action):
        """Normalized action -> RW motor torque command [N m] (WP 2200)."""
        if self.action_type == "wheel_torque":
            return action * self.sim.u_max
        # Body torque L; the wheels act on the hub with -Gs u, so u = -pinv(Gs) L (minimum norm).
        # If a wheel would saturate, all torques are scaled down together (keeps the torque direction).
        u = -self._Gs_pinv @ (action * self._L_max)
        peak = np.max(np.abs(u))
        return u * (self.sim.u_max / peak) if peak > self.sim.u_max else u

    def _frame(self):
        """One normalized (and optionally noisy) observation frame."""
        obs_cfg = self.env_cfg["observation"]
        p = self.perturbation
        sigma = self.state.sigma_BR
        omega = self.state.omega_BR_B
        if p.attitude_noise_std > 0.0:
            sigma = mrp_add(self.np_random.normal(0.0, p.attitude_noise_std / 4.0, 3), sigma)
        if p.rate_noise_std > 0.0:
            omega = omega + self.np_random.normal(0.0, p.rate_noise_std, 3)

        # Short-rotation MRP -> quaternion with q0 >= 0 (removes the q / -q ambiguity)
        parts = [mrp_to_quat(mrp_shadow(sigma)) if obs_cfg["attitude"] == "quaternion" else mrp_shadow(sigma)]
        if obs_cfg["include_rate"]:
            parts.append(omega / obs_cfg["rate_scale"])
        if obs_cfg["include_wheel_speeds"]:
            parts.append(self.state.wheel_speeds / self.sim.omega_max)
        if obs_cfg["include_prev_action"]:
            parts.append(self.prev_action)
        return np.clip(np.concatenate(parts), -OBS_BOUND, OBS_BOUND)

    def _observe(self):
        """Stacked observation, oldest frame first."""
        return np.concatenate(self._frames).astype(np.float32)

    def _info(self):
        """Physical state for evaluation, logging and the PD baseline."""
        s = self.state
        return {
            "time": s.time,
            "sigma_BR": s.sigma_BR.copy(),
            "q_BR": mrp_to_quat(mrp_shadow(s.sigma_BR)),
            "omega_BR_B": s.omega_BR_B.copy(),
            "wheel_speeds": s.wheel_speeds.copy(),
            "wheel_torques": s.wheel_torques.copy(),
            "angle_deg": float(np.rad2deg(principal_angle(s.sigma_BR))),
        }
