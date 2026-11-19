"""
Gymnasium environment for the reorientation task. It connects the RL agent with the simulation.

What the agent sees (state), one frame:
    error quaternion q_BR (4)          how far we are from the target, scalar first with q0 >= 0
    angular velocity / rate_scale (3)
    wheel speeds / max speed (N)
    previous action (3)
The simulation works with MRPs, so sigma_BR is converted to a quaternion here.
Optionally the last few frames are stacked (stack_frames in env.yaml).

What the agent does (action):
    3 numbers in [-1, 1] -> body torque L = action * L_max. The simulation splits it across the
    wheels with Basilisk's rwMotorTorque. Other option "wheel_torque": one value per wheel.

The reward is in reward.py
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_reorientation.attitude import (mrp_add, mrp_shadow, mrp_to_quat, principal_angle, random_mrp,
                                       random_mrp_with_angle, random_unit_vector)
from rl_reorientation.envs.reward import compute_reward
from rl_reorientation.sim import BasiliskReorientationSim, Perturbation, sample_perturbation

# used if no curriculum phase is given: big rest-to-rest slews without uncertainty
DEFAULT_PHASE = {
    "name": "default",
    "initial_angle_deg": [0.0, 180.0],
    "initial_rate_deg_s": [0.0, 0.0],
    "max_steps": 3000,
    "uncertainty_scale": 0.0,
}

OBS_BOUND = 2.0     # observations are clipped to [-2, 2]
ATTITUDE_DIMS = {"quaternion": 4, "mrp": 3}


class ReorientationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config, phase=None, sim=None):
        """
        :param config:  dict from load_project_config()
        :param phase:  curriculum phase (see configs/curriculum/), DEFAULT_PHASE if None
        :param sim:  simulation object. By default the Basilisk simulation, the tests pass a simple numpy one
        """
        super().__init__()
        self.sc_cfg = config["spacecraft"]
        self.env_cfg = config["env"]
        self.unc_cfg = config["uncertainty"]
        sim_cfg = self.env_cfg["simulation"]
        obs_cfg = self.env_cfg["observation"]
        self.action_type = self.env_cfg["action"]["type"]

        # -------------------------
        # 1 - Simulation
        # -------------------------
        if sim is None:
            sim = BasiliskReorientationSim(self.sc_cfg, sim_cfg["dt"], sim_cfg["integration_dt"],
                                           control=self.action_type)
        self.sim = sim
        self.sim.control = self.action_type  # tells the simulation if step() gets a body torque or wheel torques
        self.n_wheels = self.sim.n_wheels
        self.set_phase(phase or DEFAULT_PHASE)

        # -------------------------
        # 2 - Action space
        # -------------------------
        # the agent outputs values in [-1, 1], we multiply them by action_scale to get the torque in Nm
        if self.action_type == "body_torque":
            self.action_dim = 3
            self.action_scale = float(self.env_cfg["action"].get("body_torque_max") or self.sim.u_max)  # null -> u_max
        elif self.action_type == "wheel_torque":
            self.action_dim = self.n_wheels
            self.action_scale = self.sim.u_max
        else:
            raise ValueError("action.type must be 'body_torque' or 'wheel_torque'")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

        # -------------------------
        # 3 - Observation space
        # -------------------------
        if obs_cfg["attitude"] not in ATTITUDE_DIMS:
            raise ValueError(f"observation.attitude must be one of {list(ATTITUDE_DIMS)}")
        frame_dim = ATTITUDE_DIMS[obs_cfg["attitude"]]
        if obs_cfg["include_rate"]:
            frame_dim += 3
        if obs_cfg["include_wheel_speeds"]:
            frame_dim += self.n_wheels
        if obs_cfg["include_prev_action"]:
            frame_dim += self.action_dim

        self.stack_frames = int(obs_cfg.get("stack_frames", 1))
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(frame_dim * self.stack_frames,),
                                            dtype=np.float32)

        # these are set in reset()
        self.state = None
        self.perturbation = None
        self.prev_action = np.zeros(self.action_dim)
        self.prev_angle_deg = 0.0
        self.steps = 0
        self._frames = deque(maxlen=self.stack_frames)  # the last few frames, oldest first

    def set_phase(self, phase):
        """Switch to another curriculum phase. During training: env.env_method("set_phase", phase)"""
        self.phase = dict(DEFAULT_PHASE)
        self.phase.update(phase)  # missing keys keep the default value
        self.max_steps = int(self.phase["max_steps"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # random target, and a start attitude that is a random angle (range from the phase) away from it
        sigma_RN = random_mrp(rng)
        angle = np.deg2rad(rng.uniform(*self.phase["initial_angle_deg"]))
        sigma_BN = mrp_add(random_mrp_with_angle(angle, rng), sigma_RN)
        rate = np.deg2rad(rng.uniform(*self.phase["initial_rate_deg_s"]))
        omega_BN_B = rate * random_unit_vector(rng)

        # new inertia and disturbance for every episode (nominal model if the phase has no uncertainty)
        scale = float(self.phase["uncertainty_scale"])
        if scale > 0.0:
            self.perturbation = sample_perturbation(self.sc_cfg, self.unc_cfg, scale, rng)
        else:
            self.perturbation = Perturbation.nominal(self.sc_cfg)

        self.state = self.sim.reset(sigma_BN, omega_BN_B, sigma_RN, self.perturbation)
        self.prev_action = np.zeros(self.action_dim)
        self.prev_angle_deg = np.rad2deg(principal_angle(self.state.sigma_BR))
        self.steps = 0

        # at the start there is no history yet, so fill the stack with copies of the first frame
        frame = self._frame()
        self._frames.clear()
        for _ in range(self.stack_frames):
            self._frames.append(frame)
        return self._observe(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.state = self.sim.step(action * self.action_scale)
        self.steps += 1

        angle_deg = np.rad2deg(principal_angle(self.state.sigma_BR))
        reward, terms, terminated = compute_reward(
            angle_deg, self.prev_angle_deg, action, self.state.omega_BR_B, self.state.wheel_speeds,
            self.sim.omega_max, self.env_cfg["reward"], self.env_cfg.get("termination"))
        self.prev_angle_deg = angle_deg
        self.prev_action = action

        # time limit reached -> truncated (not terminated, the satellite didn't fail)
        truncated = (not terminated) and self.steps >= self.max_steps

        self._frames.append(self._frame())
        info = self._info()
        info["reward_terms"] = terms
        return self._observe(), reward, terminated, truncated, info

    def _frame(self):
        """One observation frame, normalized (and with sensor noise if it's turned on)"""
        obs_cfg = self.env_cfg["observation"]
        sigma = self.state.sigma_BR
        omega = self.state.omega_BR_B

        # sensor noise (off by default)
        if self.perturbation.attitude_noise_std > 0.0:
            # for small angles |sigma| is about angle / 4
            noise = self.np_random.normal(0.0, self.perturbation.attitude_noise_std / 4.0, 3)
            sigma = mrp_add(noise, sigma)
        if self.perturbation.rate_noise_std > 0.0:
            omega = omega + self.np_random.normal(0.0, self.perturbation.rate_noise_std, 3)

        sigma = mrp_shadow(sigma)
        if obs_cfg["attitude"] == "quaternion":
            parts = [mrp_to_quat(sigma)]  # short rotation MRP -> quaternion with q0 >= 0
        else:
            parts = [sigma]
        if obs_cfg["include_rate"]:
            parts.append(omega / obs_cfg["rate_scale"])
        if obs_cfg["include_wheel_speeds"]:
            parts.append(self.state.wheel_speeds / self.sim.omega_max)
        if obs_cfg["include_prev_action"]:
            parts.append(self.prev_action)
        return np.clip(np.concatenate(parts), -OBS_BOUND, OBS_BOUND)

    def _observe(self):
        """All stacked frames in one array, oldest frame first"""
        return np.concatenate(self._frames).astype(np.float32)

    def _info(self):
        """Physical state, used for evaluation, logging and by the PD controller"""
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
