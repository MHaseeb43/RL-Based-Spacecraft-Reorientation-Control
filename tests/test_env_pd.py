"""
Environment + PD baseline tests with a simple numpy rigid-body model standing in for Basilisk.
They check the env interface, the observation layout and the PD sign convention.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_reorientation import attitude as att
from rl_reorientation.config import load_project_config, load_yaml
from rl_reorientation.controllers import PDController
from rl_reorientation.envs import ReorientationEnv
from rl_reorientation.evaluation.monte_carlo import monte_carlo, pd_controller, run_episode
from rl_reorientation.sim import BasiliskReorientationSim, SimState


class FakeSim(BasiliskReorientationSim):
    """Rigid body + reaction wheels, explicit Euler with substeps (no gyroscopic terms)."""

    def reset(self, sigma_BN, omega_BN_B, sigma_RN, perturbation, wheel_speeds=None):
        self.p = perturbation
        self.J = perturbation.inertia
        self.Gs = perturbation.spin_axes_B.T
        self.Js = float(self.sc_cfg["reaction_wheels"]["Js"])
        self.t, self.sigma, self.omega, self.sigma_RN = 0.0, np.array(sigma_BN), np.array(omega_BN_B), sigma_RN
        self.Omega = np.zeros(self.n_wheels) if wheel_speeds is None else np.array(wheel_speeds)
        return self._state(np.zeros(self.n_wheels))

    def step(self, wheel_torque_cmd):
        u = np.clip(wheel_torque_cmd, -self.u_max, self.u_max)
        n_sub = int(round(self.dt / self.integration_dt))
        for _ in range(n_sub):
            s = self.sigma
            B = (1 - s @ s) * np.eye(3) + 2 * att.tilde(s) + 2 * np.outer(s, s)
            self.sigma = att.mrp_shadow(s + self.integration_dt * 0.25 * B @ self.omega)
            tau = -self.Gs @ u + self.p.external_torque(self.t)
            self.omega = self.omega + self.integration_dt * np.linalg.solve(self.J, tau)
            self.Omega = self.Omega + self.integration_dt * u / self.Js
            self.t += self.integration_dt
        self.t = round(self.t, 9)
        return self._state(u)

    def _state(self, u):
        sigma_BR = att.mrp_error(self.sigma, self.sigma_RN)
        # Inertial reference -> omega_RN = 0, so omega_BR_B = omega_BN_B
        return SimState(self.t, self.sigma.copy(), self.omega.copy(), sigma_BR, self.omega.copy(),
                        self.Omega.copy(), np.array(u, dtype=float))


def make_env(phase=None):
    config = load_project_config()
    sim_cfg = config["env"]["simulation"]
    return ReorientationEnv(config, phase=phase, sim=FakeSim(config["spacecraft"], sim_cfg["dt"], sim_cfg["integration_dt"])), config


def make_pd(config):
    rw = config["spacecraft"]["reaction_wheels"]
    gains = load_yaml("pd.yaml")
    return PDController(gains["K"], gains["P"], rw["spin_axes_B"], rw["u_max"])


def test_gymnasium_api():
    env, _ = make_env({"max_steps": 20})
    check_env(env, skip_render_check=True)


def test_pd_converges_nominal():
    env, config = make_env({"initial_angle_deg": [60.0, 60.0], "initial_rate_deg_s": [0.0, 0.0], "max_steps": 3000})
    hist = run_episode(env, pd_controller(make_pd(config)), seed=1)
    assert hist["angle_deg"][0] > 59.0
    assert hist["angle_deg"][-1] < 0.25


def test_quaternion_observation():
    env, _ = make_env({"max_steps": 5})
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        q = obs[:4]
        assert obs.shape == (4 + 3 + 2 * env.n_wheels,)
        assert q[0] >= 0.0                                   # short rotation, no q / -q ambiguity
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
        assert np.rad2deg(2 * np.arccos(q[0])) == pytest.approx(info["angle_deg"], abs=1e-3)


def test_rest_to_rest_default():
    env, _ = make_env()
    _, info = env.reset(seed=3)
    assert np.allclose(info["omega_BR_B"], 0.0)


def test_frame_stacking():
    env, config = make_env({"max_steps": 10})
    config["env"]["observation"]["stack_frames"] = 3
    sim_cfg = config["env"]["simulation"]
    env = ReorientationEnv(config, sim=FakeSim(config["spacecraft"], sim_cfg["dt"], sim_cfg["integration_dt"]))
    frame = 4 + 3 + 2 * env.n_wheels
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 * frame,)
    np.testing.assert_allclose(obs[:frame], obs[-frame:])    # filled with the first frame
    obs, *_ = env.step(np.ones(env.n_wheels))
    assert not np.allclose(obs[-frame:], obs[:frame])        # newest frame last
    check_env(env, skip_render_check=True)


def test_body_torque_action():
    _, config = make_env()
    config["env"]["action"]["type"] = "body_torque"
    sim_cfg = config["env"]["simulation"]
    env = ReorientationEnv(config, sim=FakeSim(config["spacecraft"], sim_cfg["dt"], sim_cfg["integration_dt"]))
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (4 + 3 + env.n_wheels + 3,)
    check_env(env, skip_render_check=True)

    env.reset(seed=0)
    Gs = env.perturbation.spin_axes_B.T
    a = np.array([0.5, -0.2, 0.1])
    u = env.wheel_torques(a)
    np.testing.assert_allclose(-Gs @ u, a * env.sim.u_max, atol=1e-12)   # hub receives the commanded torque
    u_sat = env.wheel_torques(np.ones(3))
    assert np.max(np.abs(u_sat)) <= env.sim.u_max + 1e-15                # saturation scaling
    L = -Gs @ u_sat
    np.testing.assert_allclose(L / np.linalg.norm(L), np.ones(3) / np.sqrt(3), atol=1e-12)  # direction kept


def test_termination_on_spin_up():
    env, config = make_env({"initial_angle_deg": [90.0, 90.0], "max_steps": 100000})
    env.reset(seed=0)
    terminated, steps = False, 0
    action = np.zeros(env.n_wheels)
    action[0] = 1.0                                       # full torque on one wheel spins the spacecraft up
    while not terminated and steps < 50000:
        _, reward, terminated, truncated, info = env.step(action)
        steps += 1
    assert terminated and not truncated
    assert info["reward_terms"]["termination"] == -config["env"]["termination"]["penalty"]


def test_monte_carlo_with_uncertainty():
    env, config = make_env({"initial_angle_deg": [10.0, 90.0], "max_steps": 2000, "uncertainty_scale": 1.0})
    df = monte_carlo(env, pd_controller(make_pd(config)), n_episodes=3, settling_cfg=config["env"]["settling"])
    assert len(df) == 3
    assert (df["final_error_deg"] < df["initial_error_deg"]).all()
