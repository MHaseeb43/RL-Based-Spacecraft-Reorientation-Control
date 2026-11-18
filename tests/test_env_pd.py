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
from rl_reorientation.sim.allocation import allocate_body_torque, rw_axes_matrix


class FakeSim(BasiliskReorientationSim):
    """
    Rigid body + reaction wheels, explicit Euler with substeps (no gyroscopic terms).
    Follows the step() interface contract of BasiliskReorientationSim: body torque commands are split
    with the nominal wheel axes (as rwMotorTorque), the true (possibly misaligned) axes act on the hub.
    """

    def reset(self, sigma_BN, omega_BN_B, sigma_RN, perturbation, wheel_speeds=None):
        self.p = perturbation
        self.J = perturbation.inertia
        self.Gs = perturbation.spin_axes_B.T                                   # true axes
        self.Gs_nominal = rw_axes_matrix(self.sc_cfg["reaction_wheels"]["spin_axes_B"])
        self.Js = float(self.sc_cfg["reaction_wheels"]["Js"])
        self.t, self.sigma, self.omega, self.sigma_RN = 0.0, np.array(sigma_BN), np.array(omega_BN_B), sigma_RN
        self.Omega = np.zeros(self.n_wheels) if wheel_speeds is None else np.array(wheel_speeds)
        return self._state(np.zeros(self.n_wheels))

    def step(self, cmd):
        u_cmd = allocate_body_torque(cmd, self.Gs_nominal) if self.control == "body_torque" else cmd
        u = np.clip(u_cmd, -self.u_max, self.u_max)                            # RW effector torque limit
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


def make_env(phase=None, config=None):
    config = config or load_project_config()
    sim_cfg = config["env"]["simulation"]
    return ReorientationEnv(config, phase=phase, sim=FakeSim(config["spacecraft"], sim_cfg["dt"], sim_cfg["integration_dt"])), config


def make_pd(env):
    gains = load_yaml("pd.yaml")
    return PDController.for_env(env, gains["K"], gains["P"])


def frame_dim(env):
    return 4 + 3 + env.n_wheels + env.action_dim


def test_gymnasium_api():
    env, _ = make_env({"max_steps": 20})
    check_env(env, skip_render_check=True)


def test_default_action_is_body_torque():
    env, _ = make_env()
    assert env.action_type == "body_torque" and env.sim.control == "body_torque"
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (14,)          # 4 + 3 + 4 wheels + 3 previous action


@pytest.mark.parametrize("action_type", ["body_torque", "wheel_torque"])
def test_pd_converges_nominal(action_type):
    config = load_project_config()
    config["env"]["action"]["type"] = action_type
    env, _ = make_env({"initial_angle_deg": [60.0, 60.0], "initial_rate_deg_s": [0.0, 0.0], "max_steps": 3000}, config)
    hist = run_episode(env, pd_controller(make_pd(env)), seed=1)
    assert hist["angle_deg"][0] > 59.0
    assert hist["angle_deg"][-1] < 0.25


def test_quaternion_observation():
    env, _ = make_env({"max_steps": 5})
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        q = obs[:4]
        assert obs.shape == (frame_dim(env),)
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
    frame = frame_dim(env)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 * frame,)
    np.testing.assert_allclose(obs[:frame], obs[-frame:])    # filled with the first frame
    obs, *_ = env.step(np.ones(env.action_dim))
    assert not np.allclose(obs[-frame:], obs[:frame])        # newest frame last
    check_env(env, skip_render_check=True)


def test_body_torque_allocation():
    """Allocation (reference for rwMotorTorque): the hub receives exactly the commanded body torque."""
    env, _ = make_env()
    env.reset(seed=0)
    Gs = env.sim.Gs_nominal
    L = np.array([0.5, -0.2, 0.1]) * env.action_scale
    u = allocate_body_torque(L, Gs)
    np.testing.assert_allclose(-Gs @ u, L, atol=1e-15)
    assert np.isclose(np.linalg.norm(u), np.linalg.norm(np.linalg.lstsq(-Gs, L, rcond=None)[0]))  # minimum norm

    _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(info["wheel_torques"], allocate_body_torque([env.action_scale, 0, 0], Gs))


def test_body_torque_sign():
    """Interface contract: a positive x body torque command increases omega_x (check for the Basilisk sim too)."""
    env, _ = make_env({"initial_angle_deg": [90.0, 90.0]})
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0]))
    assert info["omega_BR_B"][0] > 0.0


def test_wheel_torque_option():
    config = load_project_config()
    config["env"]["action"]["type"] = "wheel_torque"
    env, _ = make_env({"max_steps": 20}, config)
    assert env.sim.control == "wheel_torque"
    assert env.action_space.shape == (env.n_wheels,)
    assert env.observation_space.shape == (4 + 3 + 2 * env.n_wheels,)
    check_env(env, skip_render_check=True)


def test_termination_on_spin_up():
    env, config = make_env({"initial_angle_deg": [90.0, 90.0], "max_steps": 100000})
    env.reset(seed=0)
    terminated, steps = False, 0
    action = np.array([1.0, 0.0, 0.0])                    # full torque about x spins the spacecraft up
    while not terminated and steps < 50000:
        _, reward, terminated, truncated, info = env.step(action)
        steps += 1
    assert terminated and not truncated
    assert info["reward_terms"]["termination"] == -config["env"]["termination"]["penalty"]


def test_monte_carlo_with_uncertainty():
    env, config = make_env({"initial_angle_deg": [10.0, 90.0], "max_steps": 2000, "uncertainty_scale": 1.0})
    df = monte_carlo(env, pd_controller(make_pd(env)), n_episodes=3, settling_cfg=config["env"]["settling"])
    assert len(df) == 3
    assert (df["final_error_deg"] < df["initial_error_deg"]).all()
