import numpy as np
import pytest

from rl_reorientation.config import load_project_config
from rl_reorientation.envs.reward import compute_reward
from rl_reorientation.sim import Perturbation, sample_perturbation
from rl_reorientation.sim.uncertainty import is_valid_inertia


@pytest.fixture
def config():
    return load_project_config()


@pytest.mark.parametrize("scale, max_rel", [(1 / 3, 0.10), (2 / 3, 0.20), (1.0, 0.30)])
def test_inertia_within_range_and_valid(config, scale, max_rel):
    rng = np.random.default_rng(0)
    J0 = np.diag(config["spacecraft"]["hub"]["inertia"])
    rel = []
    for _ in range(300):
        J = sample_perturbation(config["spacecraft"], config["uncertainty"], scale, rng).inertia
        assert is_valid_inertia(J)
        rel.append(np.diag(J) / J0 - 1.0)
    rel = np.abs(rel)
    assert rel.max() <= max_rel + 1e-9
    assert rel.max() > 0.9 * max_rel                       # range is actually used


def test_invalid_inertia_detected():
    assert not is_valid_inertia(np.diag([1.0, 1.0, 3.0]))  # violates triangle inequality
    assert not is_valid_inertia(np.diag([1.0, -1.0, 1.0]))
    assert is_valid_inertia(np.diag([1.0, 1.0, 1.5]))


def test_nominal_has_no_disturbance(config):
    p = Perturbation.nominal(config["spacecraft"])
    for t in (0.0, 12.3, 100.0):
        np.testing.assert_allclose(p.external_torque(t), 0.0)


def test_periodic_disturbance(config):
    p = sample_perturbation(config["spacecraft"], config["uncertainty"], 1.0, np.random.default_rng(1))
    lo, hi = config["uncertainty"]["external_torque"]["period_range_s"]
    assert lo <= p.ext_torque_period <= hi
    t = np.linspace(0.0, p.ext_torque_period, 200, endpoint=False)
    tau = np.array([p.external_torque(ti) for ti in t])
    np.testing.assert_allclose(tau.mean(axis=0), p.ext_torque_const_B, atol=1e-12)   # periodic part averages out
    assert np.ptp(tau, axis=0).max() > 0.0


def test_reward_goal_requires_rest(config):
    cfg = config["env"]["reward"]
    zeros = np.zeros(4)
    _, at_rest, _ = compute_reward(0.1, 0.1, zeros, np.zeros(3), zeros, 600.0, cfg)
    _, moving, _ = compute_reward(0.1, 0.1, zeros, np.array([0.05, 0, 0]), zeros, 600.0, cfg)
    assert at_rest["goal"] == cfg["goal_bonus"]
    assert moving["goal"] == -cfg["time_penalty"]


def test_progress_telescopes(config):
    """Slewing back and forth earns no net progress reward."""
    cfg = config["env"]["reward"]
    angles = [90.0, 60.0, 80.0, 40.0, 70.0, 90.0]
    total = sum(compute_reward(a, p, np.zeros(4), np.zeros(3), np.zeros(4), 600.0, cfg)[1]["progress"]
                for p, a in zip(angles[:-1], angles[1:]))
    assert total == pytest.approx(0.0)


def test_termination(config):
    cfg, term = config["env"]["reward"], config["env"]["termination"]
    fast = np.array([term["max_rate"] * 1.01, 0.0, 0.0])
    r, terms, terminated = compute_reward(10.0, 10.0, np.zeros(4), fast, np.zeros(4), 600.0, cfg, term)
    assert terminated and terms["termination"] == -term["penalty"]
    _, _, terminated = compute_reward(10.0, 10.0, np.zeros(4), fast, np.zeros(4), 600.0, cfg, None)
    assert not terminated
