import numpy as np

from rl_reorientation.evaluation.metrics import episode_metrics, settling_time


def test_settling_time_requires_staying_inside():
    t = np.arange(6.0)
    angle = np.array([10.0, 0.1, 5.0, 0.2, 0.1, 0.1])  # dips in at t=1 but leaves again
    rate = np.zeros(6)
    assert settling_time(t, angle, rate, 0.25, 0.01) == 3.0


def test_never_settles():
    t = np.arange(3.0)
    assert np.isnan(settling_time(t, [10.0, 5.0, 1.0], np.zeros(3), 0.25, 0.01))


def test_control_effort():
    u = np.full((10, 3), 1e-4)
    m = episode_metrics(np.arange(10) * 0.1, np.linspace(10, 0, 10), np.zeros(10), u, 0.1, 0.25, 0.01)
    assert np.isclose(m["control_effort_l1"], 30 * 1e-4 * 0.1)
    assert np.isclose(m["max_torque"], 1e-4)
