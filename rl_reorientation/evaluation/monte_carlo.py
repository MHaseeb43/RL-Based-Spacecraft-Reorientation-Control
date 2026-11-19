"""
Run episodes and Monte Carlo simulations.

A controller is any function controller(obs, info) -> action in [-1, 1]. That way a trained agent
and the PD controller are tested on exactly the same scenarios (same seeds).
"""

import numpy as np
import pandas as pd

from rl_reorientation.evaluation.metrics import episode_metrics

# values from the info dict we save at every step
HISTORY_KEYS = ("time", "angle_deg", "sigma_BR", "omega_BR_B", "wheel_speeds", "wheel_torques")


def sb3_controller(model):
    """Turn a trained Stable Baselines3 model into a controller (no random exploration)"""
    def controller(obs, info):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return controller


def pd_controller(pd):
    """Turn a PDController into a controller. The PD uses the true state from info, not obs"""
    def controller(obs, info):
        return pd(info)
    return controller


def run_episode(env, controller, seed=None):
    """Run one episode and return the time history of every value"""
    obs, info = env.reset(seed=seed)
    hist = {}
    for key in HISTORY_KEYS:
        hist[key] = [info[key]]
    hist["reward"] = [0.0]

    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(controller(obs, info))
        done = terminated or truncated
        for key in HISTORY_KEYS:
            hist[key].append(info[key])
        hist["reward"].append(reward)

    return {key: np.asarray(values) for key, values in hist.items()}


def monte_carlo(env, controller, n_episodes, settling_cfg, base_seed=0):
    """Run n_episodes episodes (seeds base_seed, base_seed+1, ...). Returns a table with one row per episode"""
    rows = []
    for i in range(n_episodes):
        h = run_episode(env, controller, seed=base_seed + i)
        rate = np.linalg.norm(h["omega_BR_B"], axis=1)
        m = episode_metrics(h["time"], h["angle_deg"], rate, h["wheel_torques"], env.sim.dt,
                            settling_cfg["angle_threshold_deg"], settling_cfg["rate_threshold"])
        m["seed"] = base_seed + i
        m["return"] = float(np.sum(h["reward"]))
        rows.append(m)
    return pd.DataFrame(rows)
