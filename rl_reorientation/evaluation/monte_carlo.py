"""
Episode rollouts and Monte Carlo simulation (WP 5100 / 5200 / 5300).

A controller is any callable `controller(obs, info) -> action` with action in [-1, 1]^N,
so trained SB3 agents and the PD baseline are evaluated on identical scenarios (same seeds).
"""

import numpy as np
import pandas as pd

from rl_reorientation.evaluation.metrics import episode_metrics


def sb3_controller(model):
    """Wrap a Stable Baselines3 model as a deterministic controller."""
    return lambda obs, info: model.predict(obs, deterministic=True)[0]


def pd_controller(pd):
    """Wrap a PDController (uses the true physical state from `info`)."""
    return lambda obs, info: pd(info)


def run_episode(env, controller, seed=None):
    """Roll out one episode. Returns a dict of time histories."""
    obs, info = env.reset(seed=seed)
    hist = {k: [info[k]] for k in ("time", "angle_deg", "sigma_BR", "omega_BR_B", "wheel_speeds", "wheel_torques")}
    hist["reward"] = [0.0]
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(controller(obs, info))
        done = terminated or truncated
        for k in hist:
            hist[k].append(reward if k == "reward" else info[k])
    return {k: np.asarray(v) for k, v in hist.items()}


def monte_carlo(env, controller, n_episodes, settling_cfg, base_seed=0):
    """Run `n_episodes` seeded episodes and return a DataFrame with one row of metrics per episode."""
    rows = []
    for i in range(n_episodes):
        h = run_episode(env, controller, seed=base_seed + i)
        m = episode_metrics(h["time"], h["angle_deg"], np.linalg.norm(h["omega_BR_B"], axis=1),
                            h["wheel_torques"], env.sim.dt,
                            settling_cfg["angle_threshold_deg"], settling_cfg["rate_threshold"])
        m["seed"] = base_seed + i
        m["return"] = float(np.sum(h["reward"]))
        rows.append(m)
    return pd.DataFrame(rows)
