"""
Reward function and early termination (WP 2300).

Scenario: large-angle rest-to-rest maneuver.
- progress:  w_p * (phi_prev - phi) on the principal error angle phi [deg]. Its sum over an episode
             is phi_0 - phi_T, so slewing back and forth cannot be exploited (a problem reported for
             PPO with shaped rewards by Elkins et al. 2022).
- goal:      +bonus per step while at rest at the target (angle AND rate below threshold),
             -time penalty otherwise -> rewards reaching and holding the target quickly.
- effort:    -w_u * sum|a_i|
- optional rate and wheel saturation penalties.
- termination: one-time penalty when the spacecraft spins up beyond max_rate (episode ends).
"""

import numpy as np


def is_goal(angle_deg, rate, cfg):
    """At rest at the target: error angle [deg] and rate [rad/s] below the goal thresholds."""
    return angle_deg <= cfg["goal_angle_deg"] and rate <= cfg["goal_rate"]


def compute_reward(angle_deg, prev_angle_deg, action, omega_BR_B, wheel_speeds, omega_max, cfg, term_cfg=None):
    """
    Args:
        angle_deg: current principal attitude error [deg]
        prev_angle_deg: principal attitude error at the previous step [deg]
        action: (N,) normalized wheel torque command in [-1, 1]
        omega_BR_B: (3,) rate error [rad/s]
        wheel_speeds: (N,) wheel speeds [rad/s]
        omega_max: max wheel speed [rad/s]
        cfg: `reward` block of configs/env.yaml
        term_cfg: `termination` block of configs/env.yaml (None = never terminate)
    Returns:
        reward (float), terms (dict) for logging, terminated (bool)
    """
    rate = float(np.linalg.norm(omega_BR_B))
    terms = {
        "progress": cfg["w_progress"] * (prev_angle_deg - angle_deg),
        "goal": cfg["goal_bonus"] if is_goal(angle_deg, rate, cfg) else -cfg["time_penalty"],
        "effort": -cfg["w_effort"] * float(np.sum(np.abs(action))),
        "rate": -cfg["w_rate"] * rate,
        "wheel_saturation": 0.0,
        "termination": 0.0,
    }
    if np.max(np.abs(wheel_speeds)) > cfg["saturation_ratio"] * omega_max:
        terms["wheel_saturation"] = -cfg["w_wheel_saturation"]

    terminated = term_cfg is not None and rate > term_cfg["max_rate"]
    if terminated:
        terms["termination"] = -term_cfg["penalty"]
    return float(sum(terms.values())), terms, terminated
