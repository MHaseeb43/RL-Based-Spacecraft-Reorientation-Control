"""
Reward function and early termination.

The task is a rest-to-rest maneuver, so the agent has to reach the target AND stop there.
Reward terms (the weights are in configs/env.yaml):
- progress:     reward for making the error angle smaller. Over a whole episode this adds up to
                start angle - end angle, so turning back and forth doesn't earn anything
- goal:         bonus for every step at rest on the target, small time penalty for every other step
- effort:       penalty for using a lot of torque
- rate, wheel saturation: optional penalties
- termination:  big one-time penalty if the satellite spins too fast, then the episode ends
"""

import numpy as np


def is_goal(angle_deg, rate, cfg):
    """True if we are on the target and at rest (angle and rate both small enough)"""
    return angle_deg <= cfg["goal_angle_deg"] and rate <= cfg["goal_rate"]


def compute_reward(angle_deg, prev_angle_deg, action, omega_BR_B, wheel_speeds, omega_max, cfg, term_cfg=None):
    """
    :param angle_deg:  attitude error now [deg]
    :param prev_angle_deg:  attitude error one step before [deg]
    :param action:  normalized action of the agent, values in [-1, 1]
    :param omega_BR_B:  angular velocity error [rad/s]
    :param wheel_speeds:  reaction wheel speeds [rad/s]
    :param omega_max:  max wheel speed [rad/s]
    :param cfg:  "reward" part of env.yaml
    :param term_cfg:  "termination" part of env.yaml (None = the episode never ends early)
    :return: total reward, dict with every term (for logging), terminated
    """
    rate = float(np.linalg.norm(omega_BR_B))

    progress = cfg["w_progress"] * (prev_angle_deg - angle_deg)

    if is_goal(angle_deg, rate, cfg):
        goal = cfg["goal_bonus"]
    else:
        goal = -cfg["time_penalty"]

    effort = -cfg["w_effort"] * float(np.sum(np.abs(action)))
    rate_penalty = -cfg["w_rate"] * rate

    # penalty if any wheel gets close to its max speed
    wheel_saturation = 0.0
    if np.max(np.abs(wheel_speeds)) > cfg["saturation_ratio"] * omega_max:
        wheel_saturation = -cfg["w_wheel_saturation"]

    # stop the episode if the satellite spins too fast
    terminated = term_cfg is not None and rate > term_cfg["max_rate"]
    termination = 0.0
    if terminated:
        termination = -term_cfg["penalty"]

    terms = {
        "progress": progress,
        "goal": goal,
        "effort": effort,
        "rate": rate_penalty,
        "wheel_saturation": wheel_saturation,
        "termination": termination,
    }
    return float(sum(terms.values())), terms, terminated
