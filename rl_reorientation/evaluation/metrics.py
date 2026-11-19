"""
Metrics to compare the controllers: settling time, accuracy and control effort.
"""

import numpy as np


def settling_time(time, angle_deg, rate, angle_threshold_deg, rate_threshold):
    """
    First time from which the angle error AND the rate stay below their thresholds until the end.
    Returns nan if the satellite never settles.
    """
    inside = (np.asarray(angle_deg) <= angle_threshold_deg) & (np.asarray(rate) <= rate_threshold)
    if not inside[-1]:
        return np.nan

    # settled = one step after the last time we were outside
    outside_idx = np.flatnonzero(~inside)
    if outside_idx.size == 0:
        first_settled = 0
    else:
        first_settled = outside_idx[-1] + 1
    return float(np.asarray(time)[first_settled])


def episode_metrics(time, angle_deg, rate, wheel_torques, dt, angle_threshold_deg, rate_threshold,
                    steady_state_window=0.1):
    """
    :param time:  time of every step [s]
    :param angle_deg:  attitude error of every step [deg]
    :param rate:  |omega_BR_B| of every step [rad/s]
    :param wheel_torques:  applied wheel torques, one row per step [Nm]
    :param dt:  control step [s]
    :param steady_state_window:  last part of the episode (fraction) used for the steady state error
    """
    angle_deg = np.asarray(angle_deg)
    u = np.asarray(wheel_torques)
    n_ss = max(1, int(len(angle_deg) * steady_state_window))

    max_torque = 0.0
    if u.size:
        max_torque = float(np.max(np.abs(u)))

    return {
        "initial_error_deg": float(angle_deg[0]),
        "final_error_deg": float(angle_deg[-1]),
        "steady_state_error_deg": float(np.mean(angle_deg[-n_ss:])),
        "settling_time_s": settling_time(time, angle_deg, rate, angle_threshold_deg, rate_threshold),
        "control_effort_l1": float(np.sum(np.abs(u)) * dt),  # sum of |u_i| over time [Nms]
        "control_effort_l2": float(np.sum(u ** 2) * dt),     # sum of u_i^2 over time [N^2 m^2 s]
        "max_torque": max_torque,
    }
