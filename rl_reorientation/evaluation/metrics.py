"""
Performance metrics (WP 5400): settling time, control accuracy and control effort.
"""

import numpy as np


def settling_time(time, angle_deg, rate, angle_threshold_deg, rate_threshold):
    """
    First time after which the error angle AND rate stay below their thresholds until the end.
    Returns np.nan if the spacecraft never settles.
    """
    inside = (np.asarray(angle_deg) <= angle_threshold_deg) & (np.asarray(rate) <= rate_threshold)
    if not inside[-1]:
        return np.nan
    outside_idx = np.flatnonzero(~inside)
    first_settled = 0 if outside_idx.size == 0 else outside_idx[-1] + 1
    return float(np.asarray(time)[first_settled])


def episode_metrics(time, angle_deg, rate, wheel_torques, dt, angle_threshold_deg, rate_threshold,
                    steady_state_window=0.1):
    """
    Args:
        time: (T,) [s]
        angle_deg: (T,) principal attitude error [deg]
        rate: (T,) |omega_BR_B| [rad/s]
        wheel_torques: (T, N) applied RW motor torques [N m]
        dt: control step [s]
        steady_state_window: fraction of the episode (at the end) used for steady-state accuracy
    """
    angle_deg = np.asarray(angle_deg)
    u = np.asarray(wheel_torques)
    n_ss = max(1, int(len(angle_deg) * steady_state_window))
    return {
        "initial_error_deg": float(angle_deg[0]),
        "final_error_deg": float(angle_deg[-1]),
        "steady_state_error_deg": float(np.mean(angle_deg[-n_ss:])),
        "settling_time_s": settling_time(time, angle_deg, rate, angle_threshold_deg, rate_threshold),
        "control_effort_l1": float(np.sum(np.abs(u)) * dt),        # integral of sum_i |u_i| dt  [N m s]
        "control_effort_l2": float(np.sum(u ** 2) * dt),           # integral of ||u||^2 dt     [N^2 m^2 s]
        "max_torque": float(np.max(np.abs(u))) if u.size else 0.0,
    }
