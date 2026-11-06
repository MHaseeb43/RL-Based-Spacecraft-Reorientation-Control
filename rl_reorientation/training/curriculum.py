"""Curriculum learning phases (WP 4100)."""

from rl_reorientation.config import load_yaml

REQUIRED_KEYS = ("name", "timesteps", "initial_angle_deg", "initial_rate_deg_s", "max_steps", "uncertainty_scale")


def load_curriculum(path):
    """Load and validate a curriculum file. Returns (name, list of phase dicts)."""
    cur = load_yaml(path)
    phases = cur["phases"]
    for i, phase in enumerate(phases):
        missing = [k for k in REQUIRED_KEYS if k not in phase]
        if missing:
            raise ValueError(f"Curriculum phase {i} is missing keys: {missing}")
        for key in ("initial_angle_deg", "initial_rate_deg_s"):
            lo, hi = phase[key]
            if lo > hi:
                raise ValueError(f"Curriculum phase {i}: {key} has min > max")
    return cur.get("name", "curriculum"), phases
