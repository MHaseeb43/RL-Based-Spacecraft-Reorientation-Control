"""Project paths and loading of the yaml config files"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_yaml(path):
    """Load a yaml file. A relative path like "env.yaml" is looked up in the configs folder"""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_project_config(spacecraft="spacecraft.yaml", env="env.yaml", uncertainty="uncertainty.yaml"):
    """Load the spacecraft, environment and uncertainty configs into one dict"""
    return {
        "spacecraft": load_yaml(spacecraft),
        "env": load_yaml(env),
        "uncertainty": load_yaml(uncertainty),
    }
