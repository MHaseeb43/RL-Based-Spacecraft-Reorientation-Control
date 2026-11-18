"""
Monte Carlo evaluation of a trained agent and/or the PD baseline (WP 5000).

Usage:
    python scripts/evaluate.py --model outputs/runs/sac_v1/final.zip --episodes 100
    python scripts/evaluate.py --pd --episodes 100
    python scripts/evaluate.py --model outputs/runs/sac_v1/final.zip --pd --uncertainty-scale 1.0
"""

import argparse
from pathlib import Path

from rl_reorientation.config import OUTPUT_DIR, load_project_config, load_yaml
from rl_reorientation.controllers import PDController
from rl_reorientation.envs import ReorientationEnv
from rl_reorientation.evaluation.monte_carlo import monte_carlo, pd_controller, sb3_controller


def load_sb3_model(path):
    from stable_baselines3 import PPO, SAC
    for cls in (SAC, PPO):
        try:
            return cls.load(path)
        except (ValueError, KeyError, TypeError):
            continue
    raise ValueError(f"Could not load {path} as SAC or PPO model")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="path to a trained SB3 model (.zip)")
    parser.add_argument("--pd", action="store_true", help="also evaluate the PD baseline")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000, help="base seed (different from training seeds)")
    parser.add_argument("--initial-angle", type=float, nargs=2, default=[0.0, 180.0], metavar=("MIN", "MAX"))
    parser.add_argument("--initial-rate", type=float, nargs=2, default=[0.0, 0.0], metavar=("MIN", "MAX"))
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    args = parser.parse_args()
    if not args.model and not args.pd:
        parser.error("give --model and/or --pd")

    config = load_project_config()
    phase = {"name": "evaluation", "initial_angle_deg": args.initial_angle, "initial_rate_deg_s": args.initial_rate,
             "max_steps": args.max_steps, "uncertainty_scale": args.uncertainty_scale}
    env = ReorientationEnv(config, phase=phase)

    controllers = {}
    if args.model:
        controllers[Path(args.model).stem] = sb3_controller(load_sb3_model(args.model))
    if args.pd:
        gains = load_yaml("pd.yaml")   # same action interface as the agent (body torque -> rwMotorTorque)
        controllers["pd"] = pd_controller(PDController.for_env(env, gains["K"], gains["P"]))

    out_dir = OUTPUT_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, controller in controllers.items():
        df = monte_carlo(env, controller, args.episodes, config["env"]["settling"], base_seed=args.seed)
        df.to_csv(out_dir / f"{name}_mc.csv", index=False)
        print(f"\n=== {name} ({args.episodes} episodes) ===")
        print(df.drop(columns="seed").describe().loc[["mean", "std", "min", "max"]].T.to_string())
        print(f"settled: {df['settling_time_s'].notna().mean():.1%}")
    # TODO(WP 5400): plots (attitude error, rates, wheel speeds/torques) and RL vs. PD comparison


if __name__ == "__main__":
    main()
