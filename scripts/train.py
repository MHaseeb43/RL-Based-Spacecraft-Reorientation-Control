"""
Train a SAC or PPO agent with curriculum learning.

Usage:
    python scripts/train.py --algo sac --curriculum curriculum/default.yaml --run-name sac_v1
    tensorboard --logdir outputs/tensorboard
"""

import argparse

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_reorientation.config import OUTPUT_DIR, load_project_config, load_yaml
from rl_reorientation.envs import ReorientationEnv
from rl_reorientation.training import load_curriculum

ALGORITHMS = {"sac": SAC, "ppo": PPO}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=ALGORITHMS, default="sac")
    parser.add_argument("--curriculum", default="curriculum/default.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_name = args.run_name or f"{args.algo}_seed{args.seed}"
    run_dir = OUTPUT_DIR / "runs" / run_name
    config = load_project_config()
    algo_cfg = load_yaml(f"algorithms/{args.algo}.yaml")
    _, phases = load_curriculum(args.curriculum)

    # several environments in parallel (separate processes) if n_envs > 1
    n_envs = algo_cfg.get("n_envs", 1)
    vec_env_cls = SubprocVecEnv if n_envs > 1 else None
    env = make_vec_env(ReorientationEnv, n_envs=n_envs, seed=args.seed,
                       env_kwargs={"config": config, "phase": phases[0]},
                       vec_env_cls=vec_env_cls, monitor_dir=str(run_dir / "monitor"))

    model = ALGORITHMS[args.algo](algo_cfg["policy"], env, seed=args.seed, device=args.device, verbose=1,
                                  tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
                                  **algo_cfg["hyperparameters"])

    # save the model every 100k steps (and the replay buffer for SAC, so we can continue training later)
    checkpoint = CheckpointCallback(save_freq=max(100_000 // n_envs, 1), save_path=str(run_dir / "checkpoints"),
                                    name_prefix=run_name, save_replay_buffer=(args.algo == "sac"))

    # train the phases one after the other with the same model, so it keeps what it learned
    for i, phase in enumerate(phases):
        print(f"=== Curriculum phase {i + 1}/{len(phases)}: {phase['name']} ({phase['timesteps']} steps) ===")
        env.env_method("set_phase", phase)
        model.learn(total_timesteps=phase["timesteps"], callback=checkpoint, tb_log_name=run_name,
                    reset_num_timesteps=(i == 0), progress_bar=True)
        model.save(run_dir / f"phase{i + 1}")

    model.save(run_dir / "final")
    env.close()


if __name__ == "__main__":
    main()
