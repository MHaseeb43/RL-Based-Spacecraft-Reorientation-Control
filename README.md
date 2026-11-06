# Robust RL-based Spacecraft Reorientation Control

Deep reinforcement learning (SAC, PPO) for spacecraft attitude reorientation with reaction wheels,
trained with curriculum learning in a [Basilisk](https://avslab.github.io/basilisk/) simulation and
evaluated against a classical PD controller under model uncertainties and disturbances.

Project at the chair of Space Informatics and Satellite Systems, Julius-Maximilians-Universität Würzburg
(supervisor: Dr.-Ing. Juntang Yang).

**Goal:** reorient the spacecraft from an arbitrary initial attitude to a target attitude with
- high control accuracy,
- a fast maneuver (short settling time),
- low control effort,

in a large-angle rest-to-rest maneuver, and stay robust against moment of inertia uncertainty (±10–30 %) and
constant and periodic external disturbance torques.

---

## Setup

Requires Python 3.9 – 3.13 (developed with 3.11).

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
pip install -e .                    # makes the rl_reorientation package importable
pytest                              # sanity check
```

Basilisk is installed from PyPI as `bsk` (imported as `Basilisk`). The PyTorch build that
Stable Baselines3 pulls in is CPU-only. For GPU training, install a CUDA build first (see `requirements.txt`).

## Project structure

```
RL-Based-Reorientation-Control/
├── configs/
│   ├── spacecraft.yaml            # hub inertia, reaction wheel layout & limits      (WP 3100)
│   ├── env.yaml                   # time step, state/action space, reward weights   (WP 2000, 3200)
│   ├── uncertainty.yaml           # uncertainty & disturbance models                (WP 3300)
│   ├── pd.yaml                    # PD baseline gains                               (WP 3400)
│   ├── algorithms/{sac,ppo}.yaml  # SB3 hyperparameters                             (WP 4200, 4300)
│   └── curriculum/default.yaml    # curriculum learning phases                      (WP 4100)
├── rl_reorientation/
│   ├── attitude.py                # MRP / quaternion / DCM kinematics               (WP 1100)
│   ├── config.py                  # paths & YAML loading
│   ├── sim/
│   │   ├── basilisk_sim.py        # Basilisk spacecraft + RW simulation             (WP 3100)
│   │   └── uncertainty.py         # per-episode sampling of uncertainties           (WP 3300)
│   ├── envs/
│   │   ├── reorientation_env.py   # Gymnasium environment                           (WP 3200)
│   │   └── reward.py              # reward function                                 (WP 2300)
│   ├── controllers/
│   │   └── pd_controller.py       # MRP PD controller baseline                      (WP 3400)
│   ├── training/
│   │   └── curriculum.py          # curriculum loading / validation                 (WP 4100)
│   └── evaluation/
│       ├── metrics.py             # settling time, accuracy, control effort         (WP 5400)
│       └── monte_carlo.py         # episode rollout & Monte Carlo simulation        (WP 5200, 5300)
├── scripts/
│   ├── train.py                   # train SAC / PPO with a curriculum               (WP 4000)
│   └── evaluate.py                # Monte Carlo evaluation, RL vs. PD               (WP 5000)
├── docs/                          # WP 1200 literature review, WP 2000 RL framework definition
├── tests/                         # unit tests (run without Basilisk)
└── outputs/                       # generated: models, TensorBoard logs, evaluation results
```

## RL framework (WP 2000) – initial design

| Element | Definition |
|---|---|
| **State** (WP 2100) | error quaternion `q_BR` (4, scalar first, `q0 ≥ 0`), rate error `ω_BR/ω_scale` (3), wheel speeds `Ω/Ω_max` (N), previous action (N); clipped to ±2; optional stacking of the last k observations |
| **Action** (WP 2200) | normalized reaction wheel motor torques `a ∈ [-1, 1]^N`, applied torque `u = a · u_max` (option: 3-axis body torque with minimum-norm wheel allocation) |
| **Reward** (WP 2300) | `w_p (φ_prev − φ)` progress on principal error angle φ [deg] · `+bonus` per step at rest at the target (φ ≤ 0.25° **and** \|ω\| ≤ 0.01 rad/s), `−time penalty` otherwise · `−w_u Σ|a_i|` effort · optional rate and wheel saturation penalties |
| **Termination** | episode ends with a one-time penalty if \|ω_BR\| exceeds `max_rate` (spacecraft spun up) |
| **Episode** | large-angle **rest-to-rest**: random target attitude, random initial error from the curriculum phase, initial rate 0 |

All weights and switches are in `configs/env.yaml`.

- **Quaternions** are the attitude parameterization required for the project. Basilisk works with MRPs, so the
  environment converts `σ_BR` (equivalent to `RigidBodyKinematics.MRP2EP`). Forcing `q0 ≥ 0` removes the
  `q` / `−q` ambiguity and always describes the short rotation. `attitude: "mrp"` is kept for comparison.
- The **progress term** telescopes: its sum over an episode is `φ_0 − φ_T`, so slewing back and forth to collect
  shaped reward (reported for PPO by Elkins et al. 2022) earns nothing.
- **Observation stacking** (`stack_frames > 1`) lets the agent infer the unknown inertia and disturbance from the
  response (Retagne et al. 2024). The default is 1; compare stacked and unstacked agents as an experiment.

## Robustness scenario (project focus)

`configs/uncertainty.yaml`, scaled per curriculum phase by `uncertainty_scale`:

- **Principal moments of inertia** `J_ii · (1 + U(−30 %, +30 %))` at scale 1 (scale 0.333 gives ±10 %). Samples
  that aren't physically valid (positive definite and triangle inequality) are redrawn.
- **External disturbance torque**: constant plus periodic (`A sin(2πt/T + φ)`, random direction, amplitude, period
  and phase), available to the simulator as `Perturbation.external_torque(t)`.
- RW misalignment, RW torque errors and sensor noise are implemented but disabled (not in the project focus).

## Curriculum learning (WP 4100)

`configs/curriculum/default.yaml` defines phases trained in sequence, with each phase continuing from the previous model:

1. small slews (0–30°), nominal model
2. large-angle slews (0–180°), nominal model
3. large-angle slews, ±10 % MOI and disturbances (`uncertainty_scale: 0.333`)
4. large-angle slews, ±30 % MOI and full disturbances (`uncertainty_scale: 1.0`)

All phases are rest-to-rest (initial rate 0).

The environment switches phases via `set_phase()`, so the same SB3 model and replay buffer continue through all phases.

## Usage

```bash
# Training (WP 4000)
python scripts/train.py --algo sac --curriculum curriculum/default.yaml --run-name sac_v1
python scripts/train.py --algo ppo --run-name ppo_v1
tensorboard --logdir outputs/tensorboard

# Monte Carlo evaluation & PD comparison (WP 5000)
python scripts/evaluate.py --model outputs/runs/sac_v1/final.zip --pd --episodes 100 --uncertainty-scale 1.0
```

Evaluation writes one CSV per controller to `outputs/evaluation/`, one row per episode, with
settling time, final and steady-state error, and control effort (L1/L2). RL agents and the PD controller
are evaluated on identical seeded scenarios.

## Status / roadmap

| WP | Task | Status |
|---|---|---|
| 1000 | Literature research | WP 1200 (deep RL) done: `docs/WP1200_literature_review.md` |
| 2100–2200 | State, action definition | done: `docs/WP2000_rl_framework.md`; to be confirmed in training |
| 2300 | Reward function | rest-to-rest goal and termination done; weights to be tuned in training |
| 3100 | Basilisk spacecraft & RW model | **interface defined, implementation next** |
| 3200 | Gymnasium wrapper | done (tested with a numpy stand-in model) |
| 3300 | Uncertainty & disturbance modelling | ±10–30 % MOI and constant + periodic torque sampling done; application in the Basilisk simulation pending |
| 3400 | PD controller baseline | done (gains to be tuned) |
| 4100 | Curriculum phases | 4-phase rest-to-rest schedule with uncertainty ramp; phase lengths to be tuned |
| 4200 / 4300 | Training with SAC / PPO | script ready, blocked by WP 3100 |
| 5100 / 5200 | Test scenarios, Monte Carlo | runner done, scenarios to be designed |
| 5300 | PD controller simulation | done via `evaluate.py --pd` |
| 5400 | Visualization & comparison | metrics done, plots pending |

> The spacecraft parameters in `configs/spacecraft.yaml` are generic small-satellite placeholders.
> They will be replaced with the **Bevo2-Sat** configuration from the Basilisk example
> `scenarioAttitudeConstraintViolation`. Then also rescale the disturbance magnitudes and `termination.max_rate`.

## Conventions

- The agent observes the error quaternion (scalar-first, `q0 ≥ 0`). Internally and in Basilisk, attitudes are
  MRPs `σ` in the short-rotation set.
- `σ_BN`: body w.r.t. inertial, `σ_RN`: reference (target) w.r.t. inertial, `σ_BR`: attitude error.
- Wheel motor torque `u_i` acts on the hub as `−Gs u`, where the columns of `Gs` are the spin axes.

## Team workflow

- `main` always passes `pytest`. Work on a branch per work package, e.g. `wp3100-basilisk-sim`, and merge
  through a pull request that another team member reviews.
- Course material (books, papers, supervisor slides) stays out of the repository (`literature_review/` is ignored).
  Share it through the university NextCloud instead.
- Generated files (`outputs/`, models, TensorBoard logs) are ignored. Share trained models as release assets if needed.
