# WP 1200 – Literature Review: Deep Reinforcement Learning for Spacecraft Reorientation

Project: Robust RL-based Spacecraft Reorientation Control (JMU Würzburg, supervisor Dr.-Ing. J. Yang)
Scope of this WP: deep RL fundamentals, the algorithms SAC and PPO, curriculum learning, and RL for spacecraft
attitude control. The output is a list of design decisions for WP 2000 (RL framework) and WP 4000 (training).

---

## 1. Sources

| Group | Source | Provided by supervisor |
|---|---|---|
| RL fundamentals | Sutton & Barto, *Reinforcement Learning: An Introduction* (2018) | – |
| | MathWorks, *Reinforcement Learning with MATLAB* (eBook) | yes |
| | Lapan, *Deep Reinforcement Learning Hands-On*, 2nd ed. (2020), Ch. 11–12 (policy gradients, actor-critic), 17 (continuous actions), 19 (PPO, TRPO, SAC) | yes |
| Algorithms | Lillicrap et al. 2016 (DDPG), Fujimoto et al. 2018 (TD3), Haarnoja et al. 2018a/b (SAC), Schulman et al. 2016 (GAE), 2017 (PPO) | – |
| Training methodology | Bengio et al. 2009, Narvekar et al. 2020 (curriculum learning); Tobin et al. 2017, Peng et al. 2018 (domain randomization); Ng et al. 1999 (reward shaping); Henderson et al. 2018 (reproducibility) | – |
| RL for attitude control | Elkins, Sood & Rumpf 2022 (TD3 vs PPO, reward design) | yes |
| | Retagne, Dauer & Waxenegger-Wilfing 2024 (SAC/PPO in Basilisk, varying mass, stacked observations) | yes |
| | Example code `Main_attitude_control_DDPG_step4.py` (Gymnasium env + SB3 DDPG) | yes |
| | Allison et al. 2019 (PPO attitude control), Tipaldi et al. 2022 (review) | – (cited in the papers above) |
| Attitude dynamics | Schaub & Junkins, *Analytical Mechanics of Space Systems*, 4th ed., Ch. 3 (attitude parameterizations) | yes |
| | Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control* (2014) | yes |
| Tools | Raffin et al. 2021 (Stable-Baselines3), Basilisk documentation | – / website |

> Sources marked "–" are standard references that are not in the supervisor folder. Get them and check the
> details before citing them in the report.

---

## 2. Reinforcement learning fundamentals

**Markov decision process (MDP).** RL treats control as an MDP $(\mathcal S, \mathcal A, P, r, \gamma)$. In state
$s_t$ the agent picks an action $a_t \sim \pi(\cdot \mid s_t)$. The environment moves to $s_{t+1} \sim P(\cdot \mid s_t, a_t)$
and returns a reward $r_t$. The agent maximizes the expected discounted return
$G_t = \sum_{k\ge 0} \gamma^k r_{t+k}$ (Sutton & Barto 2018).

**Value functions.** $V^\pi(s)$ and $Q^\pi(s,a)$ are the expected returns from a state, or from a state and action, under
policy $\pi$. Both satisfy Bellman equations, which value-based and actor-critic methods use for learning. The
MathWorks eBook makes the analogy to optimal control: the reward is the negative of an LQR-style cost, but it doesn't
have to be quadratic.

**Discount factor and horizon.** $\gamma$ sets an effective planning horizon of about $1/(1-\gamma)$ steps.
Multiplied by the control step $\Delta t$, this has to cover the slowest important dynamics. For a slew, that's the
braking phase. This links $\gamma$ to the time step choice in WP 2000 (see WP 2000 document, Sec. 2).

**Policy gradient and actor-critic.** For continuous actions such as torques, the policy is a neural network
(the actor) trained by gradient ascent on the return. A learned value function (the critic) reduces the variance
of the gradient estimate (Lapan 2020, Ch. 11–12).

**On-policy vs. off-policy.** On-policy methods (PPO) learn only from data collected by the current policy and
discard it after each update. They are stable but sample-inefficient. Off-policy methods (DDPG, TD3, SAC) store
transitions in a replay buffer and reuse them. They are more sample-efficient, which matters when each simulator
step is expensive, as with Basilisk (Elkins et al. 2022; Retagne et al. 2024).

**Partial observability.** If the observation misses quantities that affect the dynamics, such as the true inertia
or a disturbance torque, the problem is a partially observable MDP. Common remedies are stacking past observations
(Mnih et al. 2015) or recurrent networks (Elkins et al. 2022, Sec. II.C).

---

## 3. Algorithms

| | DDPG / TD3 | SAC | PPO |
|---|---|---|---|
| Type | off-policy, deterministic actor | off-policy, stochastic actor, maximum entropy | on-policy, stochastic actor |
| Core idea | Q-learning with a deterministic policy gradient. TD3 adds twin critics, delayed actor updates and target smoothing against Q overestimation | maximize return + $\alpha \cdot$ policy entropy; twin soft-Q critics; temperature $\alpha$ tuned automatically (Haarnoja 2018b) | clipped surrogate objective limits each policy update; GAE advantage estimate |
| Exploration | added Gaussian action noise | built into the entropy term | sampling from the Gaussian policy; shrinks as entropy decreases |
| Sample efficiency | high | high | low (many parallel envs needed) |
| Wall-clock per sample | higher (gradient step per env step) | higher | lower |
| Sensitivity | TD3 more robust than DDPG | robust to hyperparameters | sensitive to reward shaping (Elkins 2022) |
| Used in | Elkins 2022 (TD3), example code (DDPG) | Retagne 2024 | Allison 2019, Elkins 2022, Retagne 2024 |
| SB3 | `DDPG`, `TD3` | `SAC` | `PPO` |

**Why SAC and PPO for this project.** The task description names both. SAC is the strongest off-policy method in
SB3 for continuous control. It keeps TD3's twin critics, and its entropy term gives exploration without hand-tuned
action noise. PPO is the standard on-policy baseline and parallelizes easily over environments. Comparing the two
also compares the off-policy and on-policy families, as Elkins et al. did for TD3 vs. PPO.

---

## 4. Reward design

- **Sparse vs. shaped rewards** (MathWorks eBook, Part 2). A reward only for reaching the goal is correct but almost
  never found by random exploration. Shaped intermediate rewards guide learning, but the agent will exploit any
  shortcut they contain ("if you give an optimization algorithm a shortcut, it'll take it").
- **Potential-based shaping** (Ng et al. 1999). A shaping term $F = \gamma\Phi(s') - \Phi(s)$ leaves the optimal policy
  unchanged. A progress reward on the attitude error, $\Phi = -\varphi$, is of this form (up to $\gamma \approx 1$).
  Its sum over an episode telescopes to the net error reduction, so oscillating earns nothing.
- **Reported exploitation** (Elkins et al. 2022). With an exponential, direction-dependent shaped reward, PPO
  converged to "quickly slewing back and forth to exploit the positive reward". A high-water-mark variant, which
  rewards only new minima of the error, was needed for PPO. TD3 learned with the plain reward.
- **Goal bonus and terminal conditions.** Elkins et al. and the example code add +9 per step inside the 0.25°
  goal. Elkins et al. also give +50 if the episode ends inside the goal, and −25 with early termination if
  $|\omega| > 0.5$ rad/s. Retagne et al. terminate with −30 if the error exceeds a threshold, and penalize wheel speeds
  above 6000 rpm.
- **Control effort.** Elkins et al. subtract the normalized torque magnitude. Without it they saw jerky commands.

---

## 5. Curriculum learning and robustness

- **Curriculum learning** (Bengio et al. 2009; survey: Narvekar et al. 2020). Training on easier tasks first and
  raising the difficulty speeds up convergence and can reach better solutions. For RL, the curriculum can change the
  initial state distribution, the task parameters or the environment dynamics. Phases can switch on a fixed
  schedule or when a performance threshold is reached. The survey discusses both.
- **Domain randomization** (Tobin et al. 2017; Peng et al. 2018). Sampling simulator parameters (mass, inertia,
  friction, disturbances) from a distribution every episode gives policies that work across the whole range, and
  is the standard way to get robustness to model uncertainty. Elkins et al. cite Peng et al. for this purpose.
- **Combining the two.** Raising the randomization range in steps (nominal → ±10 % → ±30 % inertia) is a
  curriculum over dynamics. It avoids confronting an untrained agent with the hardest dynamics.
- **Inferring hidden parameters** (Retagne et al. 2024). Agents trained over a 10–1000 kg mass range without
  history learned one compromise policy. It oscillated at low masses: SAC ended at 81° error at 15 kg. With the
  last 5 observations stacked, the agents adapted their torque to the mass and beat a PID controller. PPO with
  stacking was best overall.

---

## 6. RL for spacecraft attitude control

| | Elkins et al. 2022 | Retagne et al. 2024 | Example code (supervisor) |
|---|---|---|---|
| Simulator | own rigid body, RK4 | **Basilisk** + RLlib | own rigid body, RK4, numba |
| Spacecraft / actuator | LM50, external torque ≤ 0.5 N·m per axis | 3 axis-aligned RWs (HR16, ±0.2 N·m) | external torque, 3 axes |
| Attitude representation | error quaternion | MRP | quaternion |
| State | $q_e, \dot q_e, \omega$ | $\sigma_{BN}, \omega_{BN}, \Omega$ (+ stacking) | $q, \omega, q_{0,\text{prev}}$ |
| Action | 3-axis torque | 3-axis torque → `rwMotorTorque` | 3-axis torque (normalized) |
| Scenario | rest-to-rest, 30–150°, random axis | detumble/slew from 180° about x, mass 10–1000 kg | fixed initial state |
| Reward | exp-shaped progress − effort, +9 in goal, +50 terminal, −25 spin-up | −‖σ‖² + per-axis bonuses, wheel saturation −1, −30 termination | exp-shaped progress, +9 in goal |
| Algorithms | TD3 > PPO (PPO needed HWM reward) | PPO-stacked ≥ SAC-stacked > SAC, PPO; PID consistent but fails at low mass | DDPG |
| Baseline | quaternion feedback | PID (Nelder–Mead tuned at 300 kg) | PD law in code |
| Notable | 0.25° goal = LM50 requirement; PPO needs stochastic actions even in evaluation | RL can't output exactly zero torque, so it never fully rests; settling measured at 15° | template for our Gymnasium env |

Earlier work: Allison et al. (2019) trained a PPO attitude controller that outperformed quaternion rate feedback and
worked for a wide range of masses. Tipaldi et al. (2022) review RL in spacecraft control, with attitude control as
one of the most active topics.

**Gaps our project addresses.** Neither paper combines (a) a reaction wheel model in Basilisk, (b) large-angle
rest-to-rest slews over random axes, (c) inertia uncertainty and constant + periodic disturbance torques, and
(d) a strict accuracy requirement (0.25°) with a PD comparison on identical Monte Carlo scenarios.

---

## 7. Attitude representation for the agent

Schaub & Junkins (Ch. 3) and Markley & Crassidis (Ch. 2–3) compare parameterizations:

| | Quaternion $q$ (Euler parameters) | MRP $\sigma$ | DCM / 6D |
|---|---|---|---|
| Size | 4 (unit norm) | 3 | 9 / 6 |
| Singularities | none | at 360° (avoided with the shadow set) | none |
| Uniqueness | $q$ and $-q$ are the same attitude | shadow set gives a unique short rotation | unique |
| Continuity of a unique choice | sign fix $q_0 \ge 0$ jumps at exactly 180° | shadow switch jumps at 180° | continuous |

- The project uses **quaternions** (supervisor requirement). Basilisk propagates MRPs, and
  `RigidBodyKinematics.MRP2EP` converts them. For $|\sigma| \le 1$, $q_0 = (1-\sigma^2)/(1+\sigma^2) \ge 0$, so the
  short-rotation MRP directly gives the short-rotation quaternion.
- Fixing the sign is necessary: otherwise the network sees two different inputs for the same physical state.
- Any unique 3- or 4-parameter choice is discontinuous somewhere. For a short-rotation choice that point is at
  exactly 180°, where both rotation directions are equally short. Zhou et al. (2019) show that continuous
  representations such as 6D (two DCM columns) are easier for neural networks to learn. The 180° point is a single
  attitude and the agent leaves it within a few steps, so the quaternion is adequate. 6D remains a fallback option.
- Ignoring the sign of $q_0$ in a quaternion feedback law can make the spacecraft take the long way round
  ("unwinding", Bhat & Bernstein 2000). The example code's PD law avoids this with `sign(q0)`.

---

## 8. Implications for this project

| # | Finding | Decision | Where |
|---|---|---|---|
| 1 | Supervisor requires quaternions; $q$ / $-q$ ambiguity | state uses $q_{BR}$ with $q_0 \ge 0$ | WP 2100, `configs/env.yaml` |
| 2 | Hidden inertia/disturbance makes the problem partially observable (Retagne 2024) | observation stacking (`stack_frames`), with the previous action so the cause of each response is visible; supervisor: worth trying, choose k by a performance vs. training cost sweep | WP 2100, WP 4000 |
| 3 | Literature commands 3-axis torque; we have 4 redundant wheels | 3-axis body torque as default, split across the wheels by Basilisk `rwMotorTorque` (supervisor decision); wheel torques kept as option | WP 2200, WP 3100 |
| 4 | Shaped rewards get exploited, especially by PPO (Elkins 2022) | potential-based progress term (telescoping), no direction-dependent bonus | WP 2300 |
| 5 | Goal bonus and early termination used in both papers | per-step bonus at rest in the 0.25° goal; termination with penalty on spin-up | WP 2300 |
| 6 | RL agents don't output exactly zero torque (Retagne 2024) | effort penalty; goal requires low rate; evaluate steady-state error explicitly | WP 2300, WP 5400 |
| 7 | Horizon $\Delta t/(1-\gamma)$ must cover the braking phase | choose $\Delta t$, $\gamma$ together after Bevo2-Sat parameters are known | WP 2000 / 4000 |
| 8 | Curriculum + domain randomization give robustness (Bengio 2009; Peng 2018) | phases: small → large slews, then ±10 % → ±30 % inertia + disturbances | WP 4100 |
| 9 | Results vary strongly across seeds (Henderson 2018) | train and report several seeds per algorithm | WP 4200/4300 |
| 10 | PD/PID has a steady-state offset under constant torque and degrades with model error | compare RL vs. PD under the same disturbances; expected main result | WP 5000 |

---

## References

- Allison, J., West, M., Ghosh, A., Vedant, F. (2019). Reinforcement learning for spacecraft attitude control. *70th International Astronautical Congress*.
- Bengio, Y., Louradour, J., Collobert, R., Weston, J. (2009). Curriculum learning. *ICML 2009*.
- Bhat, S. P., Bernstein, D. S. (2000). A topological obstruction to continuous global stabilization of rotational motion and the unwinding phenomenon. *Systems & Control Letters*, 39.
- Elkins, J. G., Sood, R., Rumpf, C. (2022). Bridging reinforcement learning and online learning for spacecraft attitude control. *Journal of Aerospace Information Systems*, 19(1), 62–69. doi:10.2514/1.I010958
- Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing function approximation error in actor-critic methods. *ICML 2018*.
- Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018a). Soft actor-critic: Off-policy maximum entropy deep RL with a stochastic actor. *ICML 2018*.
- Haarnoja, T., et al. (2018b). Soft actor-critic algorithms and applications. arXiv:1812.05905.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D. (2018). Deep reinforcement learning that matters. *AAAI 2018*.
- Lapan, M. (2020). *Deep Reinforcement Learning Hands-On*, 2nd ed. Packt.
- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. *ICLR 2016*.
- Markley, F. L., Crassidis, J. L. (2014). *Fundamentals of Spacecraft Attitude Determination and Control*. Springer.
- MathWorks. *Reinforcement Learning with MATLAB* (eBook).
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., Stone, P. (2020). Curriculum learning for reinforcement learning domains: A framework and survey. *JMLR*, 21.
- Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML 1999*.
- Peng, X. B., Andrychowicz, M., Zaremba, W., Abbeel, P. (2018). Sim-to-real transfer of robotic control with dynamics randomization. *ICRA 2018*.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *JMLR*, 22.
- Retagne, W., Dauer, J., Waxenegger-Wilfing, G. (2024). Adaptive satellite attitude control for varying masses using deep reinforcement learning. *Frontiers in Robotics and AI*, 11:1402846. doi:10.3389/frobt.2024.1402846
- Schaub, H., Junkins, J. L. (2018). *Analytical Mechanics of Space Systems*, 4th ed. AIAA.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR 2016*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Tipaldi, M., Iervolino, R., Massenio, P. R. (2022). Reinforcement learning in spacecraft control applications: Advances, prospects, and challenges. *Annual Reviews in Control*, 54, 1–23.
- Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS 2017*.
- Zhou, Y., Barnes, C., Lu, J., Yang, J., Li, H. (2019). On the continuity of rotation representations in neural networks. *CVPR 2019*.
