# WP 2000 – RL Framework Definition (State, Action, Reward)

Project: Robust RL-based Spacecraft Reorientation Control
Implementation: `configs/env.yaml`, `rl_reorientation/envs/reorientation_env.py`, `rl_reorientation/envs/reward.py`
Background and references: `docs/WP1200_literature_review.md`

---

## 1. Control problem

| | |
|---|---|
| Maneuver | large-angle **rest-to-rest** reorientation: from an arbitrary attitude at rest to a target attitude at rest |
| Actuators | reaction wheels (spacecraft: Bevo2-Sat from Basilisk `scenarioAttitudeConstraintViolation`; placeholder values until WP 3100 delivers them) |
| Requirements | control accuracy (goal 0.25°), fast maneuver (short settling time), low control effort |
| Robustness | principal moments of inertia ±10–30 %, constant and periodic external disturbance torques |
| Comparison | PD controller on identical Monte Carlo scenarios (WP 5000) |

The agent replaces the attitude controller. Every control step $\Delta t$ it reads the observation and commands
reaction wheel motor torques. Basilisk integrates the dynamics with a finer step.

---

## 2. MDP formulation

| Element | Definition | Config |
|---|---|---|
| Control step | $\Delta t = 0.1$ s (agent decision rate 10 Hz) | `simulation.dt` |
| Integration step | 0.01 s (Basilisk) | `simulation.integration_dt` |
| Episode length | up to `max_steps` (3000 steps = 300 s), set per curriculum phase | `simulation.max_steps` |
| Initial state | random target attitude $\sigma_{RN}$; initial error angle $\varphi_0$ from the phase range about a uniformly random axis; $\omega_0 = 0$, wheels at rest | curriculum phase |
| Dynamics randomization | one perturbation sample per episode (inertia, disturbance torque) | `configs/uncertainty.yaml` |
| Termination | spin-up: $\lVert\omega_{BR}\rVert > \omega_{\text{lim}}$ (penalty, true terminal state) | `termination` |
| Truncation | time limit reached. Reported as `truncated`, so SB3 bootstraps the value and doesn't treat it as terminal | – |
| Discount | $\gamma$ (algorithm config, currently 0.99) | `configs/algorithms/*.yaml` |

**Time step and discount (open point).** The effective planning horizon is about $\Delta t/(1-\gamma)$. That's
10 s for 0.1 s and 0.99. With the placeholder spacecraft, a full-torque braking phase lasts on the order of 10–20 s.
The agent may then brake too late and overshoot. Once the Bevo2-Sat parameters are known, compute the braking time
$\omega_{\max}/\dot\omega_{\max}$ and choose $\Delta t$ and $\gamma$ so the horizon covers it, for example $\gamma = 0.995$
at 0.1 s (20 s) or $\Delta t = 0.5$ s at 0.99 (50 s). For reference, Elkins et al. used about 11 Hz and Retagne et
al. 0.1 Hz.

---

## 3. State space (WP 2100)

One observation frame:

$$
o_t = \Big[\; q_{BR} \;,\; \tfrac{\omega_{BR}^B}{\omega_s} \;,\; \tfrac{\Omega}{\Omega_{\max}} \;,\; a_{t-1} \;\Big]
\in [-2, 2]^{\,4 + 3 + N + n_a}
$$

| Component | Dim | Normalization | Why |
|---|---|---|---|
| Error quaternion $q_{BR}$ (scalar first, $q_0 \ge 0$) | 4 | unit norm | attitude error relative to the target. The supervisor requires quaternions. Fixing $q_0 \ge 0$ removes the $q$/$-q$ ambiguity and always describes the short rotation |
| Rate error $\omega_{BR}^B$ | 3 | ÷ $\omega_s$ = 0.1 rad/s | needed to brake; together with $q$ this is the full rigid-body state |
| Wheel speeds $\Omega$ | N | ÷ $\Omega_{\max}$ | wheels have a speed limit (saturation). Stored wheel momentum also shapes the hub dynamics through the gyroscopic term |
| Previous action $a_{t-1}$ | $n_a$ | already in [−1, 1] | lets the policy produce smooth torque profiles and react to its own last command |

- With 4 wheels and wheel-torque actions, the frame has **15** inputs.
- Values are clipped to ±2 to match the declared observation space. Termination limits the rate before clipping matters.
- **Conversion.** Basilisk provides the MRP error $\sigma_{BR}$ (shadow set, $|\sigma| \le 1$). The quaternion is
  $q_0 = \frac{1-\sigma^2}{1+\sigma^2},\; \mathbf q = \frac{2\sigma}{1+\sigma^2}$ (equivalent to
  `RigidBodyKinematics.MRP2EP`). $q_0 \ge 0$ follows automatically from $|\sigma| \le 1$. The principal error angle
  is $\varphi = 2\arccos q_0$.
- **Markov property.** For a rigid body with reaction wheels, $(q, \omega, \Omega)$ is the complete dynamic state.
  The true inertia and the disturbance torque change each episode but aren't observed, which makes the problem
  partially observable. Option: `stack_frames: k` feeds the last $k$ frames, so the agent can infer them from the
  response (Retagne et al. 2024 used 5). The default is 1. Compare stacked and unstacked agents in WP 4000.
- **Known limitation.** A unique short-rotation choice jumps at exactly $\varphi = 180°$, where both rotation
  directions are equally short. This holds for the quaternion with $q_0 \ge 0$ as well as for the MRP shadow set.
  It's a single attitude, so this is acceptable. A 6D (two DCM columns) representation would be the continuous
  alternative (Zhou et al. 2019).
- **Alternatives kept in code:** `attitude: "mrp"` (3-D MRP error, as in Retagne et al.) for comparison.

---

## 4. Action space (WP 2200)

| Option | `action.type` | Action | Mapping to wheels |
|---|---|---|---|
| **A (default)** | `wheel_torque` | $a \in [-1,1]^N$ | $u = a \, u_{\max}$ |
| B | `body_torque` | $a \in [-1,1]^3$ | $L = a\,L_{\max}$, $u = -G_s^{+} L$; scaled down together if any $\lvert u_i\rvert > u_{\max}$ |

$G_s = [\hat g_1 \dots \hat g_N]$ holds the **nominal** spin axes. The wheels act on the hub with $-G_s u$.

| | A: wheel torques | B: body torque + allocation |
|---|---|---|
| Size | N (4) | 3 |
| Literature | – | Retagne 2024 (`rwMotorTorque`), Elkins 2022, example code |
| Wheel redundancy | agent can use the null space of $G_s$ (4 wheels, 3 axes) to manage wheel speeds and avoid saturation | fixed minimum-norm allocation, null space unused |
| Learning difficulty | larger action space; agent must learn the allocation | smaller, directly physical |
| Matches Basilisk interface | `ArrayMotorTorqueMsg` directly | same message after allocation |
| Misalignment | learned implicitly | allocation uses nominal axes |

**Decision.** Option A is the default, because it uses all degrees of freedom of the 4-wheel configuration.
Option B is implemented so the two can be compared in WP 4000, which the literature suggests. The PD baseline always
runs with option A. `evaluate.py` refuses `--pd` with option B.

The normalized range [−1, 1] matches SB3's squashed Gaussian policies (SAC uses tanh). Physical limits are applied
by scaling, so the network never has to learn the torque magnitude scale.

---

## 5. Reward function (WP 2300)

$$
r_t = \underbrace{w_p(\varphi_{t-1} - \varphi_t)}_{\text{progress}}
\;+\; \underbrace{\begin{cases} +b & \text{at rest in goal} \\ -c & \text{otherwise}\end{cases}}_{\text{goal / time}}
\;-\; \underbrace{w_u \textstyle\sum_i |a_i|}_{\text{effort}}
\;-\; \underbrace{w_\omega \lVert\omega_{BR}\rVert}_{\text{rate (opt.)}}
\;-\; \underbrace{w_s\,[\max_i|\Omega_i| > 0.9\,\Omega_{\max}]}_{\text{wheel saturation}}
\;-\; \underbrace{P\,[\text{spin-up}]}_{\text{termination}}
$$

"At rest in goal" means $\varphi_t \le 0.25°$ **and** $\lVert\omega_{BR}\rVert \le 0.01$ rad/s. This is the same
criterion as the settling metric.

| Term | Weight | Value | Purpose |
|---|---|---|---|
| Progress ($\varphi$ in deg) | $w_p$ | 1.0 | dense guidance toward the target |
| Goal bonus / time penalty | $b$ / $c$ | 1.0 / 0.01 per step | reach the target quickly and stay there at rest |
| Effort | $w_u$ | 0.01 | low control effort; suppresses jerky commands (Elkins 2022) |
| Rate | $w_\omega$ | 0 (off) | optional extra damping if overshoot appears in training |
| Wheel saturation | $w_s$ | 0.1 per step | keep wheels below 90 % of max speed |
| Termination | $P$ | 100 (once) | spacecraft spun up beyond $\omega_{\text{lim}}$ = 0.2 rad/s |

**Design rationale.**
- **The progress term can't be exploited by oscillating.** It is potential-based with $\Phi = -w_p\varphi$
  (Ng et al. 1999). Its undiscounted sum is $w_p(\varphi_0 - \varphi_T)$, independent of the path. Under
  discounting, moving away first and coming back later is a net loss. This addresses the back-and-forth slewing
  that Elkins et al. observed with PPO.
- **The goal requires rest**, as the rest-to-rest scenario does. Otherwise the agent could pass through the target
  at speed and still collect the bonus.
- **Magnitudes.** Total progress is at most 180 per episode. The goal bonus pays off as soon as the agent settles:
  every step at rest adds 1.0. The time penalty and effort (≤ 0.04 per step with 4 wheels) are small compared with
  the progress per step during a slew (several tenths of a degree). The termination penalty is larger than any
  achievable progress, so spinning up never pays.
- **Residual torque.** Retagne et al. found that RL agents never output exactly zero torque, so the spacecraft never
  fully comes to rest. The effort penalty and the rate condition in the goal both push the action toward zero at
  the target. Steady-state error is evaluated explicitly in WP 5400.

**Known failure modes to watch in training**

| Symptom | Likely cause | Adjustment |
|---|---|---|
| Overshoot / oscillation about the target | horizon too short (Sec. 2), no damping incentive | raise $\gamma$ or $\Delta t$; set $w_\omega > 0$ |
| Reaches 1–2° but not 0.25° | progress per step too small near the goal compared with noise and effort | lower $w_u$ near the goal; raise $b$ |
| Wheels saturate on long slews | stored momentum not managed | raise $w_s$; observation already contains $\Omega$ |
| Frequent spin-up terminations early in training | exploration with full torque | expected at first; curriculum phase 1 uses small slews |

---

## 6. Parameters that depend on other WPs

| Parameter | Current | Depends on | Action |
|---|---|---|---|
| $u_{\max}$, $\Omega_{\max}$, $J$, wheel layout | placeholders | WP 3100 (Bevo2-Sat) | take from Basilisk example |
| $\omega_s$ (rate normalization), $\omega_{\text{lim}}$ | 0.1, 0.2 rad/s | max. achievable rate with Bevo2-Sat wheels | set $\omega_{\text{lim}}$ ≈ 1.5–2 × max. slew rate, $\omega_s$ ≈ max. slew rate |
| Disturbance magnitudes | 1e-5 N·m | $u_{\max}$ of Bevo2-Sat | a few % of $u_{\max}$ |
| $\Delta t$, $\gamma$ | 0.1 s, 0.99 | braking time (Sec. 2) | decide jointly with WP 4000 |
| Reward weights | Sec. 5 | training results | tune in WP 4000 |

---

## 7. Verification

Unit tests (`pytest`, 25 tests, no Basilisk needed; a numpy rigid-body model stands in for the simulator):

- Gymnasium API compliance for both action types and with frame stacking
- Quaternion observation: unit norm, $q_0 \ge 0$, $2\arccos q_0$ equals the error angle
- Body-torque allocation delivers the commanded hub torque; saturation scaling keeps its direction
- Goal bonus only at rest; progress term telescopes to zero for a back-and-forth slew; termination on spin-up
- Inertia samples stay within ±10/20/30 % and are physically valid; periodic disturbance averages to the constant part
- PD baseline converges with the nominal model (sign conventions consistent end to end)
