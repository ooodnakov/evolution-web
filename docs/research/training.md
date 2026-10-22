# Training algorithm and self-play for Evolution RL

## 0. Bottom line

**Primary recipe: PPO-clipped over the existing prefix candidate scorer, shared weights for all seats, Monte-Carlo advantage, entropy bonus, advantage normalization, mirror self-play plus a snapshot pool/PFSP.**

Why this first:
- Prefix decisions make each policy call a small `Discrete(K)` over legal candidates; masked softmax handles variable `K` directly. No fixed action head, no oracle mask.
- PPO builds on the existing `CandidatePolicy` (`rl/python/evolution_rl/train_smoke.py:548`): add old log-prob, clipped ratio, multiple epochs, advantage normalization. No replay buffer.
- On-policy PPO avoids replay-memory blow-up from variable entity sets and avoids off-policy correction.
- MAPPO evidence: PPO is competitive or superior to off-policy MARL methods in several multi-agent testbeds, including Hanabi, with minimal tuning ([MAPPO](https://arxiv.org/abs/2103.01955)).

**Strong alternative: DouZero-style Deep Monte-Carlo (DMC) Q over candidate choices, replay buffer, ε-greedy, target network, snapshot self-play.** This is the closest published evidence for a card game with huge/variable legal actions and terminal-only reward ([DouZero](https://arxiv.org/abs/2106.06135)). Use it as a pilot if PPO’s on-policy sample efficiency is the bottleneck.

**Do not start with:** AlphaZero/MuZero/ReBeL/Student of Games, full-game NFSP/CFR, or population-based training. Reasons in §6.6.

**Recommended next step after smoke:** run a **50k-episode base2p PPO pilot with snapshot self-play**, after fixing the two current trainer defects. Gate on win rate vs random/survival, entropy, value explained variance, and engine-error rate. Then scale to 200k–1M base2p episodes. `[INFERENCE]`

---

## 1. What the current smoke actually does

Grounded in repo:

- `train_smoke.py` is REINFORCE with a value baseline, not PPO. One gradient step per batch of 8 episodes; 32 episodes total; gamma 1; Adam 3e-4; shared policy/value for every seat.
- Candidate scorer shapes:
  - context: `5 * 64 = 320` — global + mean(own animals) + mean(opponents) + mean(plants) + decision context.
  - choices: `4 * 64 = 256` per candidate — choice encoding + source + target + chosen entity.
  - candidate MLP input: `576`; hidden `64`; one logit per choice.
  - value head: `320 -> 64 -> 1`.
- Measured smoke: 32 episodes in 78 s; 4 optimizer updates; 111 decisions per evaluated base2p episode; CPU-only torch by default.
- Current trainer defect 1: policy loss is a **sum** over decisions, entropy is a **mean**:
  - `policy_loss = -torch.sum(...)` at `train_smoke.py:767`
  - `entropy = torch.mean(...)` at `train_smoke.py:772`
  - With ~2,700 decisions per batch, mean entropy ≈ 1 and entropy coefficient 0.01 is effectively negligible versus policy loss ≈ 15. Fix to mean policy loss or sum entropy consistently.
- Current trainer defect 2: gradient fixture fails:
  - `rl/runs/smoke/report.json` records `gradient_fixture_error = "ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"`.
  - Therefore the current smoke does not yet prove a valid learning signal, even though losses are finite.
- Random baseline matrix run: 1,430 episodes, 196 reported errors, 1,234 natural terminals, 0 truncations, 0.536 episodes/s average (`rl/runs/matrix_00_status.json`). Fix engine errors before large-scale training; otherwise training data is biased by crashes.

The smoke is a pipeline proof. It is not a strength claim.

---

## 2. Why this regime is not standard Atari/board-game RL

| Property | Consequence |
|---|---|
| Semantic action = several prefix decisions (type → card → side → position → entity → target) | Policy factorizes autoregressively; joint log-prob is a sum. One compound action is not one `Discrete` index. |
| Legal set size `K` changes every decision | Use candidate scoring + masked softmax; no fixed action head. |
| 2–8 players, hidden hands/traits, public actions | Partial observability; shared policy across seats; opponent pool/league for >2p. |
| Terminal-only reward: winner +1, others `-1/(N-1)` | No intermediate reward; value baseline is essential; GAE adds no reward information. |
| Engine is authoritative and expensive: 2p base ≈0.7 s/episode; 8p all-addons ≈90 s/episode | Data generation dominates; sample efficiency matters more than GPU FLOPs. |
| Truncation at `maxDecisions` | Truncated episodes have reward 0; do not treat as wins; monitor and exclude if frequent. |

---

## 3. Evidence from games: what transferred, what failed

| System | Setting | Method | Scale / result | Transfer to Evolution |
|---|---|---|---|---|
| [DouZero](https://arxiv.org/abs/2106.06135) | 3p DouDizhu, huge variable action set, sparse terminal reward | Deep Monte-Carlo Q, action encoding, parallel actors | 45 actors, 4 GPUs, 30 days; beats CQN in half a day, supervised agent in 2 days, DeltaDou in 10 days; A3C/DQN reported to fail | Strongest direct evidence for DMC + candidate/action encoding in a card game. Our prefix factorization already reduces per-decision `K`. |
| [Suphx](https://arxiv.org/abs/2003.13590) | 4p Riichi mahjong, hidden info | Policy gradient + entropy regularization + global reward prediction + oracle guiding | 1.5M games per agent, 44 GPUs, 2 days per agent; above 99.99% of ranked humans | PG works in 4p hidden-info games. Sparse reward needed an auxiliary predictor; we can use a value baseline instead of reward shaping. Oracle guiding supports a training-only privileged critic. |
| [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) | StarCraft II, huge compound action space | League training, main agents + exploiters, autoregressive action policy, imitation pretraining | Grandmaster, top 0.2% | League/exploiters address self-play forgetting/cycles. Autoregressive action decomposition supports our prefix policy. We lack human logs, so no imitation stage. |
| [DeepNash](https://arxiv.org/abs/2206.15378) | 2p Stratego, imperfect info, huge action space | Model-free R-NaD, self-play, no search | 768 learner + 256 actor TPU nodes; 7.21M steps; batch 768 full games; top-3 Gravon | Model-free self-play can work in 2p imperfect info, but compute scale is far beyond one 4080. R-NaD is a fallback if 2p self-play cycles. |
| [Pluribus](https://www.science.org/doi/10.1126/science.aay2400) | 6p no-limit hold’em | MCCFR blueprint + depth-limited search | 8 days on 64-core server, 12,400 CPU core hours, <512 GB RAM; superhuman | Multi-player self-play + search works, but requires abstraction and search. Full-game Evolution CFR is not practical. |
| [AlphaZero](https://arxiv.org/abs/1712.01815), [MuZero](https://arxiv.org/abs/1911.08265) | Chess, shogi, Go; Atari | MCTS + learned policy/value; MuZero learns model | Superhuman in 24 h from rules; MuZero matches AlphaZero without rules | Perfect-information search assumptions do not hold. Hidden information needs information-set search. Engine is already known; no need to learn dynamics. |
| [ReBeL](https://arxiv.org/abs/2007.13544), [Student of Games](https://arxiv.org/abs/2112.03178) | 2p poker; chess/Go/HUNL/Scotland Yard | Search + self-play + game-theoretic reasoning | ReBeL provably converges in 2p zero-sum; SoG handles both perfect and imperfect info | Search cost is prohibitive here, especially 8p. ReBeL is 2p only. |
| [NFSP](https://arxiv.org/abs/1603.01121) | 2p Leduc/Limit Hold’em | Fictitious self-play + deep RL | Approached Nash in Leduc; common RL diverged | Principled 2p fallback if PPO self-play is exploitable; sample-hungry and 2p-specific. |
| [PSRO](https://arxiv.org/abs/1711.00832) | General-sum multi-agent | Population + empirical game meta-solver + approximate best responses | InRL overfits to training opponents; PSRO improves generalization | Correct wrapper for 3–8p and cycling. Best responses are expensive; use after a base agent exists. |
| [MAPPO](https://arxiv.org/abs/2103.01955) | Cooperative MARL, incl. Hanabi | PPO with centralized value | Competitive/superior to off-policy methods with minimal tuning | Supports PPO as a strong baseline; our game is competitive/constant-sum, so centralized value is optional. |
| [Other-Play](https://arxiv.org/abs/2003.02979) | Hanabi | Self-play + symmetry-aware update | Naive self-play learns conventions that do not transfer to new partners; other-play improves zero-shot coordination | Relevant only if cooperative traits/human partners matter. Evolution is mostly competitive. |
| [Implementation Matters](https://arxiv.org/abs/2005.12729), [What Matters in On-Policy RL](https://arxiv.org/abs/2006.05990) | Policy-gradient implementations | Ablations of code-level choices | Code-level optimizations account for most PPO gain over TRPO; batch size, normalization, clipping details matter | Budget engineering time for PPO details, not only the loss formula. |

What actually transferred:
- Self-play from scratch; action/candidate encoding; value baselines; entropy regularization; snapshot/league opponents; autoregressive action factorization; reward/advantage normalization.

What failed or did not transfer directly:
- A3C/DQN in DouDizhu’s full combinatorial action space. Our prefix factorization avoids that exact failure mode.
- Naive self-play in Hanabi: conventions do not transfer; league/pool helps.
- MCTS in imperfect information without information-set reasoning; Pluribus shows the rock-paper-scissors failure of fixed leaf values.
- AlphaZero/MuZero assumptions: perfect information, deterministic simulator, search budget.
- Full-game CFR at 8p: state space and compute.

---

## 4. Credit assignment for prefix decisions

### 4.1 Joint action probability

For a compound action with prefix fields `a_1..a_L`:

```text
log π(a_1..a_L | s) = Σ_l log π(a_l | s, a_<l)
```

The policy is autoregressive over fields. Sum prefix log-probs. This is exact if each field is sampled from the current policy; no approximation.

### 4.2 Return

Terminal-only:

```text
R_i = +1                 if i is winner
R_i = -1 / (N - 1)       otherwise, N > 1
R_i = 0                  on truncation
```

All prefix decisions of the same semantic action receive the same terminal return. `gamma = 1`. No discounting across prefix fields: discounting would make early fields depend on prefix position, which is artificial.

### 4.3 No GAE

With terminal-only reward and `gamma = 1`:
- GAE(λ=1) telescopes to `R - V(s)`.
- GAE(λ<1) reweights the same terminal return through value differences; it adds no new reward information, only bias/variance trade-off from the learned value.

Recommendation: use **Monte-Carlo advantage** `A = R - V(s_prefix)`. Do not implement GAE initially. If intermediate rewards are ever added (they are not in the plan), revisit. ([GAE](https://arxiv.org/abs/1506.02438))

### 4.4 Value baseline at prefix level

Train `V(s_prefix)` on the terminal return for that agent. This is a valid control variate: it depends on state/prefix, not on the chosen next field. It reduces variance for early prefix fields.

Current model uses `prefix_mean` in `_context` (`train_smoke.py:500`), which loses prefix order and length. For a stronger agent, encode the prefix as an ordered sequence (small GRU/Transformer or ordered set encoder), not a mean. `[INFERENCE]`

### 4.5 PPO ratio per prefix vs per compound action

Two valid choices:
- Per-prefix PPO: each prefix decision is a timestep; clip ratio per field.
- Joint PPO: compute product of prefix probabilities and clip the compound ratio once.

Recommend per-prefix for simplicity and compatibility with AEC. Note that clipping each field independently is not identical to clipping the joint action; with small updates this is acceptable. Optionally aggregate the joint ratio as a refinement. `[INFERENCE]`

### 4.6 Normalization

- Advantage normalization per batch: `(A - mean(A)) / (std(A) + 1e-8)`.
- For `N > 2`, either normalize advantages or scale training rewards by `(N-1)` so winner `N-1`, losers `-1`; keep environment reward semantics unchanged.
- Normalize entropy reporting by `log(K)`; raw entropy is not comparable across decisions with different `K`.
- Value loss: MSE on terminal return. For `N=8`, targets are near 0; advantage normalization is more important than value scaling.

---

## 5. Ranked training recipes

### 5.1 Rank 1 — PPO-clipped + candidate scorer + snapshot self-play

**Algorithm**

- Behavior policy: current shared policy, sampled stochastically.
- Collect `B = 128–256` base2p episodes per iteration (≈14k–28k decisions at 111 decisions/episode).
- Store per decision:
  - raw observation (for re-encoding), chosen index, old log-prob, valid mask, agent/seat, episode id, terminal return.
- `2–4` PPO epochs; minibatch `1024` decisions.
- Re-encode raw observations each epoch, or freeze the encoder. Do **not** reuse encoder outputs across epochs if the encoder weights change: the PPO ratio would be invalid. `[INFERENCE]`
- Loss:

```text
ratio = exp(log π_new(a|s,prefix) - log π_old(a|s,prefix))
A = normalize(R - V(s,prefix))
L_pi = -mean( min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) )
L_v  = mean( (V(s,prefix) - R)^2 )
L    = L_pi + 0.5 * L_v - β * mean(H)
```

- `ε = 0.2`, `β = 0.01–0.05`, Adam `3e-4`, grad clip `1.0`.
- `gamma = 1`; no GAE; no reward shaping.
- Shared weights across seats; seat feature already in `global.seat`.
- Masked logits with `-1e30` for illegal choices; softmax over current `K`.

**Self-play**

- Phase 1: pure mirror self-play, both seats current policy; train all decisions.
- Phase 2: 50% mirror, 50% snapshot pool; in pool episodes, learner seat uses current policy, opponent seat sampled from pool; train only learner decisions; alternate learner seat.
- Phase 3: league with exploiters if cross-play shows cycles (see §6).
- Snapshot pool: last 10–20 checkpoints; sample with PFSP-style weights `(1 - win_rate)^p`, `p=2`, plus latest. `[INFERENCE]`

**Data**

- Pilot: 50k base2p episodes.
- Main: 200k–1M base2p episodes (22M–111M decisions).
- Addons: 20k–100k episodes per addon, fine-tuned from the base checkpoint.
- 8p: 10k–50k fine-tune episodes after engine optimization; not from scratch. `[INFERENCE]`

**Expected wall-clock on this box**

Measured base: 0.7 s/episode single worker. Assume 12 env workers, 70% scaling efficiency → ≈12 episodes/s.

| Workload | Episodes | Data generation | PPO overhead | Total |
|---|---:|---:|---:|---:|
| base2p pilot | 50k | ≈1.2 h | +20–50% | ≈1.5–2.5 h |
| base2p main | 200k | ≈4.6 h | +20–50% | ≈6–9 h |
| base2p main | 1M | ≈23 h | +20–50% | ≈28–40 h |
| 2p all-addons | 100k | ≈9–23 h (5–10× base) | +20–50% | ≈12–30 h |
| 8p all-addons | 10k | ≈30 h (90 s/episode, 12 workers) | small | ≈30–35 h |
| 8p all-addons | 100k | ≈12.5 days | small | not practical from scratch |

All numbers except 0.7 s and 90 s are `[INFERENCE]`. 8p all-addons is the hard constraint: 90 s/episode single worker, 12 workers ≈0.09–0.13 episodes/s.

**Stability risks**

- On-policy sample inefficiency; worst at 8p.
- Stale encoder if PPO caches encoder outputs; re-encode or freeze.
- Entropy collapse; monitor mean entropy / `log K`.
- Value overfit or poor calibration; monitor explained variance.
- Nonstationary opponent; snapshot pool/PFSP.
- Reward scale for `N > 2`; normalize advantages.
- Truncation; exclude or treat as 0, but monitor.
- Engine errors; fix before training.

**What to measure**

- Win rate vs random, survival baseline, previous snapshots.
- Cross-play matrix / Elo; for 2p, approximate best-response win rate.
- Policy loss, value loss, entropy, clip fraction, KL, grad norm, value explained variance.
- Truncation rate, invalid attempts, engine errors, natural terminal rate.
- Episodes/s, decisions/s, bytes/step, RSS/env, GPU utilization.

---

### 5.2 Rank 2 — Deep Monte-Carlo Q (DouZero-style) + replay + snapshot self-play

**Algorithm**

- Replace policy logits with `Q(s, choice)` per candidate using the same candidate scorer.
- Behavior policy: ε-greedy over legal choices at each prefix; `ε` from 0.1 → 0.01.
- Target: full-episode Monte-Carlo return `R` for each prefix decision (DouZero DMC). No bootstrapping means no max-over-actions overestimation, but it also means policy-evaluation of the behavior policy, not direct Q-learning. `[INFERENCE]`
- Loss: `mean((Q(s,prefix) - R)^2)`.
- Target network updated every 1k–10k steps.
- Shared Q across seats; seat/player-count features; per-agent return perspective.
- Replay buffer: store compact fixed-size tensors, not Python dicts. Full candidate tensors are large:
  - context 320 floats + `K × 256` floats; for `K ≈ 20`, ≈5k floats ≈20 KB/decision.
  - 1M decisions ≈20 GB; 30 GB RAM is tight. Use 0.5–2M decisions, preallocated `float32`, or store compact features and recompute candidate encodings. `[INFERENCE]`
- Alternative: n-step returns with target network; risk overestimation. Prefer full-episode MC initially.

**Data**

- 50k–200k base2p episodes; replay 0.5–2M decisions; 4–8 passes.
- Addons: 10k–50k episodes per addon; replay reuse helps.
- 8p: 5k–20k episodes, replay 0.5–1M; engine-limited.

**Expected wall-clock**

- Data generation same as PPO: 100k base2p ≈2.5 h at 12 episodes/s.
- Replay training: 0.5–2M decisions × 4–8 passes; batch 256; maybe 1–4 h on GPU/CPU for the tiny network. Total ≈4–8 h for 100k base2p. `[INFERENCE]`
- 8p 10k episodes ≈30 h data generation; training small; replay memory tight.

**Stability risks**

- Replay memory blow-up from variable entity sets.
- Off-policy self-play: Q target is behavior-policy return; greedy improvement can cycle.
- Truncation: DMC target 0 for unfinished episodes biases Q toward 0; exclude truncated episodes or handle separately.
- Partial observability: `Q(observation, action)` is not Markov; use entity/history encoder.
- Deterministic greedy policy can be exploitable; evaluate stochastic/ε-greedy policy too.
- Exploration collapse; monitor action entropy and replay coverage.

**What to measure**

- TD/MC error, Q mean/std, replay size, ε, target-update frequency.
- Win rate vs baselines/snapshots; cross-play.
- Memory, samples/s, GPU utilization.
- Truncation and engine-error rates.

---

### 5.3 Rank 3 — Fixed REINFORCE + value baseline (current smoke, corrected)

**Algorithm**

- Current `train_smoke.py` structure.
- Fix policy loss to mean over decisions; entropy consistent; advantage normalization; snapshot pool.
- One gradient pass per episode batch; no clipping.
- Good as an ablation and correctness baseline; high variance and slow.

**Data / wall-clock**

- 100k–1M base2p episodes; similar data generation; training overhead lower than PPO.
- 100k ≈4–8 h; 1M ≈1–2 days with 12 workers. `[INFERENCE]`

**Risks**

- High variance; no trust region; entropy collapse; poor sample efficiency.
- Not recommended as final algorithm.

**What to measure**

- Return variance, gradient norm, entropy, win rate vs baselines.

---

### 5.4 Rank 4 — IMPALA / V-trace asynchronous actor-learner

**Algorithm**

- Many CPU actors (12–16 env workers) generate episodes; one learner on GPU/CPU updates continuously.
- V-trace corrects for lagged behavior policy ([IMPALA](https://arxiv.org/abs/1802.01561)).
- Policy/value heads same as PPO; no clipping; V-trace importance weights.
- Advantages: high throughput with one GPU and many CPU actors; handles staleness.
- Costs: more complex; V-trace hyperparameters (`rho_bar`, `c_bar`); reproducibility harder; actor/learner queue latency.

**When to use**

…

…

…
**Recommended next action:** implement PPO + MC advantage + snapshot self-play on base2p; run 50k episodes; gate on the metrics in §8; only then scale.

[Showing lines 1-300 of 581. Use :301 to continue]
