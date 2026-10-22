# Evaluation plan for Evolution RL

## 1. Bottom line

- **Current evaluation cannot measure strength.** `train_smoke.evaluate` (`rl/python/evolution_rl/train_smoke.py:802-822`) plays 8 greedy **self-play** games against the same weights and reports `reward_player_0`. In 2p that reward is ±1, so 8 games give a 95% interval roughly ±0.30 at 50% (5/8 → Wilson `[0.31, 0.86]`). It is a reload/reproducibility check, not an eval.
- **Primary metric:** expected terminal payoff vs a frozen opponent, with balanced seats. Zero-sum by construction (`rl/node/environment.js:519-535`: winner +1, other playing seats −1/(N−1); a seat that leaves gets 0, but the RL env never dispatches `gamePlayerLeft`). Win rate is secondary (2p: payoff = 2·winrate − 1).
- **Spend games where they are cheap.** 2p base costs ~0.7 s/episode; 800 games on 16 workers ≈ 35 s. 8p all-addons costs ~90 s/episode; 800 games ≈ 75 min. Power for 2p; only non-inferiority for 8p.
- **Never compare architectures by one checkpoint scored on the test seeds.** Split dev/test/confirmation seed blocks, pre-register the estimand and N, use SPRT only as a gate, and always confirm on fresh seeds.
- **Do not treat decisions as samples.** Cluster at the episode/seed level; per-decision `attempt_rejected`/entropy counts are correlated within an episode.

## 2. What the harness already gives (evidence)

| Component | Produces | Gap |
|---|---|---|
| `rollout.play_episode` | `decisions`, `prefix_choices`, `committed`, `attempts_rejected`, `terminated`, `truncated`, `truncation_reason`, `winner`, per-seat `rewards`, `observation_bytes`, `duration_seconds` | no CI, no Elo, no seat rotation, no pairing |
| `rollout.summarize` | counts, truncation reasons, `reward_sum`, failures | no win rate, no per-config/per-seat breakdown, no uncertainty |
| `benchmark.measure_config` | wall, reset median/p95, committed actions/s, prefix choices/s, avg choices/step, avg select s, JSON bytes/step, RSS, terminals/truncations | throughput only; no p50/p95 step latency |
| `train_smoke.evaluate` | 8 greedy self-play games, `reward_player_0`, actions | no opponent, no CI, no sampled-policy eval |
| `matrix_specs` | 128 addon subsets × players 2..8 × seeds × policies | a correctness soak, not a strength design |

The engine also has `snapshot()/restore()/exportReplay()` (`environment.js:917-1005`), seeded Mulberry32 runtime, and deterministic command replay. Those are the levers for paired evaluation.

## 3. Units, seeds, pairing

**Unit of analysis.** One episode = (config, seed, seat assignment, opponent pair). Aggregate over episodes. Decisions are not independent (an episode shares one deck, one RNG stream, one policy trajectory). For per-decision metrics (latency, rejection rate, entropy) report episode-level means and cluster-bootstrap over seeds.

**Two different seeds.** Do not conflate:
- *training seeds* — replicate the learning algorithm; the unit for architecture/algorithm claims;
- *evaluation seeds* — replicate decks and engine randomness for a fixed checkpoint; the unit for match claims.

**Seed pools** (pre-registered, hash-derived so they cannot collide by accident, e.g. `seed = SipHash(base, config, index) mod 2^32`):
- train: `0 .. 99_999`
- dev/model-selection: `20_000_000 .. 20_099_999`
- test: `30_000_000 .. 30_099_999`
- confirmation: `40_000_000 .. 40_099_999`

Current smoke already avoids overlap (train 7..38, eval 10000..10007, `train_smoke.py:1155,1190`), but 8 eval seeds are unusable for decisions.

**Seat rotation.** With shared weights, A-vs-B means two checkpoints. For 2p alternate seats every game (A=player_0 on even index). For N players assign A to a balanced set of seats; rotate through the cyclic shifts (or all $\binom{N}{k}$ assignments if $k$ is not 1 or N−1). Report the per-seat breakdown; a seat effect is a rule bug or a policy bug.

**Pairing.**
- Run A and B on the same evaluation seed list.
- With deterministic/greedy policies, same seed ⇒ same initial shuffle and same RNG state until the trajectories diverge. Deck order is shared; engine draws after divergence are not. Measure the realized correlation $\rho$ between paired outcomes; only claim variance reduction if $\rho>0$.
- Use McNemar for paired binary outcomes (same seed, A vs B), and a paired bootstrap over seeds for payoffs. Report discordant counts, not just the p-value.
- **[INFERENCE]** For exact common random numbers, the engine would need a "randomness tape": record the ordered RNG draws of a reference episode and replay them for the other arm (or use counter-based streams keyed by `(purpose, occurrence)`). The current runtime is a single Mulberry32 stream (`rl/node/runtime.js`), so any difference in the number of draws desynchronizes the rest of the episode. `snapshot/restore` gives exact pairing only for one-step counterfactuals ("root-paired" comparisons), not for whole games.

**Greedy vs sampled.** `evaluate` uses `greedy=True` (`train_smoke.py:809`). Greedy is low-variance but can hide a policy that only works when sampled (and vice versa). Report both: primary = the deployment mode, secondary = the other. Greedy self-play is also a mirror match, so expected payoff is 0 by symmetry; use it as a sanity check, not a strength score.

## 4. Match matrix

| Tier | Purpose | Config / opponents | Games per cell | Seeds | Est. wall, 16 workers |
|---|---|---|---|---|---|
| T0 fixtures | correctness regression | scripted YAML scenarios (gradient fixture, ambush, regeneration, Anguish+Communication) | 1 | fixed | seconds |
| T1 sanity | harness/seat/RNG check | base 2p: mirror, random, survival | 200 each | dev | ~1 min |
| T2 core strength | **primary decision** | base 2p, challenger vs frozen champion | 800 test + 800 confirmation | test, confirmation | ~1 min each |
| T3 addon/seat coverage | non-inferiority | base 4p, base 8p, all-addons 2p/4p/8p vs champion | 400 | test | 2 min – 1.5 h |
| T4 league | multi-way ranking | 6–8 frozen members, round robin | 200 per pair | dev | 2 min (2p) – 5 h (all 8p) |
| T5 soak | rules matrix, no strength claims | 128 addon subsets × 7 player counts × 20 seeds × 2 baselines = 35,840 | 1 | fixed | ~5–10 h **[INFERENCE]** |
| T6 exploitability | diagnostics on micro-games | 2p tiny scenarios, exact best response | exhaustive | n/a | seconds–minutes |

T5 is the plan's required rules sweep; it answers "does the engine crash / produce invalid leaves / leak IDs", not "is model A better". Do not report a pooled win rate across 128 configurations as a strength result: it is a different estimand and the config mix is arbitrary.

**Opponent ladder.** random → survival → frozen champion → historical checkpoints (league) → mirror. Always include random and survival in T1/T2; a trained model that fails to beat `random` has no measurable signal, and a model that beats the champion but loses to `random` is a sign of a mask/protocol bug.

## 5. Power: episodes vs detectable effect

Fixed-sample, two-sided $\alpha=0.05$, power $0.8$ ($z_{1-\alpha/2}+z_{1-\beta}=2.8016$):

$$N=\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,p(1-p)}{\delta^2}\quad\text{(head-to-head vs }p_0=0.5\text{)}$$

Two independent arms evaluated against a third opponent need $2\times$ per arm. Paired games reduce variance by $(1-\rho)$: $N_{\text{paired}}=N(1-\rho)$.

**2p, base (per-episode cost ~0.7 s):**

| $\delta$ win rate | Elo | N (head-to-head) | wall @16 workers |
|---|---|---|---|
| 20 pp | 147 | 50 | ~2 s |
| 15 pp | 107 | 88 | ~4 s |
| 10 pp | 70 | 197 | ~9 s |
| 7.5 pp | 52 | 349 | ~15 s |
| **5 pp** | **35** | **785** | **~35 s** |
| 4 pp | 28 | 1,227 | ~54 s |
| 3 pp | 21 | 2,181 | ~95 s |
| 2 pp | 14 | 4,906 | ~3.6 min |
| 1 pp | 7 | 19,623 | ~14 min |

Paired effect (5 pp): $\rho=0.2$ → 628; $\rho=0.5$ → 393; $\rho=0.8$ → 157; $\rho=0.95$ → 40.

**8p, all-addons (cost ~90 s; payoff variance at equal skill is $1/(N-1)=1/7$):**

$$N=\frac{7.849\,\sigma^2}{\delta^2},\qquad \sigma^2=1/7\ \text{(equal skill)}$$

| $\delta$ mean payoff | ≈ win-rate pp | N | wall @16 workers |
|---|---|---|---|
| 0.10 | 8.75 | 112 | ~11 min |
| 0.05 | 4.4 | 449 | ~42 min |
| 0.02 | 1.75 | 2,804 | ~4.4 h |
| 0.01 | 0.9 | 11,213 | ~17.5 h |

Conclusion: 8p is a **non-inferiority** regime. Set the margin explicitly (e.g., lose no more than 0.05 payoff ≈ 4.4 pp) and buy 449 games, not 20 seeds.

**Across training seeds.** If $\sigma_s$ is the between-training-seed SD of the final score, detecting a mean difference $\delta$ needs

$$N_{\text{seeds}}=\frac{2\,(2.8016)^2\,\sigma_s^2}{\delta^2}.$$

If $\sigma_s=0.05$ payoff, 5 seeds resolve only $\delta\approx0.089$; if $\sigma_s=0.1$, only $\delta\approx0.177$. **Most architecture effects will be invisible at 5 seeds.** Measure $\sigma_s$ in a 3-seed pilot before committing; otherwise report "inconclusive".

## 6. CI and test methods

| Quantity | Method | Notes |
|---|---|---|
| win rate, one match | **Wilson score interval** | correct near 0/1 and at small N; do not use Wald/normal for N<200 |
| win rate, very small N or a hard guarantee | Clopper–Pearson | conservative |
| mean payoff | t-interval (N≥30) or **bootstrap** on episode payoffs | ±1 rewards make the t-test nearly exact |
| paired payoff | **paired bootstrap over seeds**, or McNemar for binary | report discordant pairs and $\rho$ |
| per-decision metrics | cluster bootstrap **by seed** | decisions within an episode are not independent |
| Elo | BT fit + bootstrap refit, or Bayesian posterior | delta method: $\mathrm{SE}(\Delta)=\frac{400}{\ln 10}\frac{\mathrm{SE}(p)}{p(1-p)}$; 400 games at 50% → ±~35 Elo (2 SD), matching KataGo's 239/400 → 69 ± 36 |
| across training seeds | **stratified bootstrap / IQM** (rliable) | 3–10 runs is the stated practical range; report all runs |
| sequential gate | **SPRT / GSPRT** | see below |
| many configs | Holm–Bonferroni, or a pre-registered primary + exploratory secondaries | never test 128 configs and report the best |

**SPRT.** Wald's test: $\Lambda_n=\sum_k \log\frac{f_1(x_k)}{f_0(x_k)}$; accept H1 at $\log\frac{1-\beta}{\alpha}$, accept H0 at $\log\frac{\beta}{1-\alpha}$; $\alpha=\beta=0.05$. Use it only for a **promotion gate**, with hypotheses in payoff units (e.g., H0: $\Delta\le0$, H1: $\Delta\ge0.05$ ≈ 2.5 pp at 2p). Because Evolution has no draws, the chess "pentanomial" model collapses to a 3-outcome pair score (0, 1, 2 wins for A over a seat-swapped pair); the pairing variance reduction still applies. Use the generalized LLR with an MLE over the pair-score distribution, and correct for discrete-time overshoot (Siegmund).

Three rules:
1. **Do not peek outside the SPRT.** Monitoring a fixed-N test and stopping when $p<0.05$ inflates the type-I error.
2. **Do not report the SPRT stopping estimate as the effect size.** Truncated sampling biases it upward; report a fixed-N estimate from the confirmation block.
3. **Do not apply SPRT per addon config.** Use it for the one pre-registered primary comparison.

**Elo / Bradley–Terry.** Fit $P(i\succ j)=\gamma_i/(\gamma_i+\gamma_j)$, $\theta_i=\log\gamma_i$, with an L2 prior or BayesElo-style prior; anchor a fixed member (random = 0) so ratings are comparable across time. For N>2, the engine only exposes the winner, so either (a) derive pairwise "winner beats each loser" comparisons and fit BT, or (b) fit **Plackett–Luce** over partial rankings (winner first, rest tied). BT/Elo assumes transitivity; check it (see §7) and report the pairwise matrix alongside ratings. A league of 6 members with 200 games/pair gives roughly ±20–25 Elo per member at 2p; 400/pair gives ~±15.

**Exploitability.** For 2p zero-sum, $\text{exploitability}=(v_1(\sigma_2,BR_1)+v_2(BR_2,\sigma_1))/2$; the multi-player generalization is NashConv (sum of per-player improvements). Exact best response requires walking the infoset tree, which is intractable in the full game. Restrict to micro-scenarios (e.g., the 2-animal feeding fixture) where the infoset tree is enumerable; report it as a **diagnostic**, never a gate. In multiplayer, report the vector or the mean; a single scalar can hide a seat that is exploitable.

## 7. Metrics

**Primary**
- `payoff_mean` = mean terminal reward for the evaluated policy over balanced seats; 95% CI; the training objective, defined for every N.
- `win_rate` = fraction of games with `winner == seat`; Wilson CI; report per seat.
- `elo_delta` vs the frozen champion (or league rating); CI by bootstrap.

**Secondary (report, but do not promote on them alone)**
- `natural_terminal_rate` = terminated / (terminated + truncated); validity metric.
- `truncation_rate` by `truncation_reason` (mostly `max_decisions`); a policy that loops is not "safe", it is unmeasured.
- `attempt_rejected_rate` = Σ`attempt_rejected` / Σ`decisions`; a mask/enumerator health signal. Any rejection with `preparationPending=false` is an engine invariant violation, not a policy weakness.
- `committed_rate`, `decisions_per_episode`, `prefix_choices_per_episode`: cost and behaviour.
- `step_latency_p50/p95` split into engine+transport and policy forward; `committed_actions_per_second`, `prefix_choices_per_second`, `json_bytes_per_step`, RSS/env (already in `benchmark.py`; add p50/p95).
- `entropy_norm` = mean $H(\pi)/\log K$ over decisions with $K>1$, plus entropy of the first (type) decision. Must be compared **only at the same decision distribution**; different policies visit different states, so entropy is a diagnostic, not a cross-policy score.
- `value_brier` / `value_ece` for 2p, measured **on-policy sampled** rollouts (the value head is trained on the behaviour distribution; greedy evaluation is off-policy and will look miscalibrated). Brier baseline at $v=0$ is 1.0 for ±1 targets.
- `grad_norm_p95`, NaN/non-finite count, loss IQR across seeds: training health only.

**Learning-curve metrics** (for architecture choice)
- Evaluate a checkpoint every fixed number of **environment episodes** (not wall-clock) on a 100–300 game dev mini-suite.
- `auc_return` (area under the dev payoff curve) and `episodes_to_threshold`.
- Report both equal-episodes and equal-wall-clock curves; the primary axis should be episodes, because the engine/transport dominates wall-clock and architecture compute is small relative to it. **[INFERENCE]** measure this with `benchmark` before assuming it.

## 8. A/B protocol for architectures / training methods

Controls that must be identical: engine commit, `schema_version`, catalog hash, observation whitelist, env config, seat rotation, dev/test seed blocks, evaluation mode, tuning budget, checkpoint-selection rule.

Suggested arms for the starter comparison:
- A: current candidate-scoring MLP (mean-pool, 16-d embeddings, 64 hidden).
- B: permutation-invariant DeepSets/attention encoder over the same entity features.
- C: transformer over entity tokens (same features, more parameters).
- D: same MLP + a different objective (e.g., clipped/PPO-style or search-augmented targets).

Protocol:
1. **Pilot** 3 seeds × 2k episodes per arm; estimate between-seed SD and per-episode cost. This sets the real MDE.
2. **Tune** each arm with the same budget (e.g., 8 random trials at 25% episodes); log every trial. Do not give one arm more tuning.
3. **Train** 5 seeds × a fixed episode budget (e.g., 20k base2p; ~3.9 h/seed, ~4 h wall for 5 seeds in parallel). Save every checkpoint.
4. **Dev screen**: every 2k episodes, 100–300 games vs random+survival+champion on dev seeds. Used for model selection only.
5. **Test**: 800 games vs champion on test seeds (primary). Compute paired difference, Wilson/t/bootstrap CIs, and Elo.
6. **Confirm**: pre-registered 800 games on confirmation seeds, fresh run of the same checkpoint. Promotion requires the pooled lower bound > 0, not just the test block.
7. **League**: add the winner to the round-robin, refit BT, check transitivity. A challenger that beats the champion but loses to a third member is a game-theoretic red flag, not a promotion.

Decision rule for an **architecture** (distribution over training seeds):
- Report per-seed results and IQM with a stratified bootstrap CI over (training seed, eval seed).
- Claim a difference only if the CI excludes 0 **and** the effect exceeds the pre-registered MDE.
- With 5 seeds, a sign test is significant only at 5/5 (one-sided $p=0.031$); 4/5 is $p=0.19$. Do not dress up 4/5 as evidence.
Decision rule for a **checkpoint** (deployment):
- Lower 95% CI of payoff difference vs champion > 0 on test **and** confirmation blocks, and all hard gates pass.

What not to claim: a single training seed's final checkpoint is not evidence about the architecture; a training-loss improvement is not a strength improvement; a win-rate gain on the same seeds used for tuning is not a gain.

## 9. Promotion criteria

Hard gates (any failure = reject, regardless of win rate):
1. Zero `ENGINE_INVARIANT` / `ENGINE_STALLED` / protocol errors across the full match.
2. Zero `attempt_rejected` with `preparationPending=false`; rejection rate no worse than champion +0.5 pp.
3. Truncation rate no worse than champion +2 pp; natural-terminal rate no worse than champion −1 pp on base 2p.
4. Step latency p95 ≤ 1.5× champion; throughput ≥ 0.7× champion (same worker count).
5. Same `schema_version`/catalog; observation whitelist unchanged (privacy tests still pass).
6. Beats `random` and `survival` on the sanity suite; mirror match is consistent with 50% (2p) within CI.

Statistical gates:
7. Primary: lower 95% CI of payoff difference vs champion > 0 on the test block, **and** the confirmation block does not reverse it (pooled lower bound > 0).
8. Addon/seat coverage: no config regresses by more than the non-inferiority margin (e.g., −0.05 payoff) after Holm correction across the pre-registered T3 cells.
9. Effect size ≥ the pre-registered MDE. If the CI includes 0 and the CI upper bound is below the MDE, declare **inconclusive**, not "no difference"; extend only to the pre-registered cap (e.g., 4,000 games), then stop.

Optional diagnostic: league Elo gain ≥ 20 Elo with a CI excluding 0; exploitability not worse on micro-scenarios.

## 10. Results-table template

**Match table**

| match_id | engine/schema | config (addons, N) | opponent | seeds | games | A payoff | 95% CI | win rate | Wilson 95% | Elo Δ | 95% CI | trunc % | reject/dec | nat-term % | s/ep | p95 step ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base2p-A-vs-champ-test` | `<ver>/1` | `[], 2` | `champ@<hash>` | `3.0e7..3.0e7+799` | 800 | | | | | | | | | | | |
| `base2p-A-vs-champ-conf` | | | | `4.0e7..` | 800 | | | | | | | | | | | |
| `all8p-A-vs-champ` | | `[all], 8` | | | 449 | | | | | | | | | | | |

**Training table**

| arm | training seed | episodes | wall | final train return | dev IQM | dev 95% CI | test payoff | test 95% CI | grad-norm p95 | NaN | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (MLP) | 0..4 | 20k | | | | | | | | | |
| B (DeepSets) | 0..4 | 20k | | | | | | | | | |

**League table**

| member | anchor | Elo | 95% CI | games | wins | losses | notes |
|---|---|---|---|---|---|---|---|
| random | 0 | 0 | — | | | | fixed anchor |
| survival | | | | | | | |
| champ@hash | | | | | | | |
| challenger | | | | | | | |

**Pre-registration block** (must be committed before the run): engine/schema/catalog hashes, config, seed blocks, N, primary metric, MDE, stopping rule, promotion thresholds, number of arms, tuning budget, and the list of secondary metrics. Anything not in the block is exploratory.

## 11. Threats to validity

- **Test-set leakage / best-checkpoint selection.** Choosing the best of many checkpoints on the test seeds and reporting that score is optimistically biased. Use dev for selection, test+confirmation for reporting; pre-register the checkpoint rule (last, or best-dev at fixed intervals).
- **Seed reuse and non-independence.** Same seed across arms is good (pairing) only if you model it; treating 800 games as 800 independent units when they share 20 seeds understates the CI. Cluster-bootstrap by seed; with <20 clusters, bootstrap is unreliable — use t-intervals or exact methods, and say so.
- **Self-play mirror is not strength.** Shared weights on both seats guarantee a symmetric expected outcome; the current `evaluate` cannot distinguish a strong model from a weak one.
- **Greedy-only evaluation.** A policy can be strong greedily and brittle when sampled (or the reverse). Report both; choose the deployment mode as primary.
- **SPRT bias.** Stopping-time estimates are biased; confirmation block required.
- **Elo non-transitivity.** A beats B, B beats C, C beats A is possible in this game (matchup-dependent traits). Check with mElo/Nash averaging/α-Rank and report the pairwise matrix; a single Elo ordering can be wrong.
- **Multiplayer payoff is coarse.** One winner per 8p game; per-episode payoff is ±1/−1/7. Do not over-interpret 8p ratings from a few hundred games.
- **Truncation handling.** Treating `max_decisions` truncation as a 0 payoff mixes "weak" with "did not finish". Report both intention-to-treat and completers-only, and gate on truncation rate.
- **Engine-version drift.** Changes to rules, RNG consumption, or observation schema invalidate cross-version comparisons. Store `engine_version`, `schema_version`, catalog hash, and replay hash with every result.
- **P-hacking across 128 configs.** The matrix is a correctness sweep. Pre-register the handful of configs used for strength, and Holm-correct the rest.
- **Reward semantics.** The reward is zero-sum only because all seats remain `playing`; if a future mode allows leaving, recompute the metric. The winner tiebreak consumes RNG (`scoreRandom = getRandom()` in `games-rdx-server.js:518-531`), so deterministic policies still have seed-dependent outcomes.
- **Bootstrap with few units.** Percentile bootstrap CIs are too narrow at N<20; prefer Wilson/exact for binary data and report the method.

## Sources

1. Wald, *Sequential Tests of Statistical Hypotheses*, Ann. Math. Stat. 16(2), 1945 — https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-16/issue-2/Sequential-Tests-of-Statistical-Hypotheses/10.1214/aoms/1177731118.full
2. Stockfish Fishtest mathematics (SPRT, elo model, pentanomial) — https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html
3. van den Bergh, pentanomial SPRT (paired games, GSPRT, overshoot correction) — https://github.com/vdbergh/pentanomial
4. Li, Liu, Ying, *Generalized SPRT for Separate Families of Hypotheses* — http://stat.columbia.edu/~jcliu/paper/GSPRT_SQA3.pdf
5. Elo, *The Rating of Chessplayers, Past and Present*, 1978.
6. Bradley & Terry, *Rank Analysis of Incomplete Block Designs*, Biometrika 39(3/4), 1952 — https://doi.org/10.1093/biomet/39.3-4.324
7. Hunter, *MM Algorithms for Generalized Bradley–Terry Models*, Ann. Stat. 32(1), 2004 — https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/MM-algorithms-for-generalized-Bradley-Terry-models/10.1214/aos/1079120141.full
8. Coulom, BayesElo / Whole-History Rating — https://www.remi-coulom.fr/Bayesian-Elo/ , https://www.remi-coulom.fr/whole-history-rating/
9. Balduzzi et al., *Re-evaluating Evaluation*, NeurIPS 2018 — https://arxiv.org/abs/1806.02643
10. Lanctot et al., *Evaluating Agents using Social Choice Theory*, 2023 — https://arxiv.org/abs/2312.03121
11. Yang et al., *α-Rank: Multi-Agent Evaluation by Evolution*, 2019 — https://arxiv.org/abs/1903.01373
12. Lanctot et al., *A Unified Game-Theoretic Approach to Multiagent RL* (NashConv), 2017 — https://arxiv.org/abs/1711.00832
13. OpenSpiel exploitability/best-response implementation — https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/exploitability.py ; OpenSpiel paper — https://arxiv.org/abs/1908.09453
14. Wu, *Accelerating Self-Play Learning in Go* (400-game matches, Elo ±2 SD, BayesElo) — https://arxiv.org/abs/1902.10565
15. Silver et al., *Mastering Chess and Shogi by Self-Play*, 2017 — https://arxiv.org/abs/1712.01815 ; AlphaGo Zero, Nature 2017 — https://doi.org/10.1038/nature24270
16. Henderson et al., *Deep Reinforcement Learning That Matters*, AAAI 2018 — https://arxiv.org/abs/1709.06560
17. Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (stratified bootstrap, IQM, performance profiles, 3–10 runs) — https://arxiv.org/abs/2108.13264
18. Colas et al., *A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms* — https://arxiv.org/abs/1904.06979
19. Dodge et al., *Show Your Work: Improved Reporting of Experimental Results* — https://arxiv.org/abs/1909.03004
20. Patterson et al., *Empirical Design in Reinforcement Learning*, 2023 — https://arxiv.org/abs/2304.01315
21. Wilson, *Probable Inference, the Law of Succession, and Statistical Inference*, JASA 22, 1927 — https://doi.org/10.1080/01621459.1927.10502953
22. Clopper & Pearson, Biometrika 26, 1934 — https://doi.org/10.1093/biomet/26.4.404
23. McNemar, *Note on the sampling error of the difference between correlated proportions*, Psychometrika 12, 1947 — https://doi.org/10.1007/BF02295996
24. Efron, *Bootstrap Methods: Another Look at the Jackknife*, Ann. Stat. 7(1), 1979 — https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full ; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993.
25. Cameron, Gelbach & Miller, *Bootstrap-Based Improvements for Inference with Clustered Errors*, Rev. Econ. Stat. 90(3), 2008 — https://doi.org/10.1162/rest.90.3.414
26. Holm, *A Simple Sequentially Rejective Multiple Test Procedure*, 1979 — https://doi.org/10.2307/4615733
27. Johari et al., *Peeking at A/B Tests: Why It Matters, and What to Do About It*, KDD 2017 — https://doi.org/10.1145/3097983.3097992
28. Plackett, *The Analysis of Permutations*, JRSS C 24, 1975 — https://doi.org/10.2307/2346567 ; Luce, *Individual Choice Behavior*, 1959.
29. Herbrich, Minka & Graepel, *TrueSkill: A Bayesian Skill Rating System*, NeurIPS 2007 — https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/
30. Salmon et al., *Parallel Random Numbers: As Easy as 1, 2, 3* (counter-based RNG for reproducible streams), SC 2011 — https://doi.org/10.1145/2063384.2063405
31. Siegmund, *Sequential Analysis: Tests and Confidence Intervals*, 1985 (SPRT overshoot / discrete-time correction).
32. Gelman & Loken, *The Statistical Crisis in Science*, American Scientist 102(6), 2014 (garden of forking paths) — https://www.americanscientist.org/article/the-statistical-crisis-in-science

No repository files were modified. All repo claims above are from `rl/python/evolution_rl/{train_smoke,rollout,benchmark,policies}.py`, `rl/node/{environment,runtime}.js`, and `server/reducers/games-rdx-server.js`; quantitative tables are derived from the stated formulas, with assumptions flagged inline.
