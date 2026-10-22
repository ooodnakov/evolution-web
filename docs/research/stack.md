## 1. Bottom line

**Adopt nothing wholesale. Keep the custom AEC + candidate-scoring stack; borrow TorchRL only as an optional buffer/distribution library at milestone M1.**

Reason: the env, spaces, prefix enumerator, transport and trainer already exist in-repo (~7.6k LOC, `rl/node/*.js`, `rl/python/evolution_rl/*.py`), the smoke run works, and *every* surveyed framework either refuses a per-decision `Discrete(len(choices))`/ragged observation, or supports it only by fixing a global action universe and padding to a cap — which the plan explicitly forbids. I verified the two most attractive integrations empirically; both fail today (section 2).

## 2. Measured integration probes (this session)

Env: torch 2.14.1+cpu, torchrl 0.14.1, tensordict 0.14.3, pettingzoo 1.27.0, gymnasium 1.4.0, stable-baselines3 2.9.0, sb3-contrib 2.9.0. Probe scripts in `/tmp` (not repo).

| Probe | Result |
|---|---|
| `PettingZooWrapper(env=evolution_rl.raw_env(), use_mask=True)` | Fails at construction: `ValueError: Sequence spaces must have the stack argument set to ``True``.` (`torchrl/envs/libs/gym.py:598`, from our `VariableSequence` in `spaces.py`) |
| Synthetic AEC env, `action_space(agent)` = `Discrete(2)` then `Discrete(3)` | Wrapper snapshots `CategoricalBox(n=2)` at construction; next step: `RuntimeError: The expanded size of the tensor (2) must match the existing size (3) at non-singleton dimension 0.` (`torchrl/envs/libs/pettingzoo.py:813`) |
| SB3 `PPO("MlpPolicy")` on `Discrete(2)`, then `env.action_space = Discrete(5)` | `model.action_space.n` stays 2, `policy.action_net.out_features` stays 2, 50 stochastic `predict()` calls return only {0,1}, `learn()` runs **without complaint** — silent wrongness |

Cause is in the wrapper source, not a config flag: `_make_group_specs` calls `self.action_space(agent)` once (`pettingzoo.py:367`), `_step_aec` converts actions through that frozen `input_spec`, and `_update_action_mask` assigns each step's mask into a tensor sized from construction. The doc also warns it is "tested using version == 1.24.3" while we run 1.27.0.

The AEC contract itself is fine with dynamic spaces: PettingZoo lists "action and observation spaces which can change over time, and differ per agent" ([AEC API](https://pettingzoo.farama.org/api/aec/)); our env keeps the static `action_spaces` dict and overrides the `action_space()` method, cached per `decisionId` (`aec.py:360-373`). That is the correct side of the contract; the wrappers are the problem.

## 3. Comparison

| Framework | Per-step dynamic `Discrete(n)` | Ragged entity observations | AEC | Self-play / shared weights | Offline replay of decisions | Maturity | Integration cost (est.) | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Custom (current)** | Native — `Discrete(len(choices))` per `decisionId` | Native — `Sequence`, pointer scoring | Native | Shared weights per seat, trivial | Episode lists in RAM; no disk buffer yet | Working, 7.6k LOC, smoke trains | 0 to keep; ~1–2 wk for PPO+GAE, ~2–3 wk league | **Keep** |
| **TorchRL 0.14** | No: spec fixed at wrapper construction (measured) | No via wrapper (`Sequence` requires `stack=True`); `TensorDict` supports `shape=-1` but `check_env_specs` fails when keys vary | Yes (`PettingZooWrapper`, `use_mask=True` compulsory) | Generic multi-agent + `MarlGroupMapType`; no league utility | `TensorDictReplayBuffer` + `SliceSampler` handle ragged **trajectory** length, not ragged feature width | 0.x, active, APIs churn | 1–2 d for buffer only; 2–4 wk to make env fit (caps/padding) | **Hybrid, buffer only** |
| **RLlib 2.59** | Only via `Discrete(max_avail_actions)` + `action_mask` + custom model; "not all algorithms support parametric actions" | Only via custom `RLModule`; specs must be fixed | Yes via `PettingZooEnv` wrapper (PettingZoo's AEC Hold'em tutorial) | Built-in multi-agent policies, self-play and league-based self-play claimed on the landing page | BC / CQL / MARWIL, Ray Data ingestion | Mature, large API churn | 3–6 wk + Ray ops; permanent caps | No |
| **SB3 / sb3-contrib 2.9** | No: `MaskablePPO` masks a **fixed** `n`; single-agent only ("SB3 is designed for single-agent RL") | No: PettingZoo wrapper "assumes that the action space and observation space is the same for each agent" | No (AEC only through a hand-written gym-shaped adapter) | Demo only, via vectorized self-play | Manual replay buffer; no dataset API | Very mature (single-agent) | Not viable without caps; 3–5 d with caps and hidden breakage | No |
| **CleanRL** | No (fixed head); explicitly "not a modular library … not meant to be imported" | No | Tutorials only, Parallel | Script-level only | None | Reference scripts | 0 (we already write CleanRL-style code) | Template, not a dep |
| **MARLlib 1.0** | No: built on RLlib; per-env metadata lists `Action Space: Discrete` | No | No (PettingZoo integration is the MPE parallel example) | Yes (MAPPO-family, parameter sharing) | Off-policy sibling repo | JMLR 2023, maintenance thin | Inherits RLlib cost + extra abstraction | No |
| **OpenSpiel** | Yes at the game-API level: `legal_actions()` / `legal_actions_mask()` per state | No: fixed-shape `observation_tensor` / `information_state_tensor`, `num_distinct_actions()` fixed | N/A (its own EFG abstraction) | AlphaZero (C++/TF), CFR/Deep CFR | Replay of game states, not our JSONL | Mature research framework | Months: port Evolution rules into C++/Python, then keep parity | No (later: 2p CFR/exploitability) |
| **LightZero (AlphaZero/MuZero)** | No: `legal_actions()` over `action_space.n`; mask `np.ones(env.action_space.n)` | No: board/Atari tensors | No | MCTS self-play | MCTS replay | Active, 0.2 (2026) | Research project with imperfect information | No |
| **DouZero-style self-play** | No: fixed 10⁴ action encoding + legal mask | No | No (own engine) | Yes: parallel actors, PMT-style shared buffers | Yes | Research code (ICML 2021) | Borrow design, not code | Template for M2 |

## 4. What each option actually requires (and what stays ours)

**TorchRL.** The wrapper is the only one that accepts an AEC *instance* and produces a `TensorDict` multi-agent view, and it has genuinely useful pieces: `MaskedCategorical` (mask, `padding_value`, entropy over valid entries), `TensorDictReplayBuffer` + `SliceSampler` (`trajs_per_batch`, `strict_length=False`, ragged 1-D storage, optional `pad_output` with `("collector","mask")`), and `Composite` specs. But the env path needs two contract changes we cannot make: flatten observations into fixed-width tensors (losing `Sequence` entity sets) and expose a fixed action universe with a per-step mask (losing `Discrete(len(choices))`). `slice_len`/`num_slices` sampling also solves sequence length, not per-entity set size — collation stays ours.

**RLlib.** Its canonical "variable-length / parametric action" recipe is exactly our design, one level down: `self.action_space = Discrete(max_avail_actions)`, `observation_space = Dict({action_mask, avail_actions, real_obs})`, custom model computes logits as dot products against `avail_actions`, invalid entries get `-inf`; "to enable batching, the number of actions can be allowed to vary from 1 to some max number". Its autoregressive recipe requires a custom model **and** a custom action distribution over a fixed `Tuple` space ("sampling an N-part action requires N forward passes"). Both reintroduce caps. The AEC path (`PettingZooEnv`) works for Leduc Hold'em-style fixed spaces; our 8p/90 s episodes would also pay Ray serialization for large dict observatio…

**SB3.** `MaskablePPO` requires a mask whose shape equals the fixed `action_space`; PettingZoo's own SB3 tutorial says the adapter "assumes that the action space and observation space is the same for each agent". My probe shows the failure mode is silent when the space changes. Reject.

**OpenSpiel.** Conceptually the closest match to our problem class (sequential, imperfect information, n-player, legal action sets): `legal_actions`, `legal_actions_mask`, `num_distinct_actions`, `information_state_tensor`, and shipped AlphaZero + CFR/Deep CFR. But it is a game implementation framework: adopting it means reimplementing Evolution's rules and all addons in C++/Python, duplicating the authoritative JS engine. Only interesting if 2p CFR/exploitability becomes a goal.

**MCTS stacks.** LightZero's env contract is a fixed `action_space` plus `legal_actions()` mask over it; MCTS expands one action per node. Our compound action would become a chain of prefix nodes (possible), but planning with hidden information (opponent hands, ambush answers) needs determinization/PIMC, which is a research project and would not use the Node engine cheaply. Defer.

**DouZero/AlphaStar are architecture references, not dependencies.** DouZero validates fixed action encoding + mask + parallel actors + shared architecture; AlphaStar Unplugged is the direct published precedent for our prefix scheme: "Actions are comprised of seven arguments … sampled autoregressively … the next two are sampled from the vector and units modalities using pointer networks … `unit_tags` is actually obtained by sampling the pointer network 64 times autoregressively" ([arXiv 2308.03526](https://arxiv.org/abs/2308.03526) §4.2). That is precisely "type → card → side → position → entity → target" over variable entity sets, and it was trained without a global action head.

**Always ours, in every option:** candidate collation/padding, episode→batch assembly, prefix log-prob grouping, replay storage, league/opponent pool, two-model (current + frozen opponent) batched inference, checkpoint/catalog compatibility, and the reward assignment rule (terminal-only, ±1 / −1/(N−1), γ=1).

## 5. Three-milestone plan

### M0 — finish the on-policy core (1–2 weeks, no new deps)

Keep `train_smoke.py`; make the record explicit so later stages do not re-derive it.

```python
Decision = {                      # one prefix choice
  "decision_id": int, "episode": int, "step": int,   # step = prefix index within episode
  "agent": int, "kind": int,                          # NORMAL/DEFENSE/.../TRANSIENT
  "context":  float32[Dc],                            # encoder output, 64
  "choice_rows": float32[K, Dk], "choice_mask": bool[K],
  "chosen": int, "logprob_old": float, "value_old": float,
}
Episode = {"seed": int, "config": dict, "decisions": [Decision, ...],
           "returns": float32[N], "terminated": bool, "truncated": bool}
```

```python
def collate(decisions) -> Batch:      # pad K to batch max; padding mask=False
    return {"context": f32[B,Dc], "choices": f32[B,Kmax,Dk],
            "choice_mask": bool[B,Kmax], "chosen": i64[B],
            "logprob_old": f32[B], "value_old": f32[B],
            "return_": f32[B], "agent": i64[B], "episode": i64[B], "step": i64[B]}
```

Update loop (γ = 1, terminal return, `-inf` on padding):
```python
logits = score(context, choices)                       # [B, Kmax]
logits = logits.masked_fill(~choice_mask, -inf)
logp   = log_softmax(logits, -1).gather(-1, chosen[..., None]).squeeze(-1)
adv    = ret - value(context).detach()
loss   = -(logp * adv).sum()/B + 0.5*((value(context)-ret)**2).mean() \
         - beta * masked_entropy(logits, choice_mask).mean()
```
For PPO later, compute the ratio on the **whole compound action** (`exp(Σ_t logp_new − Σ_t logp_old)`), not per prefix step, whenever all steps of a compound action share one terminal return.

### M1 — replay + off-policy/recurrent (2–4 weeks; optional `torchrl`)

Buffer interface (own implementation, ~150 LOC; TorchRL optional):

```python
buffer.add_episode(Episode)                 # ring buffer over episodes, optional .npz spill
buffer.sample(n_episodes) -> BatchT         # [E, Tmax, ...] + time_mask[E,Tmax]
compute_gae(BatchT, gamma=1.0, lam=None)    # per-agent, prefix-step level
```
```
BatchT = {"context": f32[E,T,Dc], "choices": f32[E,T,Kmax,Dk], "choice_mask": bool[E,T,Kmax],
          "chosen": i64[E,T], "logprob_old": f32[E,T], "value_old": f32[E,T],
          "agent": i64[E,T], "step": i64[E,T], "time_mask": bool[E,T]}
```
Adopt `torchrl.data.TensorDictReplayBuffer` + `SliceSampler(strict_length=False)` only if we add a sequence model; then store fixed-width encoded decisions (padding done by us at insert) and let the sampler handle ragged episode length. Everything else stays.

### M2 — league, evaluation, throughput (4–8 weeks)

```python
League = {"members": [{"path": str, "elo": float, "created_at_episode": int,
                       "catalog_hash": str, "schema_version": int, "config": dict}],
          "sampler": "pfsp", "history": [...]}
checkpoint = {"model": state_dict, "optimizer": state_dict, "catalog": dict,
              "schema_version": 1, "config": dict, "seed": int, "league": League}
```
Seat-rotated self-play; opponents frozen and run in the same process (CPU inference, batched across the decisions of one actor step); evaluate on fixed seeds against `random`, `survival`, and past snapshots; every evaluation reuses the same `config`/`catalog_hash`. Optional 2p MCTS over the prefix tree using the candidate scorer as prior — flagged as research because of hidden information.

## 6. Risks

- **Stay custom:** we own PPO/GAE/league correctness; risk of subtle advantage/ratio bugs. Mitigation: the existing scripted gradient fixture plus a small tabular sanity env. Lowest integration risk, no caps.
- **TorchRL hybrid:** version churn (0.x), wrapper pinned in practice to pettingzoo 1.24.3, and any future attempt to use its `Collector` re-opens the fixed-action problem; also its docs warn that dynamic specs plus parallel envs need careful benchmarking.
- **TorchRL full:** forces a fixed action universe and flattened padded observations; destroys the pointer/entity structure that makes 8p tractable, and violates the no-cap requirement.
- **RLlib:** Ray dependency and API churn, plus the same caps; the AEC path is a wrapper over fixed-space `MultiAgentEnv`, and `-inf` masking "might not work with all algorithm options".
- **SB3:** silent wrong behaviour on space changes (measured); single-agent design.
- **OpenSpiel:** full rules reimplementation and permanent parity risk against the JS engine; only justified for 2p CFR.
- **MCTS/MuZero:** imperfect information requires determinization or belief search; not a cheap path to strength.

**Inference flags:** all engineer-day estimates are mine; "TorchRL buffer handles only trajectory-length raggedness" is inferred from `SliceSampler`'s storage contract (1-D flat sequences, fixed feature width), not measured; RLlib was not installed or run (docs-only).

## 7. Sources

- TorchRL PettingZoo wrapper: <https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.PettingZooEnv.html> and source <https://github.com/pytorch/rl/blob/main/torchrl/envs/libs/pettingzoo.py>
- TorchRL dynamic specs: <https://docs.pytorch.org/rl/stable/reference/envs_libraries.html#dynamic-specs>
- TorchRL `SliceSampler`: <https://docs.pytorch.org/rl/stable/reference/generated/torchrl.data.replay_buffers.SliceSampler.html>; `MaskedCategorical`: <https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.MaskedCategorical.html>
- RLlib parametric/autoregressive actions: <https://docs.ray.io/en/releases-2.5.1/rllib/rllib-models.html> (§ "Variable-length / Parametric Action Spaces", "Autoregressive Action Distributions")
- RLlib PettingZoo + multi-agent/self-play/league/offline: <https://docs.ray.io/en/latest/rllib/index.html>; PettingZoo AEC Hold'em tutorial: <https://pettingzoo.farama.org/tutorials/rllib/holdem/>
- SB3/PettingZoo: <https://pettingzoo.farama.org/tutorials/sb3/> and <https://pettingzoo.farama.org/tutorials/sb3/connect_four/>; `MaskablePPO`: <https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html>
- CleanRL: <https://github.com/vwxyzjn/cleanrl> ("not a modular library … not meant to be imported")
- MARLlib: <https://github.com/Replicable-MARL/MARLlib>; env metadata <https://marllib.readthedocs.io/en/latest/handbook/env.html>
- OpenSpiel: API <https://openspiel.readthedocs.io/en/latest/api_reference.html>; algorithms <https://github.com/google-deepmind/open_spiel/blob/master/docs/algorithms.md>
- LightZero env contract: <https://github.com/opendilab/LightZero/blob/main/docs/source/tutorials/envs/customize_envs.md>
- DouZero: <https://github.com/kwai/DouZero>, paper <https://arxiv.org/abs/2106.06135>
- AlphaStar Unplugged (autoregressive args + pointer networks): <https://arxiv.org/abs/2308.03526> §4.2; Pointer Networks <https://arxiv.org/abs/1506.03134>; Deep Sets <https://arxiv.org/abs/1703.06114>
- Invalid action masking: <https://arxiv.org/abs/2006.14171>

[Some lines truncated to 768 chars. Use agent://StackResearch:raw:39-39 to read line 39 whole]
