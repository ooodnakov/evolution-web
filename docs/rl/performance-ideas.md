# Performance ideas for RL workers

Status: proposals. Row 9 (precompiled RL modules) is implemented via
`npm run rl:build` + Bun 1.4.4; the rest are not. The mask/slot optimizations
in `e534a5e` are already in. Do not trade correctness for speed: no action
caps, no oracle masks, no rule changes, no reduced observation schemas beyond
the actor-only transfer the plan already allows.

## Measured baseline

| Scenario | Cost |
|---|---|
| Base 2p episode | ~0.7 s |
| 8p all-addons episode | ~90 s single worker |
| Train smoke, 32 base2p episodes | ~78 s (CPU) |
| Worker process start (Node 24 + Babel register) | measure per change |

## Ranked ideas

| # | Idea | Where | Expected effect | Risk |
|---|---|---|---|---|
| 1 | Reuse one env per worker thread; pull episodes from a queue and `reset()` | `rl/python/evolution_rl/rollout.py` `_run_task` currently creates `raw_env()` per episode | Removes Node start + `@babel/register` + module compile from every one of 1792 matrix episodes; keeps Babel cache warm | Reset is already covered by specs; watch memory growth and snapshot handles |
| 2 | Make replay checkpoints opt-in | `rl/node/environment.js` `completeDecision` calls `stateHash()` for every committed command: `game.toJS()` → canonical JSON → SHA-256 | Removes a full state serialization + hash per decision (large at 8p) | Replay artifact must keep hashes when requested; tests that compare checkpoints keep them on |
| 3 | Actor-only `Frame.observations` | `rl/node/environment.js` `frame()` builds every agent's whitelist projection | At 8p: ~8× fewer projections and JSON bytes per step; plan allows actor-only transfer | Python `observe(agent)` must fall back to the RPC; update contract notes and AEC caching |
| 4 | Split observation cache: static state vs dynamic choices | `publishDecision` clears `observationCache` on every prefix step, rebuilding the actor projection ~2× per compound action | Cache the state portion per `(stateVersion, agent)` and overlay `choices/context/mask` | Must preserve privacy filtering and slot alignment |
| 5 | Cache `game.toOthers(playerId)` | `makeContext()` rebuilds the masked GameModel (deck + opponent hands) for every decision and validation batch | Small but broad win | Cache per `(stateVersion, playerId)`; invalidate on every dispatch |
| 6 | Lazy level-by-level action enumeration | `rl/node/actions.js` `buildLeaves` materializes all leaves before filtering | 8p trait combinations can create thousands of leaves for tens of visible choices; generate only the next level with memoized completion search | Larger refactor; all pruning tests must still pass |
| 7 | Optional "trust enumerator" mode for training | `leafAdmissible` runs `validateGameRequest` per candidate | Skips projection validation on hot paths | Never enable in correctness gates; an enumerator bug would no longer surface as `ENGINE_INVARIANT` |
| 8 | Python transport trimming | `_cache_observations` normalizes every agent's dicts; catalog handshake per episode | Normalize only the requested agent, reuse catalog across episodes, use `orjson` for frame parse/dump | Optional dependency; protocol unchanged |
| 9 | Precompile RL Node modules (**implemented**: `npm run rl:build` -> `dist/rl`, run by Bun 1.4.4) | Ship `dist/rl` built once with Babel; run plain Node or Bun | Removes `@babel/register` and transform cost from worker startup | Build step to maintain; rebuild after source changes |

## Measurement protocol

Run on a quiet machine, fixed seeds, medians over repeated runs:

```bash
# 2p
uv run --locked python -m evolution_rl.rollout --profile base --players 2 --episodes 20 --seed 0 --workers 1
# 8p
uv run --locked python -m evolution_rl.rollout --profile all --players 8 --episodes 1 --seed 0 --workers 1 --policy survival
```

Step-level breakdown (hash vs observation vs enumeration vs dispatch) needs
temporary instrumentation in `environment.js`/`step`; add it as a local patch,
do not ship a profiling flag without a gate.

For each change record: wall time, decisions/s, bytes/step, RSS, and that
determinism (state hashes) and `npm run rl:test` / `pytest` still pass.
