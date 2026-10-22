# RL environment: implementation notes

Companion to the approved plan (`docs/rl/plan.md`) and the research reports in
`docs/research/`. This file records what was actually built, how it was
verified, which engine defects were fixed on the way, and what is still open.

## Layout

```
rl/schema.json                     observation / Frame / protocol contract
rl/node/
  worker.cjs                       JSONL launcher (Babel, logs to stderr, --smoke)
  environment.js                   reset/observe/step/snapshot/restore/replay
  decisions.js                     terminal/question/ambush/regeneration discovery
  store.js                         isolated store: games reducer + episode room slice
  scheduler.js                     logical timers (no wall clock)
  runtime.js                       Mulberry32 RNG, id allocator, logical clock
  actions.js                       prefix action enumerator + resolve
  observation.js                   privacy-filtered observation projection
  catalog.js                       stable vocabulary (0 = NONE)
  slots.js                         local animal/plant/shell slot order
  test/*.spec.js                   mocha behavioral specs (no globals.js)
rl/python/evolution_rl/
  aec.py                           PettingZoo AEC adapter, per-decision Discrete
  bridge.py                        one subprocess per env, deadline + stderr drain
  spaces.py                        Gymnasium Dict/Sequence spaces from the schema
  policies.py                      random / survival baselines
  rollout.py                       self-play rollouts, matrix, progress, status file
  benchmark.py                     throughput measurements
  train_smoke.py                   CPU/GPU REINFORCE + value baseline smoke
shared/actions/game-validation.js  pure staged request guards (shared with web)
```

## Contracts

- Decisions are **prefix steps**: one choice index over the current `choices`;
  a compound game action spans several steps. The action space is
  `Discrete(len(choices))`, recreated only when the decision token changes.
- `Frame = {actor, decisionId, observations, rewards, terminated, truncated,
  info, committed}`. Rewards are deltas of the last step; terminal rewards are
  issued once (`winner +1`, other active seats `-1/(N-1)`); truncation pays 0.
- Observations are a whitelist: own hand only, opponents get counts; opponent
  `hidden` traits omitted, transient values suppressed; no deck order, seed,
  UUID or timer values.
- `mask_semantics = public_preconditions`: choices are enumerated and pruned on
  the player projection (`game.toOthers(playerId)`), never on full state.
- Runtime context is synchronous (`withGameRuntime`); async callbacks are
  rejected. Every episode owns RNG, id counter and logical clock.

## Commands

```bash
npm run rl:test                    # 72 mocha specs, no web harness
npm run rl:smoke                   # worker smoke, exits 0 on natural terminal
uv run --locked pytest rl/python/tests -q
uv run --locked python -m evolution_rl.rollout --profile base --players 2 --episodes 100 --seed 0 --workers 2
uv run --locked python -m evolution_rl.rollout --matrix full --seeds 0:0 --workers 8 \
  --status-file rl/runs/matrix_00_status.json
uv run --locked python -m evolution_rl.benchmark --profile all --players 8 --episodes 20 --workers 1,4
uv run --locked --extra train python -m evolution_rl.train_smoke --episodes 32 --seed 7 --output rl/runs/smoke
```

Legacy web suite:

```bash
TEST=true NODE_ENV=test JWT_SECRET=secret ./node_modules/.bin/mocha \
  --require @babel/register --require @babel/polyfill \
  --require source-map-support/register --require ignore-styles \
  --require ./globals --require ./shared/test-helper.js "shared/**/*.spec.js"
```

## Verification performed

| Check | Result |
|---|---|
| RL node specs | 72 passing |
| Python specs | 49 passing |
| Legacy shared suite | 415 passing, 7 pending |
| PettingZoo `api_test` (400 cycles) | passed |
| PettingZoo `seed_test` (100 cycles) | passed |
| `rl:smoke` | natural terminal, 111 decisions |
| Base 2p random, seeds 0–19 | 20/20 natural terminals |
| 8p all-addons survival | natural terminal, 0 errors |
| Determinism (same seed + choices) | identical state hash, unaffected by an interleaved env |
| Snapshot/restore branch | identical final state hash, foreign handle rejected |
| Scripted carnivorous fixture | safe `7:6` `+1`; poisonous `3:5` `-1`; one update raises safe log-prob `-1.39545 → -1.38649` |
| Train smoke (32 eps) | finite loss/gradients, weights changed, checkpoint reload reproducible, 8 fixed eval seeds |
| Full addon matrix, seed 0 | in progress at the time of writing; see `rl/runs/matrix_00_status.json` |

## Engine defects found and fixed

| Where | Defect |
|---|---|
| `game.plantarium.js` pest spawn | inclusive RNG upper bound `size` could pick `undefined`; empty plant list now skips the draw |
| `trait.js` Intellect default | passed undeclared `defenseAnimal` instead of the resolved target animal |
| `PlantTraitHiddenCarnivorous.action` | called with a leading `playerId` from one path and without it from another; normalized both call shapes |
| `games-rdx-server.js` `traitParalyze` | linked partner could be gone, and plant partners were indexed through the animal path |
| `games-rdx-server.js` `traitAmbushActivate` | activation after the ambush record was consumed threw `invalid keyPath`; now a no-op |
| `trait.js` ambush start/perform/end | missing ambush record crashed on `targets`; guarded |
| `game.plantarium.js` spawn | `concat(null)` injected a null card into the plant deck |
| `game.js` `server$animalDeath` | trematode link outliving its host animal crashed on `hasTrait` |
| `actions.js` normal activation | defense/transient/non-controllable traits were offered as normal actions; now mirrors the web option search |
| `observation.js` | cooldowns encode numeric duration; unresolved references become `NONE` instead of throwing or leaking ids |

## Known gaps

- Matrix errors are still being triaged from the seed-0 run; training-scale work
  must wait until the error rate is zero.
- `train_smoke` policy loss is summed while entropy is averaged, so the entropy
  coefficient is effectively inert.
- The type embedding is shared across catalog namespaces whose ids are
  namespace-local (trait id 1 aliases card id 1).
- Hand-card choice rows do not expose the card's two trait types, so the
  candidate scorer cannot see them.
- The plan's full matrix (`--seeds 0:19`, 35,840 episodes) has not been run;
  seed 0 is the current evidence.
- `attempt_rejected` state atomicity is implemented and covered by protocol
  error tests, but no deterministic scenario reproduces the rejection path.
- GPU training is opt-in: the locked environment pins the CPU torch wheel, so
  `--device auto` resolves to cpu until a CUDA wheel is installed manually.
