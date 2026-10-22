# Repository Guidelines

## Project Overview

Evolution board game web app plus a headless reinforcement-learning environment built on the same authoritative rules.

- **Web game**: React/Redux client, Express + Socket.IO server, MongoDB persistence.
- **Rules engine**: shared JavaScript game models and action handlers, identical for browser, server and RL.
- **RL stack**: Node worker process speaking JSONL to a Python PettingZoo AEC environment, with self-play rollouts, baseline policies and a training smoke test.

Rules must never be reimplemented outside `shared/`. The RL layer consumes the same handlers, reducers and models as the web app.

## Architecture & Data Flow

### Web request flow

1. Client dispatches an action `{type, data, meta}`; `meta.server` actions are emitted over the socket.
2. `server/socket.js` verifies `JWT_SECRET`, looks up `clientToServer[type]` and dispatches the returned thunk with `{connectionId, userId}`.
3. Thunk validates state (throws `ActionCheckError` with `userLevelError: true` for user errors), then dispatches domain events.
4. `server/reducers/games-rdx-server.js` reduces Immutable.js `GameModel` state under `state.games[gameId]`.
5. Socket middleware broadcasts actions to the affected sockets; `serverToClient[type]` on the client rebuilds models and updates the client store.
6. `meta.clientOnly` actions are never passed to authoritative reducers.

Game-end side effects (room self-destruct, `toDatabase`, logging) live in `server/middleware/gameLifecycle.js`, triggered by the `gameFinished` event; browser-only effects (redirect, sounds) live in `client/configuration/game-handlers.js`.

### RL data flow

1. Python `bridge.py` starts one `node rl/node/worker.cjs` per environment; JSONL ops: `hello/reset/observe/step/snapshot/restore/export_replay/close`.
2. `rl/node/environment.js` owns an isolated store (games reducer + minimal episode rooms), a logical scheduler and an episode runtime.
3. Decisions are **prefix steps**: one integer choice over the current `choices`; a compound action spans several steps. `Frame` carries `actor`, `decisionId`, `observations`, `rewards`, `terminated`, `truncated`, `info`, `committed`.
4. Observations are a strict whitelist from `rl/schema.json`; action legality is enumerated on the player projection (`game.toOthers(playerId)`), never on full state.
5. Python `aec.py` exposes per-decision `Discrete(len(choices))`; `train_smoke.py` consumes observations and choices directly.

## Key Directories

| Path | Purpose |
|---|---|
| `shared/models/game/` | Game records and vocabulary: `GameModel`, `PlayerModel`, `CardModel`, `GameSettings`, evolution traits. |
| `shared/models/game/evolution/` | Animal/trait models, `traitsData/` rule implementations, `plantarium/` plants and plant traits. |
| `shared/actions/` | Authoritative domain actions, checks, socket handler maps, `game-validation.js`. |
| `shared/selectors/` | Redux selectors (`selectGame`, `selectUsersInGame`, ...). |
| `client/`, `server/` | Web UI, socket configuration, Express routes, reducers, middleware. |
| `rl/node/` | Headless runtime: environment, decisions, actions, observation, catalog, scheduler, store, worker. |
| `rl/python/evolution_rl/` | AEC env, JSONL bridge, Gymnasium spaces, baselines, rollout, benchmark, trainer. |
| `rl/python/tests/`, `rl/node/test/` | RL tests. |
| `docs/` | Approved plan, implementation notes, archived research reports. |
| `rl/schema.json` | Observation, Frame, choice-row and protocol contract (bump `schema_version` on breaking changes). |

## Development Commands

More copy-paste recipes (matrix monitoring, checkpoint inspection, cleanup): [`docs/commands.md`](docs/commands.md).

```bash
npm ci --ignore-scripts          # install JS deps from package-lock.json
npm run build                    # webpack server + client bundles
npm run start                    # nodemon + babel-node server

npm run rl:test                  # mocha RL specs (no globals.js, no JWT)
npm run rl:smoke                 # worker smoke, natural terminal required

uv sync --locked --extra train   # Python 3.11 environment + CPU torch
uv run --locked pytest rl/python/tests -q

uv run --locked python -m evolution_rl.rollout --profile base --players 2 --episodes 20 --seed 0 --workers 2
uv run --locked python -m evolution_rl.rollout --matrix full --seeds 0:0 --workers 8 --status-file rl/runs/matrix_00_status.json
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

## Code Conventions & Common Patterns

- **Language/format**: Babel ESM source, no TypeScript. Two-space indent, single quotes, no semicolons, comma-first object style (match surrounding code).
- **Imports**: `babel-plugin-root-import` maps `~/` to the repo root. Prefer relative imports inside `shared/`.
- **Immutability**: Immutable.js 3.x everywhere (`Record`, `Map`, `List`, `OrderedMap`). Never mutate game state; return new records.
- **Actions**: plain `{type, data, meta}`. Naming: `server$...` authoritative handlers, `client$...` client-only, `...Request` client intents, domain events like `gameNextPlayer`.
- **Thunks**: handlers receive `(data, meta)` and return `(dispatch, getState) => ...`.
- **Errors**: user mistakes throw `ActionCheckError` (`userLevelError: true`, `name`, `context`); use `throwError`, `passesChecks`. A non-user-level exception is an engine bug.
- **Validation**: pure guards live in `shared/actions/game-validation.js` with stages `before_sharing | after_sharing | complete`. Validators must not dispatch, draw RNG or run auto-sharing.
- **Randomness/time/IDs**: gameplay code uses `getRandom`, `getIntRandom`, `gameNow()`, `gameId(kind)` from `shared/utils/gameRuntime.js`. The RL runtime supplies seeded Mulberry32, monotonic logical time and an id counter. `withGameRuntime` is synchronous; returning a Promise is an error.
- **Timers**: game timeouts are control actions (`addTimeout`/`cancelTimeout`/`checkTimeout`). The RL scheduler stores them logically and never fires on wall clock; do not remove the turn marker, it changes FEEDING rules.
- **RL action domain**: options are pruned to those with an admissible completion; each option carries internal ids for resolution and observation-safe slots for the policy.
- **Observation privacy**: own hand and own hidden traits only; opponents get counts; no deck order, seed, UUIDs, timers or raw hidden values.

## Important Files

- Entry points: `server/index.js`, `client/index.js` (via webpack), `rl/node/worker.cjs`.
- Store setup: `server/configureStore.js`, `client/configuration/configureStore.js`, `rl/node/store.js`.
- Rules core: `shared/actions/game.js`, `shared/actions/trait.js`, `shared/actions/game.plantarium.js`, `shared/models/game/GameModel.js`.
- Reducers: `server/reducers/games-rdx-server.js`, `server/reducers/rooms-rdx-server.js`.
- Validation/checks: `shared/actions/checks.js`, `shared/actions/trait.checks.js`, `shared/actions/rooms.checks.js`, `shared/actions/game-validation.js`.
- Contracts: `rl/schema.json`, `rl/python/evolution_rl/spaces.py`.
- Build: `package.json`, `.babelrc`, `webpack.client.babel.js`, `webpack.server.babel.js`, `gulpfile.babel.js`, `pyproject.toml`, `uv.lock`.

## Runtime/Tooling Preferences

- **Web runtime**: Node 16 (`.nvmrc`, `engines`). Do not migrate web Node/npm tooling for RL work.
- **RL runtime**: Node 24 for the headless engine; Python 3.11 managed by `uv` (`uv sync --locked`, `uv run --locked`). No pip/requirements.txt/manual venv.
- **Package manager**: `npm` with the committed `package-lock.json`; do not regenerate it for RL changes.
- **Python deps**: `numpy`, `gymnasium`, `pettingzoo`; `torch` is an opt-in extra pinned to the CPU wheel. GPU requires a manual CUDA wheel, then `--device cuda`.
- **Logging**: `LOG_STREAM=stderr` keeps JSONL stdout clean; `winston` Google transports only when `GOOGLE_LOGS_FILE` is set.
- **Environment**: `MONGO_URL`, `JWT_SECRET`, `GOOGLE_*`, `TEST`, `NODE_ENV`. The RL worker must not inherit Mongo/Google/JWT credentials.
- **Headless boundary**: RL code must not import `globals.js`, `server/database.js`, `server/socket.js` or `client/**`; logs belong on stderr.

## Testing & QA

- **Web**: mocha + chai, specs next to code as `*.spec.js`. The web harness requires `globals.js` and `shared/test-helper.js`; gulp targets only report pass/fail, so run mocha directly when exit codes matter.
- **RL Node**: `npm run rl:test` runs `rl/node/**/*.spec.js` with `@babel/register` and `rl/node/test/setup.cjs` — Chai only, no JWT, no browser globals.
- **RL Python**: `pytest rl/python/tests`; fake-child transport tests plus policy/rollout unit tests. End-to-end checks use `pettingzoo.test.api_test` and `seed_test`.
- **Determinism**: same seed plus same choices must produce identical state hashes; `observe`, catalog building and choice enumeration must not consume RNG/ids/clock.
- **Behavioral gates**: environment specs cover terminal rewards, truncation, snapshot/restore, reset idempotence, scenario validation, linked traits and transient windows; the addon matrix covers all 128 addon subsets for 2–8 players.
- **Do not** run the full web suite or heavy matrices mid-change; run the targeted spec first, then the full suite once on the integrated tree. Report only checks that were actually executed.
