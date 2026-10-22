# Reusable commands

Copy-paste recipes for this repository. Run everything from the repo root.
`package.json` scripts stay the single source of truth; the table below is the
reference for what each one does.

## npm scripts

| Script | What it runs |
|---|---|
| `npm run start` | `nodemon --exec babel-node server/index.js` |
| `npm run build` | `clean` + `server:build` + `client:build` |
| `npm run server:build` | webpack server bundle |
| `npm run client:build` | webpack client bundle with progress |
| `npm run server:start` | run the built server from `dist/` |
| `npm run clean` / `server:clean` / `client:clean` | remove `dist/` / `dist/server` / `dist/client` |
| `npm run test:shared[:once]` | gulp shared specs (watch / once) |
| `npm run test:client[:once]` | gulp client specs |
| `npm run test:server[:once]` | gulp server specs |
| `npm run test:all[:once]` | gulp shared + client + server |
| `npm run rl:smoke` | `node rl/node/worker.cjs --smoke` |
| `npm run rl:test` | mocha over `rl/node/**/*.spec.js` with `@babel/register` and `rl/node/test/setup.cjs` |
| `npm run heroku-postbuild` | alias for `build` on deploy |

Notes:

- `gulp` test targets can swallow failures; use the direct mocha command below
  when the exit code matters.
- `rl:test` deliberately avoids `globals.js`, `shared/test-helper.js` and JWT,
  so a regression in the headless boundary fails the test.
- Python has no npm wrapper: use `uv run --locked ...` (see below).

## Setup

```bash
npm ci --ignore-scripts            # JS deps from package-lock.json
uv sync --locked                   # Python 3.11, no torch
uv sync --locked --extra train     # + CPU torch for training/benchmark
```

## Web app

```bash
npm run start                      # nodemon + babel-node server
npm run build                      # webpack server + client
npm run test:shared:once           # gulp shared suite (exit code is not proof)
npm run test:all:once              # gulp shared + client + server
```

Direct mocha (reliable exit code):

```bash
TEST=true NODE_ENV=test JWT_SECRET=secret ./node_modules/.bin/mocha \
  --require @babel/register --require @babel/polyfill \
  --require source-map-support/register --require ignore-styles \
  --require ./globals --require ./shared/test-helper.js "shared/**/*.spec.js"
```

Single spec / directory:

```bash
TEST=true NODE_ENV=test JWT_SECRET=secret ./node_modules/.bin/mocha \
  --require @babel/register --require @babel/polyfill \
  --require source-map-support/register --require ignore-styles \
  --require ./globals --require ./shared/test-helper.js \
  "shared/actions/plantarium-specs/plant.carnivorous.spec.js"
```

## RL tests

```bash
npm run rl:test                                  # rl/node/**/*.spec.js
npm run rl:smoke                                 # one worker smoke run
uv run --locked pytest rl/python/tests -q        # transport/policy/rollout tests
```

PettingZoo conformance:

```bash
uv run --locked python - <<'PY'
from pettingzoo.test import api_test, seed_test
from evolution_rl import raw_env
api_test(raw_env(config={'numPlayers': 2, 'maxDecisions': 400}), num_cycles=400, verbose_progress=False)
print('api_test ok')
seed_test(lambda: raw_env(config={'numPlayers': 2, 'maxDecisions': 300}), num_cycles=100)
print('seed_test ok')
PY
```

## Rollouts

```bash
# one profile, N players
uv run --locked python -m evolution_rl.rollout --profile base --players 2 --episodes 20 --seed 0 --workers 2
uv run --locked python -m evolution_rl.rollout --profile all  --players 8 --episodes 1  --seed 0 --policy survival --workers 1

# addon matrix: 128 addon subsets x players 2..8 x seeds x both policies
uv run --locked python -m evolution_rl.rollout --matrix full --seeds 0:0 --workers 8 \
  --status-file rl/runs/matrix_00_status.json

# progress on stderr, live status JSON, planned episode count
uv run --locked python -m evolution_rl.rollout --matrix full --seeds 0:0 --workers 2 --progress
uv run --locked python -m evolution_rl.rollout --matrix full --seeds 0:0 --dry-run
```

Monitor and stop:

```bash
tail -f /tmp/matrix_00.err | grep --line-buffered progress
watch -n 5 'jq "{done,total,eta_seconds,counts}" rl/runs/matrix_00_status.json'
pkill -f "rl/node/worker.cjs"
```

Keep the machine responsive: `nice -n 19 ...`, and pick `--workers` below the core count.

## Benchmark

```bash
uv run --locked python -m evolution_rl.benchmark --profile all --players 8 --episodes 20 --workers 1,4
uv run --locked python -m evolution_rl.benchmark --profile base --players 2 --episodes 20 --workers 1,4
```

## Training

```bash
# CPU smoke: 32 episodes, fixed eval seeds, checkpoint + report
uv run --locked --extra train python -m evolution_rl.train_smoke --episodes 32 --seed 7 --output rl/runs/smoke

# GPU: install a CUDA wheel once, then select the device
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
uv run --locked --extra train python -m evolution_rl.train_smoke \
  --episodes 32 --seed 7 --output rl/runs/smoke --device cuda

# inspect / evaluate a checkpoint
uv run --locked --extra train python - <<'PY'
import torch
from evolution_rl import raw_env
from evolution_rl.train_smoke import feature_spec, build_model, evaluate, _catalog_from_env
env = raw_env(config={'numPlayers': 2}); env.reset(seed=0)
catalog = _catalog_from_env(env); env.close()
model = build_model(feature_spec(catalog), seed=7)
ck = torch.load('rl/runs/smoke/checkpoint.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ck['model'])
print('device', ck.get('device'), 'schema', ck['schema_version'])
for row in evaluate(model, [10000, 10001]):
    print(row['seed'], 'reward', row['reward_player_0'])
PY
```

Training artifacts: `rl/runs/smoke/report.json`, `rl/runs/smoke/checkpoint.pt`.

## Debugging

```bash
RL_DEBUG=1 uv run --locked python -m evolution_rl.rollout --profile base --episodes 1   # engine stack traces on stderr
LOG_LEVEL=debug npm run start                                                          # web logging
node rl/node/worker.cjs --smoke | jq '{ok,decisions,terminated,truncated}'             # worker only
```

## Cleanup

```bash
rm -rf rl/runs                 # generated checkpoints, reports, matrix status
pkill -f "rl/node/worker.cjs"  # stale engine processes
git status --short
```
