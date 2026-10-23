# Bun as a runtime for the RL worker — decision memo

Status: **adopted** (2026-10-23). The RL engine now runs on Bun 1.4.4 from the
compiled artifact `dist/rl/node/worker.cjs`, built by `npm run rl:build`
(`rl/build.cjs`). The memo below is the research record that led to that decision;
it was written before adoption at repo revision `d754cf3`, when production still ran
`node rl/node/worker.cjs` (Node 24 + `@babel/register`) and no `rl/`, `server/` or
`shared/` file was modified. Its measurements, limits and candidate routes are kept
as written; the "migration effort" section is a plan, not a description of the code
that was merged.

Probed at repo revision `d754cf3`.

## Adoption status

- Build: `rl/build.cjs` -> `npm run rl:build` compiles `shared/`, `server/` and
  `rl/node/` with the repository `.babelrc` into the gitignored `dist/rl`
  (`node/`, `shared/`, `server/`, `schema.json`). `node_modules`, `client/`,
  `rl/node/test` and `rl/runs` are not copied; the artifact resolves dependencies
  from the repository root. `dist/server` and `dist/client` are untouched.
- Runtime: `bun dist/rl/node/worker.cjs`. `rl:smoke`, the Python bridge/AEC,
  rollouts, benchmarks and training all use it; `runtime_executable` defaults to
  `bun`. No fallback to Node or to source-tree Babel; a missing build or missing
  Bun produces an actionable error.
- Source: `rl/node/worker.cjs` no longer installs `@babel/register`; the compiled
  tree is CommonJS ES5, preserving the callable `Record`-subclass constructors
  that immutable 3.8.2 `reify()` requires. Rebuild after changing `shared/`,
  `server/` or `rl/node/`. Web stays on Node 16; Node 24 runs the build and specs.
- Verification (2026-10-23): compiled Bun smoke 111 decisions, natural terminal;
  `rl/node/test/worker.spec.js` covers smoke, JSONL to terminal and cross-process
  replay parity from the artifact; default Python bridge 2p seed-0 episode
  terminated naturally with identical frames/replay across two fresh Bun
  processes; snapshot/restore reproduced the same continuation observations
  (post-restore decision ids shift by +1, as recorded below); a plantarium
  hidden-carnivorous plant attack raised a natural DEFENSE question with
  `TraitShell` and decline options.

## Verdict

| Question | Answer |
| --- | --- |
| Can Bun 1.4.4 replace Node 24 as a **drop-in** runtime for `rl/node/worker.cjs`? | **No — do not adopt.** Blocked by a language-semantics dependency, not by configuration. |
| Can Bun run the same worker from **precompiled** (Babel → CommonJS ES5) output? | **Yes**, with identical output on everything measured here. This is a *pilot* candidate gated on an AOT build step. |
| Is the precompiled route worth a pilot on runtime numbers alone? | Cold start 60 ms vs 103 ms (Node, same build) vs 426 ms (current Node + warm Babel cache) / 1944 ms (cold cache). Per-step medians 17–25 % lower than Node on the same precompiled tree. |

Scope of that parity claim: 2-player base deck, seeds 0 / 1 / 7 / 42 / 123, two choice
policies, one host, one version pair. It is **not** a universal guarantee and says nothing
about 8-player or addon scenarios, concurrency, or training.

So: **do not adopt as a swap; pilot only as a build-step project.** The runtime gain is real
but modest, and it is bought with a new build artifact that must also keep the Node path and
the webpack builds coherent.

## Why the untransformed-source path is blocked

Immutable.js 3.8.2 `reify()` calls `iter.constructor(seq)` **as a method**, so the
constructor must be a callable ES5 function that accepts the existing instance as `this`:

```js
// node_modules/immutable/dist/immutable.js:3569
function reify(iter, seq) {
  return isSeq(iter) ? seq : iter.constructor(seq);
}
```

`shared/models/game/CooldownList.js` and 15 other modules declare `class X extends Record(...)`.
Babel’s ES5 output turns those into ordinary functions, so `X(seq)` re-runs the constructor
with `this === instance` and `_classCallCheck(instance, X)` passes. A **native** `class`
cannot be called without `new` in any engine, so the same source throws. This was verified
engine-independently (a 6-line `class Y extends Record({...})` reproduces it under both
`node` and `bun`), and it is why every runtime, Node included, executes this code only
through Babel. It is **not** a Bun compatibility gap.

Three staged repros on a pristine copy of `rl/node`, `shared/`, `server/`, `.babelrc`,
`package.json` (Bun 1.4.4, `bun rl/node/worker.cjs --smoke`, exit 1 each time):

1. **No config** — 152 ms, exit 1:
   `{"ok":false,…,"error":{"code":"INTERNAL","message":"Cannot find package '~' imported from /tmp/evolution-bun-native/rl/node/environment.js"}}`
   (`babel-plugin-root-import` is not applied by Bun; Bun would need `tsconfig.json`
   `compilerOptions.paths`, which is documented for Bun but absent from this repo.)
2. **+ `tsconfig.json` = `{"compilerOptions":{"baseUrl":".","paths":{"~/*":["./*"]}}}`** — 181 ms, exit 1:
   `{"…","message":"Export named 'traitMoveFood' not found in module '/tmp/evolution-bun-native/shared/actions/trait.js'."}`
   (`shared/models/game/evolution/traitsData/customff.js` imports `traitMoveFood`, which
   `shared/actions/trait.js` never exports; it is dead code that Babel’s permissive CJS
   interop tolerates and native ESM linking rejects. The AOT route below keeps this source
   untouched and works.)
3. **+ removing that unused import name** — 189 ms, exit 1:
   `{"…","message":"Cannot call a class constructor CooldownList without |new|"}`

Stage 3 is the blocker for any untransformed-source run: it fires the first time a game
reducer touches `cooldowns` (`GameModel.gameNextPlayer → immutable updateInDeepMap → map →
reify`). What must be preserved is **callable Record-subclass constructors**, not ES5
syntax as such: a targeted transform of those 16 declarations, or migrating the model layer
off Immutable 3, are plausible alternatives that were not attempted. The AOT build resolves
it without touching source, which is why it is the route examined below.

### Other native routes tried

| Route | Result |
| --- | --- |
| `bun --preload` + `Bun.plugin({onLoad})` running Babel per module | Fails. With `transform-modules-commonjs` the transformed module loses its ESM export surface (`Export named 'CooldownList' not found`), and only the entry module was transformed for the CJS entry point (`environment.js does not export createEnvironment`; `require() async module … is unsupported. use "await import()" instead` when `onLoad` was async). Turning off the CJS module transform keeps ESM but leaves native classes → stage-3 failure. |
| `@babel/register` under Bun | Inert: `require.extensions` exists in Bun, but Bun loads ESM-syntax `.js` natively and never runs the hook; the run still ends at the stage-3 error. |
| Bun’s own transpiler | Bun’s loader honours `tsconfig` paths but performs no Babel plugin pipeline (root-import, class properties, Flow, `transform-runtime`); it keeps native class syntax. |

## The route that works: AOT Babel → CJS, then run either runtime

207 `.js` modules compiled / 6 files copied in ~3.4 s with the repo’s own `@babel/core`
7.19.6 and the exact `.babelrc` preset/plugin list. `babel-plugin-root-import` is pinned to
the **build** root; because the output tree mirrors `src/`, the emitted relative paths stay
valid. The complete, tested script is in [Reproducing](#reproducing) below — it is the exact
file that produced the smoke result here (`compiled 207 modules into …/build`, then
`{"ok":true,"decisions":111,"terminated":true,…}` on both `node` and `bun`).

The only edit to copied source is stripping the now-redundant `loadBabelRegister();` call in
`rl/node/worker.cjs`, because the build is already CommonJS ES5. **No source patch of any
kind was needed**: the pristine `customff.js` import and the `extends Record(...)` classes
work exactly as under Node, because both runtimes execute the same ES5 output. (After
adoption the call is gone from the source itself, so the artifact needs no rewriting.)

## Measurements

### Methodology

- Same machine, same `node_modules` (immutable 3.8.2, `@babel/register` 7.18.9,
  `source-map-support` 0.5.21), same JSONL protocol, one worker process per measurement.
- Harness speaks the documented protocol (`hello`/`reset`/`step`/`snapshot`/`restore`/
  `export_replay`/`close`) over pipes; every stdout line was parsed as JSON (zero non-JSON
  lines observed); child stderr was redirected to a temp file (holding it in an undrained
  `PIPE` is a real measurement hazard — verbose engine logging blocks on a full pipe).
- Startup = wall time from `Popen` to the `hello` response. n = 10 fresh processes per
  variant, alternating variant order in each pass, no CPU pinning, no other deliberate load.
- Episode = `reset` → natural terminal; `numPlayers: 2`, no addons, no scenario, base deck.
  Seeds 0, 1, 7, 42, 123; two policies:
  `zero` (always choice 0, 111 decisions every seed) and `mix`
  (`((decisionId+1)·2654435761 + seed·40503) >> 8 mod len(choices)`; 145–166 decisions).
- Babel cache state is explicit: `node24-register-warm` uses the persisted cache
  (`BABEL_CACHE_PATH` default `~/.babel.json`, primed by earlier runs);
  `node24-register-cold` sets `BABEL_DISABLE_CACHE=1`.
- These are process-per-measurement wall times on a desktop host; they are not a
  throughput claim for training (that is dominated by the Python side and the transport).

### Cold start (spawn → `hello`), ms

| Variant | p50 | mean | min–max |
| --- | --- | --- | --- |
| node24 + `@babel/register`, warm cache | 426.3 | 436.4 | 402.2–590.8 |
| node24 + `@babel/register`, cache disabled | 1943.6 | 1940.4 | 1923.9–1969.5 |
| node24, precompiled CJS | 102.9 | 102.4 | 100.7–104.7 |
| **bun 1.4.4, precompiled CJS** | **59.5** | 59.6 | 58.3–61.1 |
| bun 1.4.4, raw source (hello only; fails at `reset`) | 168.3 | 169.3 | 166.9–173.1 |

The last row is **not** a usable configuration: `hello` succeeds because it only loads the
environment module and catalog, then the first `reset` dies at stage 3. Its 168 ms must not
be read as a Bun performance number for working runs.

### `reset` and per-step latency, ms

`reset` p50 over the 5 seeds (first reset in a fresh process; this is first-use work such as
store creation and the reducer/action graph, plus, on the register path, any module
compilation that had not happened during `hello`):

| Variant | reset p50 | step p50 (`zero`) | step p50 (`mix`) | step p90 (`mix`) |
| --- | --- | --- | --- | --- |
| node24 + register (warm) | 112.51 | 1.14–1.27 | 9.7–14.1 | 44.6–61.8 |
| node24 precompiled | 8.49 | 1.04–1.07 | 2.47–3.20 | 6.1–7.5 |
| bun 1.4.4 precompiled | 7.40 | 0.87–0.89 | 1.95–2.32 | 3.9–4.8 |

Full episodes, wall ms (`zero` / `mix`), step counts identical across variants:

| Variant | zero, 111 steps | mix, 145–166 steps |
| --- | --- | --- |
| node24 + register (warm) | 371, 372, 376, 374, 374 | 3338, 2867, 2759, 3294, 2987 |
| node24 precompiled | 150, 147, 146, 148, 147 | 534, 477, 459, 514, 474 |
| bun 1.4.4 precompiled | 123, 126, 124, 123, 125 | 378, 343, 320, 351, 321 |

The large `mix` gap between Node+register and Node-precompiled is **not** a Bun effect and
not a code-generation effect: running the *same precompiled tree* with `@babel/register`
merely preloaded raises the per-step p50 from 3.03 ms to 13.84 ms. Preloading only
`source-map-support/register` on the same tree accounts for roughly half of that
(3.05 → 6.00 ms p50). [INFERENCE] the remainder is `@babel/register`’s
`pirates`/`Module._compile` path and its source-map bookkeeping — an artifact of the current
loading mechanism rather than of Babel’s output. Worth its own look independently of Bun:
a precompiled build makes the *Node* worker 2.5–6× faster per step on `mix` episodes.

### Determinism parity

Two different kinds of evidence were compared, and they are not interchangeable:

**Visible behaviour** — `sha256` of the canonicalised Frame (observations, choices, rewards,
terminated/truncated, info). The full per-step list (111 / 145–166 entries, including the
terminal frame) was **byte-identical** across node+register, node-precompiled and
bun-precompiled for every seed × policy. Because `observations` is part of the frame, this
covers the privacy-filtered observation payload and the embedded choice batches.

**Authoritative state** — `export_replay()` carries `replay.checkpoints[]`, and each entry is
`{decision_id, hash: stateHash()}` where `stateHash()` hashes the full serialised game plus
the runtime RNG capture and boundary counters (`rl/node/environment.js`). The whole replay
payload (seed, config, command list, checkpoint list) was **byte-identical** across the three
variants for every run; per-decision state hashes are therefore identical, not just the
visible frames. There is no protocol op that returns `stateHash()` directly; the checkpoints
are the only window on it, and they were compared at this level.

Spot values (precompiled, both runtimes equal): replay `6e07e79bed34c8e2…`, terminal frame
`7a80f8d5abca2252…` for seed 7 `zero`; replay `38ebfe89587bb887…` for seed 7 `mix`.

Snapshot/restore (5 seeds × 2 policies × 3 variants, 30 runs): snapshot at the midpoint,
advance three decisions, `restore`, replay the *same* choice indices. The three continuation
**frames** matched the uninterrupted run in every case, and node and bun produced identical
restore behaviour. At state-hash level the checkpoints up to and including the snapshot
boundary match the uninterrupted run; afterwards `restore` republishes the boundary decision
with `decisionId + 1`, so the replay’s later decision ids shift and repeat (observed ids
`…21, 22, 23, 22, 23, 24` after restoring at 22) and the post-restore replay is no longer
structurally equal to the uninterrupted one. That is engine bookkeeping, identical in both
runtimes, and it means restore parity is established on frames plus pre-boundary state
hashes — not on an identical post-restore replay stream.

Process hygiene: every measured run ended with `close` → `{"closed":true}`, exit code 0,
no trailing stdout bytes, and no non-JSON line on stdout.

## Migration effort (concrete work, not an estimate — historical proposal)

1. Add a build step that compiles `shared/`, `server/` and `rl/node/` with the repo
   `.babelrc` to a mirrored CJS tree (the tested script below), plus the
   `loadBabelRegister()` strip for the worker copy. Decide where the tree lives
   (`dist/rl` next to the existing `dist/server`) and whether `npm run build` produces it.
2. Select the precompiled tree at launch: `rl/python/evolution_rl/bridge.py`
   (`LAUNCHER_RELPATH = Path("rl") / "node" / "worker.cjs"`), `aec.py` (forwards
   `node_executable`), `rl/python/evolution_rl/__init__.py` (re-exports it). These are the
   launcher defaults; `package.json` scripts (`rl:smoke`) also name `node` explicitly, so
   they need the same decision. (As merged, the launcher is
   `dist/rl/node/worker.cjs`, the parameter is `runtime_executable` defaulting to `bun`.)
3. Update `package.json` scripts and the mocha setup (`rl/node/test/setup.cjs`,
   `--require @babel/register`) so tests run against either the source tree or the build,
   never a mix. Note `rl:test` transpiles at test time; a build step must not silently change
   which tree tests exercise.
4. Add a parity gate: run the same seeds/policies on Node and Bun from the compiled tree and
   compare replay payloads (which include per-decision state hashes) plus frame hashes. The
   harness used here is ~200 lines of Python and depends only on the documented protocol.
5. Document the build artifact in `docs/commands.md` and leave the webpack server/client
   builds untouched (they already consume `shared/` through their own Babel/Webpack setup).

Rollback: revert the launcher/runtime selection (`LAUNCHER_RELPATH`, the `runtime_executable`
default, and the scripts changed in step 3) back to `node rl/node/worker.cjs`, and stop
shipping the build tree (that also means restoring the source `@babel/register` install). Both trees execute identical semantics by construction, but the
revert touches several call sites, so treat it as a small patch, not a one-line toggle.

## Limits of this evidence

- 2-player base deck only. No 8-player or addon scenarios, no `--workers` concurrency, no
  training run, no Python/Gymnasium/PettingZoo path — the matrix was intentionally left
  stopped and no heavy 8p run was started.
- One host, one Bun version (1.4.4) and one Node version (24.22.0); numbers are wall-clock
  on an otherwise idle desktop, not a benchmark suite.
- The precompiled route was validated as a smoke + episode study, not with `rl:test`
  (mocha), the Python test suite, or the webpack builds.
- State hashes were only observable through `export_replay()` checkpoints; no direct
  `stateHash()` protocol op exists, and restore parity was verified at frame level plus
  pre-boundary state hashes (see above).
- `@babel/register`’s residual per-step overhead is only partially attributed
  (`source-map-support` ≈ half); the rest is unmeasured [INFERENCE].
- Bun’s docs cited below track the latest release (written against Node 26 compatibility);
  the measurements are Node 24.22.0 / Bun 1.4.4.

## Primary sources

- Bun module resolution and `tsconfig`/`jsconfig` path re-mapping:
  <https://bun.sh/docs/runtime/module-resolution>
- Bun Node.js compatibility (`node:module` is partial: `require.cache`,
  `require.extensions`, `module._resolveFilename` supported; `module._load` and
  `module.register` are no-ops): <https://bun.sh/docs/runtime/nodejs-compat>
- Bun runtime plugins (`onResolve`/`onLoad`) — the route tested above:
  <https://bun.sh/docs/runtime/plugins>
- `@babel/register` (transforms `require()`ed modules after the hook is installed,
  ignores `node_modules` by default, persisted cache via `BABEL_CACHE_PATH`,
  `BABEL_DISABLE_CACHE=1` to switch it off; it also installs `source-map-support`):
  <https://v7.babeljs.io/docs/babel-register> and
  `node_modules/@babel/register/lib/hook.js`
- Immutable.js 3.8.2 `reify()` — `node_modules/immutable/dist/immutable.js:3569` (local,
  authoritative for the vendored version).
- Repo contracts: `rl/schema.json` (protocol/frame), `docs/rl/implementation-notes.md`.

## Reproducing

Copy only what the worker needs (no `client/`, no `rl/runs/`) and symlink the repo’s
`node_modules` instead of copying it.

```sh
R=/home/odnakov/Projects/evolution-web
C=/tmp/bun-memo-check
rm -rf "$C" && mkdir -p "$C/src/rl"
cp -r "$R/shared" "$R/server" "$C/src/"
cp -r "$R/rl/node" "$C/src/rl/"
cp "$R/rl/schema.json" "$C/src/rl/"
cp "$R/.babelrc" "$R/package.json" "$C/src/"
ln -s "$R/node_modules" "$C/node_modules"

# save the script below as "$C/build.cjs", then:
node "$C/build.cjs"                                   # compiled 207 modules into …/build

# working smoke, both runtimes, from the same precompiled tree
node "$C/build/rl/node/worker.cjs" --smoke            # {"ok":true,"decisions":111,…}
bun  "$C/build/rl/node/worker.cjs" --smoke            # identical payload

# native Bun on raw source: expected stage-3 failure
bun  "$C/src/rl/node/worker.cjs" --smoke
```

`build.cjs` (complete, runnable; `SRC`/`OUT` are the `src/` and `build/` siblings shown
above, and it was executed verbatim to produce the results in this memo):

```js
// Babel AOT build: mirrored CommonJS ES5 tree, using the repo's own .babelrc.
// Usage: node build.cjs   (expects src/ next to this file)
const fs = require('fs');
const path = require('path');
const babel = require('/home/odnakov/Projects/evolution-web/node_modules/@babel/core');

const ROOT = __dirname;
const SRC = path.join(ROOT, 'src');
const OUT = path.join(ROOT, 'build');
const babelrc = JSON.parse(fs.readFileSync(path.join(SRC, '.babelrc'), 'utf8'));

// babel-plugin-root-import must point at the OUTPUT root: the mirrored tree has
// the same depth as src/, so the emitted relative paths stay valid.
const plugins = (babelrc.plugins || []).map((p) => {
  const name = typeof p === 'string' ? p : p[0];
  return name === 'babel-plugin-root-import'
    ? ['babel-plugin-root-import', {root: OUT}]
    : p;
});

function walk(dir, acc) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(p, acc); else acc.push(p);
  }
  return acc;
}

fs.rmSync(OUT, {recursive: true, force: true});
fs.cpSync(SRC, OUT, {recursive: true});

let compiled = 0;
for (const file of walk(OUT, [])) {
  if (file.endsWith('.js')) {
    const res = babel.transformFileSync(file, {
      cwd: OUT, filename: file, babelrc: false, configFile: false,
      presets: babelrc.presets, plugins,
    });
    fs.writeFileSync(file, res.code);
    compiled += 1;
  } else if (file.endsWith('worker.cjs')) {
    // Already CommonJS after the build; the runtime hook would only re-transform it.
    const text = fs.readFileSync(file, 'utf8')
      .replace('  loadBabelRegister();', '  // loadBabelRegister(): skipped, precompiled build');
    fs.writeFileSync(file, text);
  }
}
console.log('compiled', compiled, 'modules into', OUT);
```

The script’s Babel dependency path is absolute on purpose so the temp tree needs no
`node_modules` of its own beyond the symlink the worker already uses.

Timing runs used a separate ~200-line Python harness (`harness.py` + `driver.py`) that speaks
the JSONL protocol directly: fresh process → `hello` (startup), then `reset` and `step` with
a runtime-independent choice policy, capturing per-request wall time, frame hashes, replay
payloads and snapshot/restore behaviour. It is probe code and is not shipped; the protocol it
depends on is `rl/schema.json`.
