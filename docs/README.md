# Documentation

- [`commands.md`](commands.md) — copy-paste recipes for setup, web, RL tests, rollouts, matrix monitoring, benchmark, training, debugging and cleanup.
- [`rl/plan.md`](rl/plan.md) — approved full plan for the RL environment (scope, constraints, waves, acceptance gates).
- [`rl/implementation-notes.md`](rl/implementation-notes.md) — what was built, verification results, engine defects fixed, known gaps.
- [`rl/performance-ideas.md`](rl/performance-ideas.md) — ranked, measured-baseline ideas for speeding up the RL workers.
- [`rl/bun-worker.md`](rl/bun-worker.md) — Bun vs Node research record for the RL worker (now adopted as a compiled-Bun runtime): why the native swap is blocked, the precompiled route, measured timings and determinism parity.
- [`research/README.md`](research/README.md) — index of architecture, training, framework and evaluation research.
- [`research/architecture.md`](research/architecture.md) — model architecture study.
- [`research/training.md`](research/training.md) — training algorithm and self-play study.
- [`research/stack.md`](research/stack.md) — framework integration study.
- [`research/evaluation.md`](research/evaluation.md) — evaluation protocol and statistics.

Machine-readable contracts live next to the code: `rl/schema.json` (observation,
Frame, JSONL protocol) and `rl/python/evolution_rl/spaces.py` (Gymnasium spaces
derived from it).
