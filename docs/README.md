# Documentation

- [`rl/plan.md`](rl/plan.md) — approved full plan for the RL environment (scope, constraints, waves, acceptance gates).
- [`rl/implementation-notes.md`](rl/implementation-notes.md) — what was built, verification results, engine defects fixed, known gaps.
- [`research/README.md`](research/README.md) — index of architecture, training, framework and evaluation research.
- [`research/architecture.md`](research/architecture.md) — model architecture study.
- [`research/training.md`](research/training.md) — training algorithm and self-play study.
- [`research/stack.md`](research/stack.md) — framework integration study.
- [`research/evaluation.md`](research/evaluation.md) — evaluation protocol and statistics.

Machine-readable contracts live next to the code: `rl/schema.json` (observation,
Frame, JSONL protocol) and `rl/python/evolution_rl/spaces.py` (Gymnasium spaces
derived from it).
