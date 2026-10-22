"""Self-play rollout driver for the Evolution RL environment.

Runs complete AEC episodes with one of the baseline policies, optionally in
parallel worker environments.  The engine is never shared between threads:
each in-flight episode owns its own ``worker.cjs`` subprocess.

Output is a single JSON document on stdout so the driver can be scripted.
Truncations and engine errors are reported as such and never counted as a
natural terminal.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .policies import make_policy

__all__ = [
    "ADDONS",
    "PROFILES",
    "build_parser",
    "episode_tasks",
    "matrix_specs",
    "play_episode",
    "run_tasks",
    "summarize",
]

ADDONS = ("base2", "timeToFly", "continents", "bonus", "plantarium", "customff", "lifecycle")
PROFILES = {
    "base": (),
    "all": ADDONS,
}
POLICIES = ("random", "survival")


# ------------------------------------------------------------------- engine
def _raw(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def engine_catalog(env: Any) -> Any:
    """Public catalog from the engine ``hello`` handshake, when available."""
    raw = _raw(env)
    engine = getattr(raw, "_engine", None)
    if engine is None:
        return None
    try:
        hello = engine.hello()
    except Exception:
        return None
    if isinstance(hello, Mapping):
        return hello.get("catalog")
    return None


def _policy_rng(seed: int | None) -> np.random.Generator:
    base = 0 if seed is None else int(seed)
    return np.random.default_rng(np.random.SeedSequence([base & 0xFFFFFFFF, 0x5EED]))


# ------------------------------------------------------------------ episode
def play_episode(
    env: Any,
    policy_name: str,
    seed: int | None,
    scenario: str | None = None,
    catalog: Any = None,
) -> dict[str, Any]:
    """Play one complete episode in an already-created AEC env."""
    start = time.perf_counter()
    options = {"scenario": scenario} if scenario is not None else None
    env.reset(seed=seed, options=options)

    if catalog is None:
        catalog = engine_catalog(env)

    seats = list(getattr(env, "possible_agents", []) or [])
    if not seats:
        seats = list(getattr(env, "agents", []) or [])
    policy = make_policy(policy_name, rng=_policy_rng(seed), catalog=catalog)

    rewards: dict[str, float] = {seat: 0.0 for seat in seats}
    decisions = 0
    committed = 0
    prefix_choices = 0
    rejected_attempts = 0
    outcome: Any = None
    truncation_reason: str | None = None
    truncated = False
    terminated = False
    observation_bytes = 0

    while getattr(env, "agents", []):
        agent = getattr(env, "agent_selection", None)
        if agent is None:
            break
        if env.terminations.get(agent, False) or env.truncations.get(agent, False):
            env.step(None)
            continue

        observation = env.observe(agent)
        observation_bytes += len(json.dumps(observation, separators=(",", ":"), default=str))
        index = policy.select(observation)
        prefix_choices += 1
        decisions += 1

        live_agents = list(env.agents)
        env.step(index)

        for seat in live_agents:
            rewards[seat] = rewards.get(seat, 0.0) + float(env.rewards.get(seat, 0.0))

        info = env.infos.get(agent) or {}
        if info.get("committed"):
            committed += 1
        if info.get("attempt_rejected"):
            rejected_attempts += 1

        for seat in list(env.agents):
            seat_info = env.infos.get(seat) or {}
            if seat_info.get("outcome") is not None:
                outcome = seat_info["outcome"]
            if seat_info.get("truncation_reason") is not None:
                truncation_reason = str(seat_info["truncation_reason"])

        if env.terminations and all(env.terminations.values()):
            terminated = True
        if env.truncations and any(env.truncations.values()):
            truncated = True

    elapsed = time.perf_counter() - start
    if not terminated and not truncated:
        truncated = True
        truncation_reason = truncation_reason or "episode_ended_without_terminal"

    winner = None
    if isinstance(outcome, Mapping):
        winner = outcome.get("winner")

    return {
        "policy": policy_name,
        "seed": seed,
        "seats": seats,
        "decisions": decisions,
        "prefix_choices": prefix_choices,
        "committed": committed,
        "attempts_rejected": rejected_attempts,
        "terminated": bool(terminated and not truncated),
        "truncated": bool(truncated),
        "truncation_reason": truncation_reason if truncated else None,
        "winner": winner,
        "rewards": {seat: float(value) for seat, value in rewards.items()},
        "outcome": outcome if isinstance(outcome, Mapping) else None,
        "observation_bytes": observation_bytes,
        "duration_seconds": elapsed,
    }


# -------------------------------------------------------------------- plans
def _config(profile: str, players: int) -> dict[str, Any]:
    return {"numPlayers": int(players), "addons": list(PROFILES[profile])}


def matrix_specs(seeds: Sequence[int]) -> list[tuple[list[str], int, int]]:
    """All addon subsets × players 2..8 × seeds, one episode per tuple."""
    specs: list[tuple[list[str], int, int]] = []
    for size in range(len(ADDONS) + 1):
        for subset in itertools.combinations(ADDONS, size):
            for players in range(2, 9):
                for seed in seeds:
                    specs.append((list(subset), players, seed))
    return specs


def episode_tasks(
    *,
    profile: str = "base",
    players: int = 2,
    episodes: int = 1,
    seed: int = 0,
    policy: str = "random",
    matrix: str | None = None,
    seeds: str | None = None,
    scenario: str | None = None,
    policies: Sequence[str] = POLICIES,
) -> list[dict[str, Any]]:
    """Build the deterministic task list for one CLI invocation."""
    tasks: list[dict[str, Any]] = []

    if matrix is not None:
        if matrix != "full":
            raise ValueError(f"unknown matrix {matrix!r}")
        if seeds is None:
            raise ValueError("--matrix full requires --seeds A:B")
        seed_values = parse_seeds(seeds)
        for addons, player_count, seed_value in matrix_specs(seed_values):
            for policy_name in policies:
                tasks.append(
                    {
                        "config": {"numPlayers": player_count, "addons": list(addons)},
                        "policy": policy_name,
                        "seed": seed_value,
                        "scenario": scenario,
                    }
                )
        return tasks

    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    if not 2 <= players <= 8:
        raise ValueError("--players must be between 2 and 8")
    if episodes < 1:
        raise ValueError("--episodes must be positive")
    config = _config(profile, players)
    for seed_value in range(int(seed), int(seed) + int(episodes)):
        tasks.append(
            {
                "config": dict(config),
                "policy": policy,
                "seed": seed_value,
                "scenario": scenario,
            }
        )
    return tasks


def parse_seeds(value: str) -> list[int]:
    """Parse an inclusive ``A:B`` seed range (or a single integer)."""
    text = str(value).strip()
    if ":" not in text:
        return [int(text)]
    start_text, end_text = text.split(":", 1)
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"--seeds range must be ascending, got {value!r}")
    return list(range(start, end + 1))


# ---------------------------------------------------------------- execution
def _run_task(task: Mapping[str, Any], catalog_cache: Any = None) -> dict[str, Any]:
    from . import raw_env

    env = None
    try:
        env = raw_env(config=dict(task["config"]))
        return play_episode(
            env,
            str(task["policy"]),
            task["seed"],
            scenario=task.get("scenario"),
            catalog=catalog_cache,
        )
    except Exception as exc:  # engine/protocol failures are reported, not hidden
        return {
            "policy": task.get("policy"),
            "seed": task.get("seed"),
            "config": dict(task.get("config") or {}),
            "error": {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
            },
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def run_tasks(tasks: Sequence[Mapping[str, Any]], workers: int = 1) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if workers == 1 or len(tasks) <= 1:
        return [_run_task(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(_run_task, task) for task in tasks]
        for future in futures:
            results.append(future.result())
    return results


# ------------------------------------------------------------------ summary
def _empty_counts() -> dict[str, int]:
    return {"episodes": 0, "natural_terminal": 0, "truncated": 0, "errors": 0}


def summarize(results: Sequence[Mapping[str, Any]], mode: str = "rollout") -> dict[str, Any]:
    counts = _empty_counts()
    truncation_reasons: dict[str, int] = {}
    per_policy: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    total_reward = 0.0

    for result in results:
        policy_name = str(result.get("policy") or "unknown")
        bucket = per_policy.setdefault(
            policy_name,
            {"counts": _empty_counts(), "truncation_reasons": {}, "reward_sum": 0.0},
        )
        counts["episodes"] += 1
        bucket["counts"]["episodes"] += 1

        if "error" in result:
            counts["errors"] += 1
            bucket["counts"]["errors"] += 1
            failures.append(
                {
                    "policy": policy_name,
                    "seed": result.get("seed"),
                    "config": result.get("config"),
                    "error": result.get("error"),
                }
            )
            continue

        if result.get("truncated"):
            counts["truncated"] += 1
            bucket["counts"]["truncated"] += 1
            reason = str(result.get("truncation_reason") or "unknown")
            truncation_reasons[reason] = truncation_reasons.get(reason, 0) + 1
            bucket["truncation_reasons"][reason] = (
                bucket["truncation_reasons"].get(reason, 0) + 1
            )
        elif result.get("terminated"):
            counts["natural_terminal"] += 1
            bucket["counts"]["natural_terminal"] += 1

        rewards = result.get("rewards") or {}
        reward_sum = sum(float(value) for value in rewards.values())
        total_reward += reward_sum
        bucket["reward_sum"] += reward_sum

    return {
        "mode": mode,
        "counts": counts,
        "truncation_reasons": truncation_reasons,
        "reward_sum": total_reward,
        "per_policy": per_policy,
        "failures": failures[:20],
    }


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evolution_rl.rollout",
        description="Run baseline self-play rollouts for the Evolution engine.",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="base")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=list(POLICIES), default="random")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix", choices=["full"], default=None)
    parser.add_argument("--seeds", default=None, help="inclusive seed range A:B")
    parser.add_argument("--scenario", default=None, help="YAML scenario file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned episode count without starting engines",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario_text = None
    if args.scenario:
        scenario_text = Path(args.scenario).read_text(encoding="utf-8")

    tasks = episode_tasks(
        profile=args.profile,
        players=args.players,
        episodes=args.episodes,
        seed=args.seed,
        policy=args.policy,
        matrix=args.matrix,
        seeds=args.seeds,
        scenario=scenario_text,
    )

    if args.dry_run:
        policies = sorted({task["policy"] for task in tasks})
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "matrix": args.matrix,
                    "planned_episodes": len(tasks),
                    "per_policy": {
                        policy: sum(1 for task in tasks if task["policy"] == policy)
                        for policy in policies
                    },
                    "workers": args.workers,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    results = run_tasks(tasks, workers=args.workers)
    summary = summarize(results, mode="matrix" if args.matrix else "rollout")
    summary["profile"] = args.profile
    summary["planned_episodes"] = len(tasks)
    summary["workers"] = args.workers
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["counts"]["errors"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
