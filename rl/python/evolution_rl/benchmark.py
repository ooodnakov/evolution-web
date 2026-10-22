"""Throughput/memory benchmark for the Evolution RL engine.

Measures only what the transport exposes: cold startup, reset latency,
committed actions and prefix choices, policy selection cost, engine RSS and
the size of the observation JSON.  No rule is changed and no action domain is
narrowed for the numbers.

The output is one JSON document on stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .policies import make_policy

__all__ = ["ADDONS", "build_parser", "measure_config", "main"]

ADDONS = ("base2", "timeToFly", "continents", "bonus", "plantarium", "customff", "lifecycle")


def _rss_bytes(env: Any) -> int | None:
    raw = getattr(env, "unwrapped", env)
    engine = getattr(raw, "_engine", None)
    process = getattr(engine, "process", None)
    pid = getattr(process, "pid", None)
    if pid is None:
        return None
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def _engine_catalog(env: Any) -> Any:
    raw = getattr(env, "unwrapped", env)
    engine = getattr(raw, "_engine", None)
    if engine is None:
        return None
    try:
        hello = engine.hello()
    except Exception:
        return None
    return hello.get("catalog") if isinstance(hello, Mapping) else None


def _worker(
    config: Mapping[str, Any],
    policy_name: str,
    seeds: Sequence[int],
    scenario: str | None,
) -> dict[str, Any]:
    from . import raw_env

    seed_list = list(seeds)
    env = None
    result: dict[str, Any] = {
        "cold_startup_seconds": None,
        "reset_seconds": [],
        "steps": 0,
        "committed": 0,
        "choices": 0,
        "select_seconds": 0.0,
        "observation_bytes": 0,
        "episodes": 0,
        "terminals": 0,
        "truncations": 0,
        "rss_bytes": None,
        "wall_seconds": 0.0,
    }
    try:
        started = time.perf_counter()
        env = raw_env(config=dict(config))
        catalog = None

        if seed_list:
            first_reset = time.perf_counter()
            env.reset(seed=seed_list[0], options={"scenario": scenario} if scenario else None)
            result["cold_startup_seconds"] = time.perf_counter() - started
            result["reset_seconds"].append(time.perf_counter() - first_reset)
            catalog = _engine_catalog(env)
            measured = _replay_after_reset(env, policy_name, catalog, seed_list[0])
            _accumulate(result, measured)
            result["episodes"] += 1
            result["terminals"] += 1 if measured["terminated"] else 0
            result["truncations"] += 1 if measured["truncated"] else 0

        for seed in seed_list[1:]:
            reset_started = time.perf_counter()
            env.reset(seed=seed, options={"scenario": scenario} if scenario else None)
            result["reset_seconds"].append(time.perf_counter() - reset_started)
            measured = _replay_after_reset(env, policy_name, catalog, seed)
            _accumulate(result, measured)
            result["episodes"] += 1
            result["terminals"] += 1 if measured["terminated"] else 0
            result["truncations"] += 1 if measured["truncated"] else 0

        result["rss_bytes"] = _rss_bytes(env)
        result["wall_seconds"] = time.perf_counter() - started
        return result
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _replay_after_reset(
    env: Any, policy_name: str, catalog: Any, seed: int | None = None
) -> dict[str, Any]:
    """Play one episode in an env that is already reset (no extra reset)."""
    from .policies import make_policy as _factory  # local import keeps module light

    rng = np.random.default_rng(
        np.random.SeedSequence([0 if seed is None else int(seed), 0x5EED])
    )
    policy = _factory(policy_name, rng=rng, catalog=catalog)
    steps = committed = choices = observation_bytes = 0
    select_seconds = 0.0
    terminated = truncated = False

    while getattr(env, "agents", []):
        agent = getattr(env, "agent_selection", None)
        if agent is None:
            break
        if env.terminations.get(agent, False) or env.truncations.get(agent, False):
            env.step(None)
            continue
        observation = env.observe(agent)
        observation_bytes += len(
            json.dumps(observation, separators=(",", ":"), default=str)
        )
        choices += len(observation.get("choices") or [])
        started = time.perf_counter()
        index = policy.select(observation)
        select_seconds += time.perf_counter() - started
        steps += 1
        env.step(index)
        info = env.infos.get(agent) or {}
        if info.get("committed"):
            committed += 1
        if env.terminations and all(env.terminations.values()):
            terminated = True
        if env.truncations and any(env.truncations.values()):
            truncated = True

    return {
        "steps": steps,
        "committed": committed,
        "choices": choices,
        "select_seconds": select_seconds,
        "observation_bytes": observation_bytes,
        "terminated": terminated and not truncated,
        "truncated": truncated,
    }


def _accumulate(result: dict[str, Any], measured: Mapping[str, Any]) -> None:
    for key in ("steps", "committed", "choices", "observation_bytes"):
        result[key] += measured[key]
    result["select_seconds"] += measured["select_seconds"]


def measure_config(
    *,
    name: str,
    profile: str,
    players: int,
    addons: Sequence[str],
    episodes: int,
    workers: int,
    policy: str = "random",
    scenario: str | None = None,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    seed_values = list(range(episodes)) if seeds is None else list(seeds)
    config = {"numPlayers": int(players), "addons": list(addons)}

    chunks = [seed_values[index::workers] for index in range(workers)]
    chunks = [chunk for chunk in chunks if chunk]
    started = time.perf_counter()
    if workers == 1:
        worker_results = [_worker(config, policy, seed_values, scenario)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_worker, config, policy, chunk, scenario) for chunk in chunks
            ]
            worker_results = [future.result() for future in futures]
    wall = time.perf_counter() - started

    reset_times = [value for result in worker_results for value in result["reset_seconds"]]
    steps = sum(result["steps"] for result in worker_results)
    committed = sum(result["committed"] for result in worker_results)
    choices = sum(result["choices"] for result in worker_results)
    select_seconds = sum(result["select_seconds"] for result in worker_results)
    observation_bytes = sum(result["observation_bytes"] for result in worker_results)
    rss_values = [result["rss_bytes"] for result in worker_results if result["rss_bytes"]]
    cold_values = [
        result["cold_startup_seconds"]
        for result in worker_results
        if result["cold_startup_seconds"] is not None
    ]

    return {
        "name": name,
        "profile": profile,
        "players": int(players),
        "addons": list(addons),
        "policy": policy,
        "episodes": sum(result["episodes"] for result in worker_results),
        "workers": int(workers),
        "wall_seconds": wall,
        "cold_startup_seconds": max(cold_values) if cold_values else None,
        "reset_seconds": {
            "median": float(statistics.median(reset_times)) if reset_times else 0.0,
            "p95": _percentile(reset_times, 0.95),
            "samples": len(reset_times),
        },
        "steps": steps,
        "committed_actions": committed,
        "committed_actions_per_second": committed / wall if wall > 0 else 0.0,
        "prefix_choices_per_second": steps / wall if wall > 0 else 0.0,
        "candidate_domain": {
            "average_choices_per_step": choices / steps if steps else 0.0,
            "average_select_seconds": select_seconds / steps if steps else 0.0,
            "choices_per_second": choices / wall if wall > 0 else 0.0,
        },
        "json_bytes_per_step": observation_bytes / steps if steps else 0.0,
        "rss_bytes_per_env": {
            "avg": int(sum(rss_values) / len(rss_values)) if rss_values else None,
            "max": max(rss_values) if rss_values else None,
        },
        "terminals": sum(result["terminals"] for result in worker_results),
        "truncations": sum(result["truncations"] for result in worker_results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evolution_rl.benchmark",
        description="Measure Evolution RL engine throughput and memory.",
    )
    parser.add_argument("--profile", choices=["base", "all", "both"], default="both")
    parser.add_argument("--players", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", default="1,4")
    parser.add_argument("--policy", choices=["random", "survival"], default="random")
    parser.add_argument("--scenario", default=None)
    return parser


def _parse_workers(value: str) -> list[int]:
    result = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        workers = int(part)
        if workers < 1:
            raise ValueError("workers must be at least 1")
        result.append(workers)
    if not result:
        raise ValueError("--workers must list at least one value")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = None
    if args.scenario:
        scenario = Path(args.scenario).read_text(encoding="utf-8")

    if args.profile == "both":
        configs = [
            ("base2p", "base", 2, ()),
            ("all8p", "all", 8, ADDONS),
        ]
    elif args.profile == "base":
        configs = [("base2p", "base", args.players or 2, ())]
    else:
        configs = [("all8p", "all", args.players or 8, ADDONS)]

    measurements = []
    workers_list = _parse_workers(args.workers)
    for name, profile, players, addons in configs:
        if not 2 <= players <= 8:
            raise ValueError("players must be between 2 and 8")
        for workers in workers_list:
            measurements.append(
                measure_config(
                    name=name,
                    profile=profile,
                    players=players,
                    addons=addons,
                    episodes=args.episodes,
                    workers=workers,
                    policy=args.policy,
                    scenario=scenario,
                )
            )

    print(
        json.dumps(
            {
                "configs": measurements,
                "workers_compared": workers_list,
                "episodes_per_config": args.episodes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
