"""Rollout CLI and matrix planning tests; no engine subprocess is spawned."""

from __future__ import annotations

import json

import pytest

from evolution_rl import rollout


def test_parse_seeds_is_inclusive():
    assert rollout.parse_seeds("0:3") == [0, 1, 2, 3]
    assert rollout.parse_seeds("5") == [5]
    with pytest.raises(ValueError):
        rollout.parse_seeds("3:1")


def test_matrix_specs_cover_all_addon_subsets():
    specs = rollout.matrix_specs([0])
    assert len(specs) == 128 * 7
    assert len({tuple(sorted(addons)) for addons, _, _ in specs}) == 128
    assert {players for _, players, _ in specs} == set(range(2, 9))


def test_matrix_full_plan_runs_both_policies():
    tasks = rollout.episode_tasks(matrix="full", seeds="0:1")
    # 128 addon subsets * 7 player counts * 2 seeds * 2 policies.
    assert len(tasks) == 128 * 7 * 2 * 2
    assert {task["policy"] for task in tasks} == {"random", "survival"}
    per_policy = {}
    for task in tasks:
        per_policy[task["policy"]] = per_policy.get(task["policy"], 0) + 1
    assert per_policy == {"random": 128 * 7 * 2, "survival": 128 * 7 * 2}

    configs = {
        (tuple(sorted(task["config"]["addons"])), task["config"]["numPlayers"])
        for task in tasks
    }
    assert len(configs) == 128 * 7


def test_profile_plan_is_sequential():
    tasks = rollout.episode_tasks(
        profile="all", players=8, episodes=3, seed=10, policy="survival"
    )
    assert len(tasks) == 3
    assert [task["seed"] for task in tasks] == [10, 11, 12]
    assert all(task["policy"] == "survival" for task in tasks)
    assert tasks[0]["config"] == {"numPlayers": 8, "addons": list(rollout.ADDONS)}


def test_cli_parses_documented_flags():
    args = rollout.build_parser().parse_args(
        [
            "--profile",
            "all",
            "--players",
            "8",
            "--episodes",
            "20",
            "--seed",
            "3",
            "--policy",
            "survival",
            "--workers",
            "4",
        ]
    )
    assert args.profile == "all"
    assert args.players == 8
    assert args.episodes == 20
    assert args.seed == 3
    assert args.policy == "survival"
    assert args.workers == 4


def test_cli_dry_run_matrix(capsys):
    code = rollout.main(["--matrix", "full", "--seeds", "0:0", "--dry-run"])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["planned_episodes"] == 128 * 7 * 2
    assert output["per_policy"] == {"random": 128 * 7, "survival": 128 * 7}


def test_summary_separates_policies_and_truncation():
    results = [
        {
            "policy": "random",
            "seed": 0,
            "terminated": True,
            "truncated": False,
            "rewards": {"player_0": 1.0, "player_1": -1.0},
        },
        {
            "policy": "random",
            "seed": 1,
            "terminated": False,
            "truncated": True,
            "truncation_reason": "max_decisions",
            "rewards": {"player_0": 0.0, "player_1": 0.0},
        },
        {
            "policy": "survival",
            "seed": 0,
            "error": {"code": "ENGINE_DEAD", "message": "boom"},
        },
    ]
    summary = rollout.summarize(results, mode="matrix")
    assert summary["counts"] == {
        "episodes": 3,
        "natural_terminal": 1,
        "truncated": 1,
        "errors": 1,
    }
    assert summary["truncation_reasons"] == {"max_decisions": 1}
    assert summary["per_policy"]["random"]["counts"]["natural_terminal"] == 1
    assert summary["per_policy"]["survival"]["counts"]["errors"] == 1
    assert summary["failures"][0]["error"]["code"] == "ENGINE_DEAD"
