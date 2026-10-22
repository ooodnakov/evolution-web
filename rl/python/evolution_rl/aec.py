"""PettingZoo AEC adapter for the headless Evolution engine.

The Node worker owns all rules. This class only:

* keeps the PettingZoo bookkeeping (agent selection, rewards, terminations),
* forwards one choice index per decision,
* exposes a stable observation space and a per-decision action space.

Variable populations are represented by ``gymnasium.spaces.Sequence`` for
observations and by a fresh ``Discrete(len(choices))`` per decision for
actions, so a fixed action head is intentionally not provided.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces as gym_spaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import AECEnv

from . import bridge
from .bridge import EngineProcess, EngineProcessError

__all__ = [
    "EvolutionAECEnv",
    "DEFAULT_CONFIG",
    "load_default_config",
]

_SPACE_SEED_OFFSET = 0x9E3779B9
_MAX_INT32 = 2**31 - 1

_SCHEMA_CACHE: dict[str, Any] | None = None


def _find_schema_path() -> Path | None:
    here = Path(__file__).resolve()
    candidates = [parent / bridge.SCHEMA_RELPATH for parent in here.parents]
    cwd = Path.cwd().resolve()
    candidates.extend(parent / bridge.SCHEMA_RELPATH for parent in (cwd, *cwd.parents))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        path = _find_schema_path()
        if path is None:
            raise RuntimeError("rl/schema.json is required to configure the Evolution env")
        with open(path, encoding="utf-8") as schema_file:
            _SCHEMA_CACHE = json.load(schema_file)
    return _SCHEMA_CACHE


def load_default_config() -> dict[str, Any]:
    """Return a fresh copy of ``rl/schema.json -> config``."""
    return copy.deepcopy(dict(_schema()["config"]))


DEFAULT_CONFIG: dict[str, Any] = load_default_config()


def _merged_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = load_default_config()
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        merged.update(dict(config))
    num_players = merged.get("numPlayers", 2)
    if not isinstance(num_players, int) or isinstance(num_players, bool):
        raise TypeError("config.numPlayers must be an integer")
    if not 2 <= num_players <= 8:
        raise ValueError("config.numPlayers must be between 2 and 8")
    return merged


class EvolutionAECEnv(AECEnv):
    """AEC environment backed by one long-lived ``rl/node/worker.cjs``."""

    metadata = {
        "name": "evolution_aec",
        "render_modes": ["ansi"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        render_mode: str | None = None,
        node_executable: str = "node",
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"unsupported render_mode {render_mode!r}; "
                f"supported: {self.metadata['render_modes']}"
            )
        self.config = _merged_config(config)
        self.render_mode = render_mode
        self.node_executable = node_executable

        num_players = int(self.config["numPlayers"])
        self.possible_agents = [f"player_{seat}" for seat in range(num_players)]
        self.agents = list(self.possible_agents)
        self.agent_selection = None

        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._cumulative_rewards: dict[str, float] = {}

        self._engine: EngineProcess | None = None
        self._decision_id: int | None = None
        self._observations: dict[str, Any] = {}
        self._snapshots: dict[int, dict[str, Any]] = {}
        self._observation_space: gym_spaces.Space[Any] | None = None
        self._action_spaces: dict[str, gym_spaces.Space[Any]] = {}
        self._action_spaces_decision: int | None = None
        self._space_rng: np.random.Generator
        self._space_rng, _ = seeding.np_random(None)

    # ------------------------------------------------------------ plumbing
    def _ensure_engine(self) -> EngineProcess:
        if self._engine is None:
            self._engine = EngineProcess(
                node_executable=self.node_executable,
                config=self.config,
            )
        return self._engine

    def _require_engine(self) -> EngineProcess:
        if self._engine is None:
            raise RuntimeError("reset() must be called before talking to the engine")
        return self._engine

    def _first_live_agent(self) -> str | None:
        for agent in self.possible_agents:
            if agent in self.agents:
                return agent
        return None

    def _make_info(self, frame: Mapping[str, Any], agent: str) -> dict[str, Any]:
        info_map = frame.get("info") or {}
        info = dict(info_map.get(agent) or {}) if isinstance(info_map, Mapping) else {}
        info.setdefault("committed", bool(frame.get("committed", False)))
        return info

    def _cache_observations(self, frame: Mapping[str, Any]) -> None:
        observations = frame.get("observations") or {}
        if isinstance(observations, Mapping):
            self._observations = dict(observations)

    # --------------------------------------------------------------- reset
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> None:
        engine = self._ensure_engine()

        scenario: str | None = None
        if isinstance(options, Mapping):
            candidate = options.get("scenario")
            if candidate is not None:
                if not isinstance(candidate, str):
                    raise TypeError("options['scenario'] must be a YAML string")
                scenario = candidate

        node_seed: int | None = None
        if seed is not None:
            _, node_seed = seeding.np_random(seed)
            node_seed = int(node_seed)
            if not 0 <= node_seed <= 0xFFFFFFFF:
                raise ValueError(
                    f"seed must fit an unsigned 32-bit integer, got {node_seed}"
                )
        space_seed = (
            None
            if node_seed is None
            else (node_seed + _SPACE_SEED_OFFSET) % (2**32)
        )
        self._space_rng, _ = seeding.np_random(space_seed)

        frame = engine.reset(seed=node_seed, scenario=scenario)
        self._reset_local_state(frame)
        return None

    def _reset_local_state(self, frame: Mapping[str, Any]) -> None:
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: self._make_info(frame, agent) for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        terminated = frame.get("terminated") or {}
        truncated = frame.get("truncated") or {}
        for agent in self.agents:
            self.terminations[agent] = bool(terminated.get(agent, False))
            self.truncations[agent] = bool(truncated.get(agent, False))

        self._cache_observations(frame)
        self._decision_id = frame.get("decision_id")
        actor = frame.get("actor")
        self.agent_selection = actor if actor is not None else self._first_live_agent()
        self._snapshots = {}
        self._skip_agent_selection = None
        self._clear_action_spaces()

    # ----------------------------------------------------------------- step
    def step(self, action: Any) -> None:
        if self.agent_selection is None:
            raise ValueError("agent_selection is None; reset() the environment first")

        agent = self.agent_selection
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            if action is not None:
                raise ValueError("a terminated/truncated agent accepts only None")
            self._was_dead_step(None)
            return

        if action is None:
            raise ValueError("action None is only valid for a terminated/truncated agent")
        try:
            choice_index = int(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action must be an integer choice index, got {action!r}") from exc
        if choice_index < 0:
            raise ValueError(f"action must be non-negative, got {choice_index}")

        engine = self._require_engine()
        if self._decision_id is None:
            raise RuntimeError("environment has no active decision; call reset()")

        self._cumulative_rewards[agent] = 0.0
        frame = engine.step(self._decision_id, choice_index)
        self._apply_step_frame(frame)

    def _apply_step_frame(self, frame: Mapping[str, Any]) -> None:
        self._cache_observations(frame)
        new_decision_id = frame.get("decision_id")
        if new_decision_id != self._decision_id:
            # Only a genuinely new decision invalidates cached action spaces;
            # a rejected attempt keeps the same decision token.
            self._decision_id = new_decision_id
            self._clear_action_spaces()
        else:
            self._decision_id = new_decision_id

        instantaneous = frame.get("rewards") or {}
        self.rewards = {
            agent: float(instantaneous.get(agent, 0.0)) for agent in self.agents
        }

        terminated = frame.get("terminated") or {}
        truncated = frame.get("truncated") or {}
        for agent in self.agents:
            if agent in terminated:
                self.terminations[agent] = bool(terminated[agent])
            if agent in truncated:
                self.truncations[agent] = bool(truncated[agent])
        self.infos = {agent: self._make_info(frame, agent) for agent in self.agents}

        self._accumulate_rewards()

        actor = frame.get("actor")
        if actor is None:
            # Terminal / truncated: seat_0 first, then every seat walks the
            # dead-step path so each gets its terminal cumulative reward once.
            self.agent_selection = self._first_live_agent()
        else:
            if actor not in self.agents:
                raise EngineProcessError(
                    "ENGINE_PROTOCOL", f"actor {actor!r} is not an active agent"
                )
            self.agent_selection = actor

    # ------------------------------------------------------------ observing
    def observe(self, agent: str) -> Any:
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        if agent in self._observations:
            return self._observations[agent]
        engine = self._engine
        if engine is None:
            raise RuntimeError("reset() must be called before observe()")
        observation = engine.observe(agent)
        self._observations[agent] = observation
        return observation

    def observation_space(self, agent: str) -> gym_spaces.Space[Any]:
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        if self._observation_space is None:
            from . import spaces as evolution_spaces

            self._observation_space = evolution_spaces.observation_space()
        return self._observation_space

    # ------------------------------------------------------------ action sp
    def _clear_action_spaces(self) -> None:
        self._action_spaces = {}
        self._action_spaces_decision = self._decision_id

    def _build_action_space(self, count: int) -> gym_spaces.Space[Any]:
        try:
            from . import spaces as evolution_spaces

            space = evolution_spaces.change_space(count)
        except (ImportError, AttributeError):  # pragma: no cover
            space = gym_spaces.Discrete(max(1, int(count)))
        if self._space_rng is not None:
            space.seed(int(self._space_rng.integers(0, _MAX_INT32, endpoint=False)))
        return space

    def action_space(self, agent: str) -> gym_spaces.Space[Any]:
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        if self._action_spaces_decision != self._decision_id:
            self._clear_action_spaces()

        space = self._action_spaces.get(agent)
        if space is None:
            observation = self._observations.get(agent)
            choices = observation.get("choices") if isinstance(observation, Mapping) else None
            count = len(choices) if choices is not None else 1
            space = self._build_action_space(count)
            self._action_spaces[agent] = space
        return space

    # ------------------------------------------------------------ rendering
    def render(self, mode: str | None = None) -> str:
        active_mode = mode if mode is not None else (self.render_mode or "ansi")
        if active_mode != "ansi":
            raise ValueError(f"unsupported render mode {active_mode!r}")
        if self.agent_selection is not None:
            return self._render_agent(self.agent_selection)
        if self.possible_agents:
            return self._render_agent(self.possible_agents[0])
        return "Evolution RL: no agents"

    def _render_agent(self, agent: str) -> str:
        observation = self._observations.get(agent)
        if not isinstance(observation, Mapping):
            return f"Evolution RL: no observation for {agent}"

        lines: list[str] = []
        global_view = observation.get("global") or {}
        seat = global_view.get("seat")
        lines.append(
            "Evolution RL | phase={phase} turn={turn} round={round} "
            "food={food} deck={deck} active={active}".format(
                phase=global_view.get("phase"),
                turn=global_view.get("turn"),
                round=global_view.get("round"),
                food=global_view.get("food"),
                deck=global_view.get("deck_count"),
                active=global_view.get("active_players"),
            )
        )
        lines.append(f"agent: {agent} (seat {seat})")

        players = observation.get("players") or []
        player_bits = []
        for player in players:
            if not isinstance(player, Mapping):
                continue
            player_bits.append(
                "{seat}:{state} hand={hand} animals={animals}".format(
                    seat=player.get("seat"),
                    state=(
                        "out"
                        if player.get("ended")
                        else ("live" if player.get("playing") else "waiting")
                    ),
                    hand=player.get("hand_count"),
                    animals=player.get("animal_count"),
                )
            )
        if player_bits:
            lines.append("players: " + " | ".join(player_bits))

        hand = observation.get("hand") or []
        hand_bits = [
            "{c}/{t1}/{t2}".format(
                c=card.get("card_type"),
                t1=card.get("trait1_type"),
                t2=card.get("trait2_type"),
            )
            for card in hand
            if isinstance(card, Mapping)
        ]
        lines.append("your hand: " + (", ".join(hand_bits) if hand_bits else "(empty)"))

        animals = observation.get("animals") or []
        animal_bits = []
        for animal in animals:
            if not isinstance(animal, Mapping):
                continue
            flags = animal.get("flags") or []
            animal_bits.append(
                "seat{owner}#{position} food={food}/{food_size} fat={fat_size} "
                "flags={flags} traits={traits}".format(
                    owner=animal.get("owner"),
                    position=animal.get("position"),
                    food=animal.get("food"),
                    food_size=animal.get("food_size"),
                    fat_size=animal.get("fat_size"),
                    flags=",".join(str(flag) for flag in flags) or "-",
                    traits=len(animal.get("traits") or []),
                )
            )
        if animal_bits:
            lines.append("animals: " + " ; ".join(animal_bits))

        plants = observation.get("plants") or []
        plant_bits = []
        for plant in plants:
            if not isinstance(plant, Mapping):
                continue
            plant_bits.append(
                "type={type} food={food} covers={covers}/{cover_slots} fruit={fruit}".format(
                    type=plant.get("type"),
                    food=plant.get("food"),
                    covers=plant.get("covers"),
                    cover_slots=plant.get("cover_slots"),
                    fruit=plant.get("fruit"),
                )
            )
        if plant_bits:
            lines.append("plants: " + " ; ".join(plant_bits))

        context = observation.get("context") or {}
        choices = observation.get("choices") or []
        lines.append(
            "decision: kind={kind} choices={choices}".format(
                kind=context.get("kind"), choices=len(choices)
            )
        )
        return "\n".join(lines)

    # ------------------------------------------------------------ extension
    def snapshot(self) -> int:
        engine = self._require_engine()
        handle = int(engine.snapshot())
        self._snapshots[handle] = {
            "agents": list(self.agents),
            "agent_selection": self.agent_selection,
            "rewards": dict(self.rewards),
            "terminations": dict(self.terminations),
            "truncations": dict(self.truncations),
            "infos": copy.deepcopy(self.infos),
            "cumulative_rewards": dict(self._cumulative_rewards),
            "observations": copy.deepcopy(self._observations),
            "space_rng_state": copy.deepcopy(self._space_rng.bit_generator.state),
            "skip_agent_selection": getattr(self, "_skip_agent_selection", None),
        }
        return handle

    def restore(self, handle: int) -> None:
        engine = self._require_engine()
        local = self._snapshots.get(int(handle))
        if local is None:
            raise EngineProcessError(
                "INVALID_SNAPSHOT", f"unknown snapshot handle {handle!r} for this environment"
            )
        frame = engine.restore(int(handle))
        self._restore_local_state(local, frame)
        return None

    def _restore_local_state(
        self, local: Mapping[str, Any], frame: Mapping[str, Any]
    ) -> None:
        self.agents = list(local["agents"])
        self.agent_selection = local["agent_selection"]
        self.rewards = dict(local["rewards"])
        self.terminations = dict(local["terminations"])
        self.truncations = dict(local["truncations"])
        self.infos = copy.deepcopy(local["infos"])
        self._cumulative_rewards = dict(local["cumulative_rewards"])
        self._observations = copy.deepcopy(local["observations"])

        observations = frame.get("observations") or {}
        if isinstance(observations, Mapping) and observations:
            self._observations.update(dict(observations))

        self._decision_id = frame.get("decision_id")
        actor = frame.get("actor")
        if actor is not None:
            self.agent_selection = actor

        space_state = local.get("space_rng_state")
        if space_state is not None:
            self._space_rng.bit_generator.state = copy.deepcopy(space_state)
        self._skip_agent_selection = local.get("skip_agent_selection")
        self._clear_action_spaces()

    def export_replay(self) -> dict[str, Any]:
        engine = self._require_engine()
        return engine.export_replay()

    def close(self) -> None:
        self._snapshots.clear()
        if self._engine is not None:
            self._engine.close()
            self._engine = None
