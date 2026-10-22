"""Short CPU/GPU REINFORCE proof for the Evolution RL environment.

The smoke run is deliberately small: 32 real base-game episodes (no ``base2``
addon), a shared policy/value network for every seat, and eight evaluation
episodes from a reloaded checkpoint.  It does not claim that the policy became
strong; it proves that legal rollouts produce a usable gradient signal.

The default ``--device auto`` uses CUDA only when ``torch.cuda.is_available()``
is true, otherwise CPU; ``EVOLUTION_RL_DEVICE`` is the fallback when the CLI
value is ``auto``.  The locked CPU wheel remains the default: GPU use requires
a CUDA torch wheel, for example::

    uv pip install --index-url https://download.pytorch.org/whl/cu126 torch

The smoke bottleneck is the Node engine/JSONL transport, not the tiny network.

Usage::

    uv run --locked --extra train python -m evolution_rl.train_smoke \
        --episodes 32 --seed 7 --output rl/runs/smoke
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # torch is an optional extra; importing this module must not require it
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by non-train runs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

__all__ = [
    "BASE_CONFIG",
    "SCENARIO",
    "FeatureSpec",
    "build_model",
    "main",
    "run_smoke",
]

BASE_CONFIG = {
    "numPlayers": 2,
    "addons": [],
    "halfDeck": False,
    "maxCards": None,
    "randomPlayers": False,
}
SCENARIO = """phase: feeding
food: 0
deck: ''
players:
  - continent: $A TraitCarnivorous, $D TraitSwimming +
  - continent: $B TraitPoisonous +, $C +, $E TraitSwimming +
"""

EMBED_DIM = 16
HIDDEN_DIM = 64
GAMMA = 1.0
LEARNING_RATE = 3e-4
ENTROPY_COEFFICIENT = 0.01
CLIP_GRAD_NORM = 1.0
BATCH_EPISODES = 8


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover - environment guard
        raise RuntimeError(
            "train_smoke requires the 'train' extra: uv run --locked --extra train ..."
        )


def _resolve_device(requested: str | None = "auto") -> Any:
    _require_torch()
    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        fallback = os.environ.get("EVOLUTION_RL_DEVICE", "").strip().lower()
        if fallback and fallback != "auto":
            choice = fallback
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device {requested!r}; expected auto, cpu or cuda")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this torch build has no CUDA device; "
            "install a CUDA wheel or use --device cpu"
        )
    return torch.device(choice)


def _schema() -> dict[str, Any]:
    from .spaces import SCHEMA_PATH

    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


SCHEMA = _schema()


def _schema_index(name: str) -> dict[str, int]:
    return {value: index for index, value in enumerate(SCHEMA[name])}


_FIELD = _schema_index("choice_field_kinds")
_DECISION_KIND = _schema_index("decision_kinds")
_ENTITY_KIND = _schema_index("entity_kinds")

FIELD_TYPE = _FIELD["TYPE"]
FIELD_DECLINE = _FIELD["DECLINE"]


# ------------------------------------------------------------------ features
@dataclass(frozen=True)
class FeatureSpec:
    """Static vocabulary sizes; stored in the checkpoint next to the model."""

    schema_version: int
    type_vocab: int
    enums: Mapping[str, int]


def _namespaces(catalog: Any) -> Mapping[str, Any]:
    if not isinstance(catalog, Mapping):
        return {}
    namespaces = catalog.get("namespaces")
    return namespaces if isinstance(namespaces, Mapping) else catalog


def _namespace_size(catalog: Any, name: str, fallback: int) -> int:
    namespace = _namespaces(catalog).get(name)
    if not isinstance(namespace, Mapping) or not namespace:
        return fallback
    highest = max(
        (value for value in namespace.values() if isinstance(value, int)), default=0
    )
    return max(fallback, highest + 1)


def feature_spec(catalog: Any = None) -> FeatureSpec:
    type_vocab = max(
        _namespace_size(catalog, "all_trait_types", 256),
        _namespace_size(catalog, "card_types", 64),
        _namespace_size(catalog, "plant_types", 32),
    )
    enums = {
        "phase": len(SCHEMA["phase_values"]),
        "decision": len(SCHEMA["decision_kinds"]),
        "entity": len(SCHEMA["entity_kinds"]),
        "field": len(SCHEMA["choice_field_kinds"]),
        "choice": len(SCHEMA["choice_kinds"]),
        "value": len(SCHEMA["value_kinds"]),
        "source": len(SCHEMA["feeding_source_types"]),
    }
    return FeatureSpec(
        schema_version=int(SCHEMA["schema_version"]),
        type_vocab=int(type_vocab),
        enums=enums,
    )


def _norm(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number / (1.0 + abs(number))


def _int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        return int(value)
    except (TypeError, ValueError, IndexError):
        return default


def _rows(container: Any, key: str) -> list[Any]:
    if not isinstance(container, Mapping):
        return []
    value = container.get(key)
    return list(value) if isinstance(value, (list, tuple)) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bits(value: Any, length: int) -> list[float]:
    if isinstance(value, np.ndarray):
        flat = [int(item) for item in value.reshape(-1)]
    elif isinstance(value, (list, tuple)):
        flat = []
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                flat.append(_int(item))
            else:
                flat.append(_int(item))
    else:
        flat = [_int(value)]
    flat.extend([0] * (length - len(flat)))
    return [float(bool(item)) for item in flat[:length]]


# -------------------------------------------------------------------- module
if nn is not None:  # pragma: no branch - class definitions need torch

    class RowMlp(nn.Module):
        """``Linear -> Tanh -> Linear -> Tanh`` row encoder."""

        def __init__(self, input_dim: int, hidden: int = HIDDEN_DIM) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class ObservationEncoder(nn.Module):
        def __init__(self, spec: FeatureSpec, device: Any = "cpu") -> None:
            super().__init__()
            self.spec = spec
            self.device = torch.device(device)
            self.type_embed = nn.Embedding(spec.type_vocab, EMBED_DIM)
            self.enum_embed = nn.ModuleDict(
                {
                    name: nn.Embedding(size, EMBED_DIM)
                    for name, size in spec.enums.items()
                }
            )
            self.trait_encoder = RowMlp(4 * EMBED_DIM + 12)
            self.animal_encoder = RowMlp(5 + len(SCHEMA["animal_flags"]) + HIDDEN_DIM)
            self.plant_encoder = RowMlp(EMBED_DIM + 4 + HIDDEN_DIM)
            self.hand_encoder = RowMlp(3 * EMBED_DIM)
            self.shell_encoder = RowMlp(EMBED_DIM + 2)
            self.player_encoder = RowMlp(6)
            self.global_encoder = RowMlp(EMBED_DIM + 9 + len(SCHEMA["addons"]))
            self.context_encoder = RowMlp(4 * EMBED_DIM + 2 + HIDDEN_DIM)
            self.choice_encoder = RowMlp(4 * EMBED_DIM + 3)

        # -- embedding helpers -------------------------------------------
        def _tensor(self, values: Any, dtype: Any = None) -> Any:
            return torch.tensor(values, dtype=dtype, device=self.device)

        def _type(self, value: Any) -> Any:
            index = max(0, min(int(self.spec.type_vocab) - 1, _int(value)))
            return self.type_embed(self._tensor([index], dtype=torch.long)).squeeze(0)

        def _enum(self, name: str, value: Any) -> Any:
            size = max(1, int(self.spec.enums[name]))
            index = max(0, min(size - 1, _int(value)))
            return self.enum_embed[name](self._tensor([index], dtype=torch.long)).squeeze(0)

        def _zeros(self) -> Any:
            return torch.zeros(HIDDEN_DIM, device=self.device)

        def _mean(self, rows: Sequence[Any]) -> Any:
            if not rows:
                return self._zeros()
            return torch.stack(list(rows), dim=0).mean(dim=0)

        # -- rows ---------------------------------------------------------
        def _trait(self, row: Any) -> Any:
            row = _mapping(row)
            cooldowns = _rows(row, "cooldowns")
            cooldown_sum = sum(_norm(_mapping(item).get("remaining")) for item in cooldowns)
            features = torch.cat(
                [
                    self._type(row.get("type")),
                    self._enum("entity", row.get("linked_entity_kind")),
                    self._enum("value", row.get("value_kind")),
                    self._enum("source", row.get("source_type")),
                    self._tensor(
                        [
                            float(bool(_int(row.get("disabled")))),
                            _norm(row.get("covers")),
                            _norm(row.get("link_source")),
                            _norm(row.get("linked_entity_slot")),
                            _norm(row.get("linked_trait_slot")),
                            _norm(row.get("value_number")),
                            _norm(row.get("value_entity_slot")),
                            _norm(row.get("value_trait_slot")),
                            _norm(row.get("source_slot")),
                            _norm(row.get("auto_share")),
                            _norm(len(cooldowns)),
                            _norm(cooldown_sum),
                        ],
                        dtype=torch.float,
                    ),
                ]
            )
            return self.trait_encoder(features)

        def _traits(self, container: Any) -> Any:
            return self._mean([self._trait(row) for row in _rows(container, "traits")])

        def _animal(self, row: Any) -> Any:
            row = _mapping(row)
            flags = _bits(row.get("flags"), len(SCHEMA["animal_flags"]))
            features = torch.cat(
                [
                    self._tensor(
                        [
                            _norm(row.get("owner")),
                            _norm(row.get("position")),
                            _norm(row.get("food")),
                            _norm(row.get("food_size")),
                            _norm(row.get("fat_size")),
                            *flags,
                        ],
                        dtype=torch.float,
                    ),
                    self._traits(row),
                ]
            )
            return self.animal_encoder(features)

        def _plant(self, row: Any) -> Any:
            row = _mapping(row)
            features = torch.cat(
                [
                    self._type(row.get("type")),
                    self._tensor(
                        [
                            _norm(row.get("food")),
                            _norm(row.get("covers")),
                            _norm(row.get("cover_slots")),
                            _norm(row.get("fruit")),
                        ],
                        dtype=torch.float,
                    ),
                    self._traits(row),
                ]
            )
            return self.plant_encoder(features)

        def _hand(self, row: Any) -> Any:
            row = _mapping(row)
            features = torch.cat(
                [
                    self._type(row.get("card_type")),
                    self._type(row.get("trait1_type")),
                    self._type(row.get("trait2_type")),
                ]
            )
            return self.hand_encoder(features)

        def _shell(self, row: Any) -> Any:
            row = _mapping(row)
            features = torch.cat(
                [
                    self._type(row.get("type")),
                    self._tensor(
                        [_norm(row.get("value")), _norm(row.get("slot"))],
                        dtype=torch.float,
                    ),
                ]
            )
            return self.shell_encoder(features)

        def _player(self, row: Any) -> Any:
            row = _mapping(row)
            features = self._tensor(
                [
                    _norm(row.get("seat")),
                    float(bool(_int(row.get("playing")))),
                    float(bool(_int(row.get("ended")))),
                    float(bool(_int(row.get("acted")))),
                    _norm(row.get("hand_count")),
                    _norm(row.get("animal_count")),
                ],
                dtype=torch.float,
            )
            return self.player_encoder(features)

        def _global(self, row: Any) -> Any:
            row = _mapping(row)
            addons = _bits(row.get("addons"), len(SCHEMA["addons"]))
            features = torch.cat(
                [
                    self._enum("phase", row.get("phase")),
                    self._tensor(
                        [
                            _norm(row.get("turn")),
                            _norm(row.get("round")),
                            _norm(row.get("seat")),
                            _norm(row.get("current_player")),
                            _norm(row.get("round_player")),
                            _norm(row.get("food")),
                            _norm(row.get("deck_count")),
                            _norm(row.get("active_players")),
                            float(bool(_int(row.get("half_deck")))),
                            *addons,
                        ],
                        dtype=torch.float,
                    ),
                ]
            )
            return self.global_encoder(features)

        def _choice(self, row: Any) -> Any:
            row = _mapping(row)
            features = torch.cat(
                [
                    self._enum("field", row.get("field_kind")),
                    self._enum("choice", row.get("choice_kind")),
                    self._type(row.get("type_id")),
                    self._enum("entity", row.get("entity_kind")),
                    self._tensor(
                        [
                            _norm(row.get("entity_slot")),
                            _norm(row.get("trait_slot")),
                            _norm(row.get("scalar")),
                        ],
                        dtype=torch.float,
                    ),
                ]
            )
            return self.choice_encoder(features)

        def _context(self, row: Any, prefix_mean: Any) -> Any:
            row = _mapping(row)
            features = torch.cat(
                [
                    self._enum("decision", row.get("kind")),
                    self._enum("entity", row.get("source_kind")),
                    self._enum("entity", row.get("target_kind")),
                    self._type(row.get("attack_trait_type")),
                    self._tensor(
                        [
                            _norm(row.get("source_slot")),
                            _norm(row.get("target_slot")),
                        ],
                        dtype=torch.float,
                    ),
                    prefix_mean,
                ]
            )
            return self.context_encoder(features)

        # -- observation --------------------------------------------------
        def forward(self, observation: Mapping[str, Any]) -> tuple[Any, Any, Any]:
            global_row = _mapping(observation.get("global"))
            seat = _int(global_row.get("seat"), -1)

            animals = _rows(observation, "animals")
            animal_vectors = [self._animal(row) for row in animals]
            own = [
                vector
                for row, vector in zip(animals, animal_vectors)
                if _int(_mapping(row).get("owner"), -2) == seat
            ]
            opponents = [
                vector
                for row, vector in zip(animals, animal_vectors)
                if _int(_mapping(row).get("owner"), -2) != seat
            ]
            plants = _rows(observation, "plants")
            plant_vectors = [self._plant(row) for row in plants]
            shells = _rows(observation, "shells")
            shell_vectors = []
            for row in shells:
                shell_vectors.append(self._shell(row))

            def entity_vector(kind: Any, slot: Any) -> Any:
                kind_id = _int(kind)
                slot_id = _int(slot, -1)
                if not 0 <= slot_id:
                    return self._zeros()
                if kind_id == _ENTITY_KIND["ANIMAL"] and slot_id < len(animal_vectors):
                    return animal_vectors[slot_id]
                if kind_id == _ENTITY_KIND["PLANT"] and slot_id < len(plant_vectors):
                    return plant_vectors[slot_id]
                if kind_id == _ENTITY_KIND["SHELL"] and slot_id < len(shell_vectors):
                    return shell_vectors[slot_id]
                return self._zeros()

            choices = _rows(observation, "choices")
            prefix_rows = _rows(_mapping(observation.get("context")), "prefix")
            prefix_mean = self._mean([self._choice(row) for row in prefix_rows])
            context_row = _mapping(observation.get("context"))
            context_vector = self._context(context_row, prefix_mean)
            global_vector = self._global(global_row)

            context = torch.cat(
                [
                    global_vector,
                    self._mean(own),
                    self._mean(opponents),
                    self._mean(plant_vectors),
                    context_vector,
                ]
            )

            source_vector = entity_vector(
                context_row.get("source_kind"), context_row.get("source_slot")
            )
            target_vector = entity_vector(
                context_row.get("target_kind"), context_row.get("target_slot")
            )
            choice_inputs = []
            for row in choices:
                choice_row = _mapping(row)
                choice_inputs.append(
                    torch.cat(
                        [
                            self._choice(choice_row),
                            source_vector,
                            target_vector,
                            entity_vector(
                                choice_row.get("entity_kind"),
                                choice_row.get("entity_slot"),
                            ),
                        ]
                    )
                )
            if choice_inputs:
                choice_tensor = torch.stack(choice_inputs, dim=0)
            else:
                choice_tensor = torch.zeros(
                    0, 4 * HIDDEN_DIM, device=self.device
                )

            mask = observation.get("action_mask")
            valid_values = _bits(mask, len(choices))
            valid = self._tensor([bool(value) for value in valid_values], dtype=torch.bool)
            return context, choice_tensor, valid

    class CandidatePolicy(nn.Module):
        """Candidate scorer plus a value head over the same context."""

        def __init__(self, spec: FeatureSpec, device: Any = "cpu") -> None:
            super().__init__()
            self.device = torch.device(device)
            self.encoder = ObservationEncoder(spec, device=self.device)
            input_dim = 5 * HIDDEN_DIM + 4 * HIDDEN_DIM
            self.candidate = RowMlp(input_dim)
            self.logit = nn.Linear(HIDDEN_DIM, 1)
            self.value_head = RowMlp(5 * HIDDEN_DIM)
            self.value = nn.Linear(HIDDEN_DIM, 1)

        def score(self, context: Any, choices: Any, valid: Any) -> Any:
            count = choices.shape[0]
            expanded = context.unsqueeze(0).expand(count, -1)
            outputs = self.logit(self.candidate(torch.cat([expanded, choices], dim=-1))).squeeze(-1)
            negative = torch.full_like(outputs, -1e30)
            return torch.where(valid, outputs, negative)

        def value_of(self, context: Any) -> Any:
            return self.value(self.value_head(context)).squeeze(-1)


def build_model(spec: FeatureSpec, seed: int = 7, device: Any = "cpu") -> Any:
    _require_torch()
    target = torch.device(device)
    torch.manual_seed(int(seed))
    if target.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = CandidatePolicy(spec, device=target)
    model.to(target)
    return model


# -------------------------------------------------------------------- batches
@dataclass
class EncodedDecision:
    context: Any
    choices: Any
    valid: Any
    chosen: int
    agent: str


@dataclass
class Episode:
    decisions: list[EncodedDecision] = field(default_factory=list)
    rewards: dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    actions: list[int] = field(default_factory=list)


def _collate(
    samples: Sequence[tuple[EncodedDecision, float]], device: Any = "cpu"
) -> tuple[Any, Any, Any, Any, Any]:
    """Pad only inside this collator; logits for padding become ``-inf``."""
    target_device = torch.device(device)
    max_choices = max(1, max(int(sample[0].choices.shape[0]) for sample in samples))
    contexts = []
    choices = []
    valid = []
    chosen = []
    targets = []
    for decision, target in samples:
        count = int(decision.choices.shape[0])
        pad = max_choices - count
        choice_rows = decision.choices
        if pad:
            choice_rows = torch.cat(
                [
                    choice_rows,
                    torch.zeros(
                        pad, choice_rows.shape[1], device=target_device
                    ),
                ],
                dim=0,
            )
        valid_rows = decision.valid
        if pad:
            valid_rows = torch.cat(
                [
                    valid_rows,
                    torch.zeros(pad, dtype=torch.bool, device=target_device),
                ],
                dim=0,
            )
        contexts.append(decision.context)
        choices.append(choice_rows)
        valid.append(valid_rows)
        chosen.append(int(decision.chosen))
        targets.append(float(target))
    return (
        torch.stack(contexts),
        torch.stack(choices),
        torch.stack(valid),
        torch.tensor(chosen, dtype=torch.long, device=target_device),
        torch.tensor(targets, dtype=torch.float, device=target_device),
    )


def _batch_logits(model: Any, context: Any, choices: Any, valid: Any) -> Any:
    batch, count, width = choices.shape
    flat_context = context.unsqueeze(1).expand(batch, count, context.shape[-1]).reshape(
        batch * count, -1
    )
    flat_choices = choices.reshape(batch * count, width)
    flat_valid = valid.reshape(batch * count)
    logits = model.logit(model.candidate(torch.cat([flat_context, flat_choices], dim=-1))).squeeze(-1)
    logits = torch.where(flat_valid, logits, torch.full_like(logits, -1e30))
    return logits.reshape(batch, count)


# ------------------------------------------------------------------ rollouts
def _catalog_from_env(env: Any) -> Any:
    raw = getattr(env, "unwrapped", env)
    engine = getattr(raw, "_engine", None)
    if engine is None:
        return None
    try:
        hello = engine.hello()
    except Exception:
        return None
    return hello.get("catalog") if isinstance(hello, Mapping) else None


def _select_index(model: Any, observation: Mapping[str, Any], greedy: bool = False) -> tuple[int, Any]:
    with torch.no_grad():
        context, choices, valid = model.encoder(observation)
        if int(valid.sum().item()) == 0:
            return 0, (context, choices, valid)
        logits = model.score(context, choices, valid)
        if greedy:
            index = int(torch.argmax(logits).item())
        else:
            probabilities = torch.softmax(logits, dim=0)
            index = int(torch.multinomial(probabilities, 1).item())
        return index, (context, choices, valid)


def collect_episode(
    env: Any,
    model: Any,
    seed: int | None,
    scenario: str | None = None,
    greedy: bool = False,
) -> Episode:
    env.reset(seed=seed, options={"scenario": scenario} if scenario else None)
    seats = list(getattr(env, "possible_agents", []) or getattr(env, "agents", []) or [])
    rewards = {seat: 0.0 for seat in seats}
    episode = Episode(rewards=rewards)

    while getattr(env, "agents", []):
        agent = getattr(env, "agent_selection", None)
        if agent is None:
            break
        if env.terminations.get(agent, False) or env.truncations.get(agent, False):
            env.step(None)
            continue
        observation = env.observe(agent)
        index, encoded = _select_index(model, observation, greedy=greedy)
        context, choices, valid = encoded
        episode.decisions.append(
            EncodedDecision(
                context=context,
                choices=choices,
                valid=valid,
                chosen=int(index),
                agent=str(agent),
            )
        )
        episode.actions.append(int(index))
        live = list(env.agents)
        env.step(index)
        for seat in live:
            rewards[seat] = rewards.get(seat, 0.0) + float(env.rewards.get(seat, 0.0))
        if env.terminations and all(env.terminations.values()):
            episode.terminated = True
        if env.truncations and any(env.truncations.values()):
            episode.truncated = True

    return episode


def _optimize(
    model: Any,
    optimizer: Any,
    episodes: Sequence[Episode],
    chunk_size: int = 128,
) -> dict[str, Any]:
    samples: list[tuple[EncodedDecision, float]] = []
    for episode in episodes:
        for decision in episode.decisions:
            samples.append((decision, float(episode.rewards.get(decision.agent, 0.0))))
    if not samples:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "finite": True,
            "gradients_finite": True,
        }

    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    finite = True
    gradients_finite = True
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        context, choices, valid, chosen, targets = _collate(
            chunk, getattr(model, "device", "cpu")
        )
        logits = _batch_logits(model, context, choices, valid)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)

        values = model.value_of(context)
        value_loss = 0.5 * torch.mean((values - targets) ** 2)
        policy_loss = -torch.sum(chosen_log_probs * (targets - values.detach()))

        probabilities = torch.softmax(logits, dim=-1)
        safe_log_probs = torch.where(valid, log_probs, torch.zeros_like(log_probs))
        entropy_rows = -torch.sum(probabilities * safe_log_probs, dim=-1)
        entropy = torch.mean(entropy_rows)

        loss = policy_loss + value_loss - ENTROPY_COEFFICIENT * entropy
        optimizer.zero_grad()
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                gradients_finite = False
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        for key, value in (
            ("loss", loss),
            ("policy_loss", policy_loss),
            ("value_loss", value_loss),
            ("entropy", entropy),
        ):
            scalar = float(value.detach())
            totals[key] += scalar
            if not math.isfinite(scalar):
                finite = False

    chunks = max(1, math.ceil(len(samples) / chunk_size))
    totals = {key: value / chunks for key, value in totals.items()}
    totals["finite"] = finite
    totals["gradients_finite"] = gradients_finite
    totals["samples"] = len(samples)
    return totals


def evaluate(model: Any, seeds: Sequence[int], scenario: str | None = None) -> list[dict[str, Any]]:
    from . import raw_env

    results = []
    env = raw_env(config=dict(BASE_CONFIG))
    try:
        for seed in seeds:
            episode = collect_episode(env, model, int(seed), scenario=scenario, greedy=True)
            player_reward = float(episode.rewards.get("player_0", 0.0))
            results.append(
                {
                    "seed": int(seed),
                    "reward_player_0": player_reward,
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                    "actions": list(episode.actions),
                }
            )
    finally:
        env.close()
    return results


# --------------------------------------------------------------- checkpoint
def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------- fixture
def _allowed(observation: Mapping[str, Any]) -> list[int]:
    choices = observation.get("choices") or []
    mask = observation.get("action_mask")
    if mask is None:
        mask = []
    allowed = []
    for index in range(len(choices)):
        value = _int(mask[index]) if index < len(mask) else 1
        if value:
            allowed.append(index)
    return allowed


def _fixture_index(observation: Mapping[str, Any], branch: str, catalog: Any) -> int:
    """Force the A->C (safe) or A->B (poisonous) prefix, defer to survival."""
    from .policies import (
        COMMAND_TYPES,
        SurvivalPolicy,
        _command_name,
        _field,
        _type_id,
    )

    choices = list(observation.get("choices") or [])
    allowed = _allowed(observation)
    if not allowed:
        return 0
    survival = SurvivalPolicy(catalog=catalog)
    command = _command_name(observation)

    if command is None:
        # Choose the Carnivorous trait activation when available.
        for index in allowed:
            row = choices[index]
            if (
                _field(row) == _FIELD["TYPE"]
                and _type_id(row) == COMMAND_TYPES.index("traitActivateRequest") + 1
            ):
                return int(index)
        return survival.select(observation)

    animals = list(observation.get("animals") or [])
    global_row = _mapping(observation.get("global"))
    seat = _int(global_row.get("seat"), 0)
    context = _mapping(observation.get("context"))
    prefix = list(context.get("prefix") or [])
    field = _field(choices[allowed[0]])

    def prefix_field(field_kind: int) -> bool:
        return any(_field(row) == field_kind for row in prefix)

    def prefix_slot(field_kind: int) -> int:
        for row in prefix:
            if _field(row) == field_kind:
                return _int(_mapping(row).get("entity_slot"), -1)
        return -1

    def target_index() -> int | None:
        wanted_position = 1 if branch == "safe" else 0
        for index in allowed:
            row = _mapping(choices[index])
            slot = _int(row.get("entity_slot"), -1)
            if 0 <= slot < len(animals):
                animal = _mapping(animals[slot])
                if (
                    _int(animal.get("owner"), -1) != seat
                    and _int(animal.get("position"), -1) == wanted_position
                ):
                    return int(index)
        return None

    if field == _FIELD["ANIMAL"] or field == _FIELD["ENTITY"]:
        if command == "traitActivateRequest" and not prefix_field(_FIELD["TRAIT"]):
            # Source: own animal carrying TraitCarnivorous (position 0).
            best: int | None = None
            carnivorous = _trait_type_name(catalog)
            for index in allowed:
                slot = _int(_mapping(choices[index]).get("entity_slot"), -1)
                if not 0 <= slot < len(animals):
                    continue
                animal = _mapping(animals[slot])
                if _int(animal.get("owner"), -1) != seat:
                    continue
                if best is None:
                    best = int(index)
                if carnivorous is not None:
                    for trait in _rows(animal, "traits"):
                        if _int(_mapping(trait).get("type"), -1) == carnivorous:
                            return int(index)
            if best is not None:
                return best
        chosen_target = target_index()
        if chosen_target is not None:
            return chosen_target
        return survival.select(observation)

    if field == _FIELD["TRAIT"]:
        carnivorous = _trait_type_name(catalog)
        source_slot = prefix_slot(_FIELD["ANIMAL"])
        if source_slot < 0:
            source_slot = prefix_slot(_FIELD["ENTITY"])
        for index in allowed:
            row = _mapping(choices[index])
            if carnivorous is not None and _int(row.get("type_id"), -1) == carnivorous:
                return int(index)
            trait_slot = _int(row.get("trait_slot"), -1)
            if 0 <= source_slot < len(animals):
                traits = _rows(animals[source_slot], "traits")
                if 0 <= trait_slot < len(traits):
                    if _int(_mapping(traits[trait_slot]).get("type"), -1) == carnivorous:
                        return int(index)
                if carnivorous is None and 0 <= trait_slot < len(traits):
                    return int(index)
        return survival.select(observation)

    if field == _FIELD["TARGET"]:
        chosen_target = target_index()
        if chosen_target is not None:
            return chosen_target
        return int(allowed[0])

    if field == _FIELD["DECLINE"]:
        # Refuse DECLINE while a legal attack branch exists.
        for index in allowed:
            if _field(choices[index]) != _FIELD["DECLINE"]:
                return int(index)
        return int(allowed[0])

    return survival.select(observation)


def _trait_type_name(catalog: Any) -> int | None:
    namespaces = _namespaces(catalog)
    for key in ("all_trait_types", "trait_types"):
        namespace = namespaces.get(key) if isinstance(namespaces, Mapping) else None
        if isinstance(namespace, Mapping) and isinstance(namespace.get("TraitCarnivorous"), int):
            return int(namespace["TraitCarnivorous"])
    return None


def _find_scores(value: Any, path: str = "") -> Any:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in ("scores", "score", "scorebyplayer", "finalscores"):
                return {str(path + "." + str(key)).lstrip("."): item}
            found = _find_scores(item, path + "." + str(key))
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_scores(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _run_fixture_branch(branch: str) -> dict[str, Any]:
    from . import raw_env

    env = raw_env(config=dict(BASE_CONFIG))
    try:
        env.reset(seed=1, options={"scenario": SCENARIO})
        catalog = _catalog_from_env(env)
        seats = list(getattr(env, "possible_agents", []))
        rewards = {seat: 0.0 for seat in seats}
        decisions: list[tuple[Mapping[str, Any], int]] = []
        terminated = False
        truncated = False

        while getattr(env, "agents", []):
            agent = getattr(env, "agent_selection", None)
            if agent is None:
                break
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                env.step(None)
                continue
            observation = env.observe(agent)
            index = _fixture_index(observation, branch, catalog)
            decisions.append((copy.deepcopy(observation), index))
            live = list(env.agents)
            env.step(index)
            for seat in live:
                rewards[seat] = rewards.get(seat, 0.0) + float(env.rewards.get(seat, 0.0))
            if env.terminations and all(env.terminations.values()):
                terminated = True
            if env.truncations and any(env.truncations.values()):
                truncated = True

        replay = env.export_replay()
        return {
            "branch": branch,
            "rewards": rewards,
            "terminated": terminated and not truncated,
            "truncated": truncated,
            "scores": _find_scores(replay),
            "decisions": decisions,
            "actions": [index for _, index in decisions],
        }
    finally:
        env.close()


def run_gradient_fixture(
    spec: FeatureSpec,
    catalog: Any,
    seed: int = 7,
    device: Any = "cpu",
) -> dict[str, Any]:
    """Confirm the scripted natural outcomes, then prove a policy gradient."""
    _require_torch()
    safe = _run_fixture_branch("safe")
    poison = _run_fixture_branch("poison")

    safe_reward = float(safe["rewards"].get("player_0", 0.0))
    poison_reward = float(poison["rewards"].get("player_0", 0.0))
    if not (safe["terminated"] and poison["terminated"]):
        raise RuntimeError("scripted fixture did not reach a natural terminal state")
    if not (math.isclose(safe_reward, 1.0) and math.isclose(poison_reward, -1.0)):
        raise RuntimeError(
            "scripted fixture rewards differ from the expected safe/poison outcomes: "
            f"safe={safe_reward!r} poison={poison_reward!r}"
        )

    scores = {"safe": safe["scores"], "poison": poison["scores"]}

    model = build_model(spec, seed=seed, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    safe_decisions = safe["decisions"]

    def log_probability() -> float:
        total = 0.0
        for observation, chosen in safe_decisions:
            with torch.no_grad():
                context, choices, valid = model.encoder(observation)
                logits = model.score(context, choices, valid)
                log_probs = torch.log_softmax(logits, dim=-1)
                total += float(log_probs[int(chosen)].item())
        return total

    before = log_probability()
    optimizer.zero_grad()
    total = torch.zeros((), dtype=torch.float, device=model.device)
    for observation, chosen in safe_decisions:
        context, choices, valid = model.encoder(observation)
        logits = model.score(context, choices, valid)
        log_probs = torch.log_softmax(logits, dim=-1)
        total = total + log_probs[int(chosen)]
    loss = -total
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
    optimizer.step()
    after = log_probability()
    stats = {
        "loss": float(loss.detach()),
        "gradients_finite": gradients_finite,
        "finite": bool(math.isfinite(float(loss.detach()))),
        "log_probability_before": before,
        "log_probability_after": after,
        "log_probability_increased": bool(after > before),
    }
    return {
        "natural_outcomes": {
            "safe_reward_player_0": safe_reward,
            "poison_reward_player_0": poison_reward,
            "scores": scores,
        },
        "gradient": stats,
    }


# ----------------------------------------------------------------------- run
def run_smoke(
    episodes: int = 32,
    seed: int = 7,
    output: str | Path = "rl/runs/smoke",
    device: Any = "auto",
) -> dict[str, Any]:
    _require_torch()
    from . import raw_env

    if episodes < 1:
        raise ValueError("episodes must be positive")
    target_device = _resolve_device(device)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    env = raw_env(config=dict(BASE_CONFIG))
    try:
        env.reset(seed=int(seed))
        catalog = _catalog_from_env(env)
    finally:
        env.close()

    spec = feature_spec(catalog)
    model = build_model(spec, seed=seed, device=target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    initial_hash = _state_hash(model)

    training_config = dict(BASE_CONFIG)
    train_env = raw_env(config=dict(training_config))
    updates = []
    completed = 0
    episode_returns = []
    try:
        batch: list[Episode] = []
        for episode_index in range(int(episodes)):
            episode = collect_episode(train_env, model, int(seed) + episode_index)
            batch.append(episode)
            completed += 1
            episode_returns.append(float(episode.rewards.get("player_0", 0.0)))
            if len(batch) >= BATCH_EPISODES:
                stats = _optimize(model, optimizer, batch)
                stats["episodes"] = completed
                updates.append(stats)
                batch = []
        if batch:
            stats = _optimize(model, optimizer, batch)
            stats["episodes"] = completed
            updates.append(stats)
    finally:
        train_env.close()

    final_hash = _state_hash(model)
    checkpoint_path = output_path / "checkpoint.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "catalog": catalog,
        "schema_version": spec.schema_version,
        "config": training_config,
        "seed": int(seed),
        "device": str(model.device),
    }
    torch.save(payload, checkpoint_path)
    checkpoint_hash = _checkpoint_hash(checkpoint_path)

    reloaded = build_model(spec, seed=seed, device=target_device)
    loaded = torch.load(
        checkpoint_path, map_location=target_device, weights_only=False
    )
    reloaded.load_state_dict(loaded["model"])

    evaluation_seeds = list(range(10000, 10008))
    evaluation = evaluate(reloaded, evaluation_seeds)
    reference = evaluate(model, evaluation_seeds)
    reproducible = all(
        left["actions"] == right["actions"]
        for left, right in zip(evaluation, reference)
    )

    fixture_error = None
    fixture = None
    try:
        fixture = run_gradient_fixture(
            spec, catalog, seed=seed, device=target_device
        )
    except Exception as exc:  # report a real rule/engine failure without hiding it
        fixture_error = f"{type(exc).__name__}: {exc}"

    report = {
        "episodes": completed,
        "seed": int(seed),
        "device": str(target_device),
        "config": training_config,
        "updates": updates,
        "losses_finite": all(update.get("finite", False) for update in updates),
        "gradients_finite": all(update.get("gradients_finite", False) for update in updates),
        "episode_returns_player_0": episode_returns,
        "mean_return_player_0": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "state_hash_initial": initial_hash,
        "state_hash_final": final_hash,
        "weights_changed": initial_hash != final_hash,
        "evaluation_seeds": evaluation_seeds,
        "evaluation": evaluation,
        "reload_reproducible": reproducible,
        "gradient_fixture": fixture,
        "gradient_fixture_error": fixture_error,
    }
    report_path = output_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evolution_rl.train_smoke",
        description="Short CPU REINFORCE smoke run with a value baseline.",
    )
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="rl/runs/smoke")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="torch device; EVOLUTION_RL_DEVICE is the fallback for auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_smoke(
        episodes=args.episodes,
        seed=args.seed,
        output=args.output,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["losses_finite"] or not report["gradients_finite"]:
        return 2
    if report["gradient_fixture_error"]:
        return 3
    fixture = report.get("gradient_fixture") or {}
    gradient = fixture.get("gradient") or {}
    if not gradient.get("log_probability_increased", False):
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
