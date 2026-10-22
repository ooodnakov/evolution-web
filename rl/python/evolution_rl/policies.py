"""Baseline policies for the Evolution RL environment.

Both policies are deliberately *policy-side only*: they choose one index of
``observation["choices"]`` and never touch the authoritative game, the engine
process or any value that is not part of the policy-visible observation
(``rl/schema.json`` -> ``observation``).

The random baseline samples one prefix choice at a time, uniformly among the
currently mask-allowed options.  It is therefore not a uniform draw over the
full set of compound actions; that distinction is intentional.

The survival baseline is a deterministic, visible-only greedy policy.  It
knows the public rule vocabulary (trait names -> food demand), but it only
uses values that a player can see in the observation.  A catalog (the public
``hello`` vocabulary) may be supplied so that trait type ids can be mapped to
names; without it the policy still works and falls back to local-order
choices instead of guessing hidden state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FIELD_TYPE",
    "FIELD_HAND_CARD",
    "FIELD_POSITION",
    "FIELD_ENTITY",
    "FIELD_TARGET",
    "FIELD_TRAIT",
    "FIELD_BOOLEAN",
    "FIELD_ANIMAL",
    "FIELD_PLANT",
    "FIELD_SHELL",
    "FIELD_CARD",
    "FIELD_DECLINE",
    "FIELD_USE_DEFAULT",
    "FIELD_READY",
    "KIND_NORMAL",
    "KIND_DEFENSE",
    "KIND_INTELLECT",
    "KIND_AMBUSH",
    "KIND_REGENERATION",
    "KIND_TRANSIENT",
    "KIND_TERMINAL",
    "PHASE_DEPLOY",
    "PHASE_FEEDING",
    "COMMAND_TYPES",
    "TRAIT_FOOD_DEMAND",
    "RandomPolicy",
    "SurvivalPolicy",
    "make_policy",
    "random_policy",
    "survival_policy",
]


# --------------------------------------------------------------- schema ids
def _schema_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [parent / "rl" / "schema.json" for parent in here.parents]
    cwd = Path.cwd().resolve()
    candidates.extend(parent / "rl" / "schema.json" for parent in (cwd, *cwd.parents))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("rl/schema.json is required to index choice rows")


def _load_schema() -> dict[str, Any]:
    with open(_schema_path(), encoding="utf-8") as schema_file:
        return json.load(schema_file)


_SCHEMA = _load_schema()


def _index(name: str) -> dict[str, int]:
    return {value: index for index, value in enumerate(_SCHEMA[name])}


_FIELD = _index("choice_field_kinds")
_CHOICE_KIND = _index("choice_kinds")
_DECISION_KIND = _index("decision_kinds")
_PHASE = _index("phase_values")
_ENTITY_KIND = _index("entity_kinds")
# ``rl/node/actions.js`` emits TYPE ids as a 1-based index into its normal
# command list (the special question/ambush/regeneration decisions have no
# TYPE level at all).
COMMAND_TYPES = (
    "gameEndTurnRequest",
    "gameDeployAnimalRequest",
    "gameDeployTraitRequest",
    "traitTakeFoodRequest",
    "traitTakeShellRequest",
    "traitTakeCoverRequest",
    "traitActivateRequest",
    "gamePlantAttackRequest",
    "plantTraitActivateRequest",
)
_COMMAND = {name: index + 1 for index, name in enumerate(COMMAND_TYPES)}
_COMMAND_BY_ID = {value: name for name, value in _COMMAND.items()}

FIELD_TYPE = _FIELD["TYPE"]
FIELD_HAND_CARD = _FIELD["HAND_CARD"]
FIELD_POSITION = _FIELD["POSITION"]
FIELD_ENTITY = _FIELD["ENTITY"]
FIELD_TARGET = _FIELD["TARGET"]
FIELD_TRAIT = _FIELD["TRAIT"]
FIELD_BOOLEAN = _FIELD["BOOLEAN"]
FIELD_ANIMAL = _FIELD["ANIMAL"]
FIELD_PLANT = _FIELD["PLANT"]
FIELD_SHELL = _FIELD["SHELL"]
FIELD_CARD = _FIELD["CARD"]
FIELD_DECLINE = _FIELD["DECLINE"]
FIELD_USE_DEFAULT = _FIELD["USE_DEFAULT"]
FIELD_READY = _FIELD["READY"]

KIND_NORMAL = _DECISION_KIND["NORMAL"]
KIND_DEFENSE = _DECISION_KIND["DEFENSE"]
KIND_INTELLECT = _DECISION_KIND["INTELLECT"]
KIND_AMBUSH = _DECISION_KIND["AMBUSH"]
KIND_REGENERATION = _DECISION_KIND["REGENERATION"]
KIND_TRANSIENT = _DECISION_KIND["TRANSIENT"]
KIND_TERMINAL = _DECISION_KIND["TERMINAL"]

PHASE_DEPLOY = _PHASE["DEPLOY"]
PHASE_FEEDING = _PHASE["FEEDING"]

ENTITY_ANIMAL = _ENTITY_KIND["ANIMAL"]

# Public rule knowledge taken from the registry data models (``food:`` on the
# card data).  Traits absent from this table add no food demand.
TRAIT_FOOD_DEMAND: dict[str, int] = {
    "TraitParasite": 2,
    "TraitMassive": 1,
    "TraitCarnivorous": 1,
    "TraitHomeothermy": 1,
    "TraitPlantHomeothermy": 1,
    "TraitAggression": 1,
    "TraitAdaptation": -1,
    "TraitVoracious": 1,
    "TraitFlea": 1,
    "TraitCyst": 1,
    "TraitTrematode": 1,
    "TraitViviparous": 1,
    "TraitIntellect": 1,
}

_FAT_TISSUE = "TraitFatTissue"
_ANGLERFISH = "TraitAnglerfish"
_CARNIVOROUS = "TraitCarnivorous"


# ------------------------------------------------------------------ helpers
def _rows(observation: Mapping[str, Any], key: str) -> list[Any]:
    value = observation.get(key)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            value = value[0]
        return int(value)
    except (TypeError, ValueError):
        return default


def _mask_values(observation: Mapping[str, Any], count: int) -> list[int]:
    raw = observation.get("action_mask")
    if raw is None:
        return [1] * count
    if isinstance(raw, np.ndarray):
        values = [int(item) for item in raw.reshape(-1)]
    elif isinstance(raw, (list, tuple)):
        values = []
        for item in raw:
            if isinstance(item, (list, tuple, np.ndarray)):
                values.append(_int(item, 1))
            else:
                values.append(int(item))
    else:
        values = [_int(raw, 1)]
    if len(values) < count:
        values.extend([0] * (count - len(values)))
    return [1 if value != 0 else 0 for value in values[:count]]


def _allowed(observation: Mapping[str, Any], choices: Sequence[Any]) -> list[int]:
    if not choices:
        return []
    mask = _mask_values(observation, len(choices))
    return [index for index, value in enumerate(mask) if value]


def _field(row: Any) -> int:
    return _int(_mapping(row).get("field_kind"), -1)


def _choice_kind(row: Any) -> int:
    return _int(_mapping(row).get("choice_kind"), -1)


def _scalar(row: Any) -> int:
    return _int(_mapping(row).get("scalar"), 0)


def _entity_slot(row: Any) -> int:
    return _int(_mapping(row).get("entity_slot"), -1)


def _trait_slot(row: Any) -> int:
    return _int(_mapping(row).get("trait_slot"), -1)


def _type_id(row: Any) -> int:
    return _int(_mapping(row).get("type_id"), 0)


def _context(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(observation.get("context"))


def _prefix(observation: Mapping[str, Any]) -> list[Any]:
    return _rows(_context(observation), "prefix")


def _command_type(observation: Mapping[str, Any]) -> int | None:
    """Command type id chosen earlier in the current prefix, if any."""
    for row in _prefix(observation):
        if _field(row) == FIELD_TYPE:
            return _type_id(row)
    return None


def _command_name(observation: Mapping[str, Any]) -> str | None:
    command = _command_type(observation)
    return _COMMAND_BY_ID.get(command) if command is not None else None


def _own_seat(observation: Mapping[str, Any]) -> int:
    return _int(_mapping(observation.get("global")).get("seat"), -1)


def _phase(observation: Mapping[str, Any]) -> int:
    return _int(_mapping(observation.get("global")).get("phase"), -1)


def _animals(observation: Mapping[str, Any]) -> list[Any]:
    return _rows(observation, "animals")


def _plants(observation: Mapping[str, Any]) -> list[Any]:
    return _rows(observation, "plants")


def _hand(observation: Mapping[str, Any]) -> list[Any]:
    return _rows(observation, "hand")


def _animal_need(animal: Any) -> int:
    row = _mapping(animal)
    return max(0, _int(row.get("food_size")) - _int(row.get("food")))


def _animal_is_own(animal: Any, seat: int) -> bool:
    return _int(_mapping(animal).get("owner"), -2) == seat


def _animal_has_regeneration(animal: Any, schema: Mapping[str, Any] | None = None) -> bool:
    flags = _mapping(animal).get("flags")
    # schema.animal_flags order: HIBERNATED, POISONED, SHELL, REGENERATION, ...
    try:
        if isinstance(flags, np.ndarray):
            return bool(flags.reshape(-1)[3])
        if isinstance(flags, (list, tuple)):
            return len(flags) > 3 and bool(_int(flags[3]))
    except (IndexError, TypeError, ValueError):
        pass
    return False


# --------------------------------------------------------------- catalog ids
def _namespaces(catalog: Any) -> Mapping[str, Any]:
    if not isinstance(catalog, Mapping):
        return {}
    namespaces = catalog.get("namespaces")
    if isinstance(namespaces, Mapping):
        return namespaces
    return catalog


def _namespace(catalog: Any, name: str) -> Mapping[str, int]:
    namespace = _namespaces(catalog).get(name)
    return namespace if isinstance(namespace, Mapping) else {}


def _ids_for_names(catalog: Any, *names: str) -> set[int]:
    """Ids for public names across the trait namespaces of a catalog."""
    ids: set[int] = set()
    for namespace_name in ("all_trait_types", "trait_types", "plant_trait_types"):
        namespace = _namespace(catalog, namespace_name)
        for name in names:
            value = namespace.get(name)
            if isinstance(value, int):
                ids.add(value)
    return ids


def _id_to_name(catalog: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    for namespace_name in (
        "all_trait_types",
        "trait_types",
        "plant_trait_types",
        "card_types",
        "plant_types",
    ):
        for name, value in _namespace(catalog, namespace_name).items():
            if isinstance(value, int) and value >= 0:
                result.setdefault(value, name)
    return result


def _demand_of_type(type_id: int, catalog: Any) -> int:
    name = _id_to_name(catalog).get(type_id)
    if name is None:
        return 0
    return TRAIT_FOOD_DEMAND.get(name, 0)


# ------------------------------------------------------------------ policies
class RandomPolicy:
    """Uniform per-prefix sampling over the currently allowed choices."""

    name = "random"

    def __init__(self, rng: Any = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.catalog: Any = None

    def select(self, observation: Mapping[str, Any], rng: Any = None) -> int:
        choices = _rows(observation, "choices")
        allowed = _allowed(observation, choices)
        if not allowed:
            return 0
        generator = rng if rng is not None else self.rng
        return int(allowed[int(generator.integers(0, len(allowed)))])


class SurvivalPolicy:
    """Deterministic visible-only greedy baseline."""

    name = "survival"

    def __init__(self, catalog: Any = None) -> None:
        self.catalog = catalog
        self._fat_ids = _ids_for_names(catalog, _FAT_TISSUE)
        self._anglerfish_ids = _ids_for_names(catalog, _ANGLERFISH)
        self._carnivorous_ids = _ids_for_names(catalog, _CARNIVOROUS)
        self._sharing_ids = _ids_for_names(catalog, "TraitCooperation", "TraitCommunication")
        self._repeat_counts: dict[tuple[str, int], int] = {}

    # -- public ----------------------------------------------------------
    def select(self, observation: Mapping[str, Any], rng: Any = None) -> int:
        """Deterministic choice with a bounded guard against exact cycles.

        The mask semantics deliberately admit attempts that a later
        auto-sharing step can reject; a stateless greedy policy would then
        repeat the same rejected proposal until ``maxDecisions``.  When the
        same visible observation and the same choice recur, move to another
        allowed choice (preferring the end-turn command).
        """
        result = self._decide(observation, rng)
        return self._avoid_cycle(observation, result)

    def _avoid_cycle(self, observation: Mapping[str, Any], result: int) -> int:
        choices = _rows(observation, "choices")
        allowed = _allowed(observation, choices)
        if not allowed:
            return int(result)
        signature = self._signature(observation)
        key = (signature, int(result))
        count = self._repeat_counts.get(key, 0) + 1
        if count > 2:
            alternatives = [index for index in allowed if index != result]
            if alternatives:
                for index in alternatives:
                    if (
                        _field(choices[index]) == FIELD_TYPE
                        and _type_id(choices[index]) == _COMMAND["gameEndTurnRequest"]
                    ):
                        result = index
                        break
                else:
                    result = alternatives[0]
                key = (signature, int(result))
                count = 1
        self._repeat_counts[key] = count
        if len(self._repeat_counts) > 2048:
            self._repeat_counts.clear()
        return int(result)

    @staticmethod
    def _signature(observation: Mapping[str, Any]) -> str:
        payload = {
            "global": observation.get("global"),
            "players": observation.get("players"),
            "hand": observation.get("hand"),
            "animals": observation.get("animals"),
            "plants": observation.get("plants"),
            "shells": observation.get("shells"),
            "context": observation.get("context"),
            "choices": observation.get("choices"),
            "action_mask": observation.get("action_mask"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _decide(self, observation: Mapping[str, Any], rng: Any = None) -> int:
        choices = _rows(observation, "choices")
        if not choices:
            return 0
        allowed = _allowed(observation, choices)
        if not allowed:
            return 0

        kind = _int(_context(observation).get("kind"), KIND_NORMAL)
        if kind == KIND_TERMINAL:
            return allowed[0]

        # ------- special question / transient windows -------------------
        if kind == KIND_DEFENSE:
            return self._first_non_decline(choices, allowed)
        if kind == KIND_INTELLECT:
            return self._first_non_decline(choices, allowed)
        if kind == KIND_AMBUSH:
            return self._ambush(choices, allowed)
        if kind == KIND_TRANSIENT:
            return self._transient(choices, allowed)
        if kind == KIND_REGENERATION:
            return self._regeneration_choice(observation, choices, allowed)

        # ------- normal (DEPLOY / FEEDING) -------------------------------
        command = _command_name(observation)
        if command is None:
            return self._normal_type(observation, choices, allowed)
        return self._field_choice(observation, command, choices, allowed)

    # -- question windows -------------------------------------------------
    def _first_non_decline(self, choices: Sequence[Any], allowed: Iterable[int]) -> int:
        for index in allowed:
            row = _mapping(choices[index])
            if _choice_kind(row) != _CHOICE_KIND["DECLINE"] and _field(row) != FIELD_DECLINE:
                return int(index)
        return int(next(iter(allowed)))

    def _ambush(self, choices: Sequence[Any], allowed: Iterable[int]) -> int:
        allowed_list = list(allowed)
        # A type level offers traitAmbushActivateRequest / Continue.
        # First level: own pending ambusher (or the continue option).
        for index in allowed_list:
            row = _mapping(choices[index])
            if _field(row) in (FIELD_BOOLEAN,) or _choice_kind(row) == _CHOICE_KIND["SCALAR"]:
                if _scalar(row) != 0:
                    return int(index)
        return int(allowed_list[0])

    def _transient(self, choices: Sequence[Any], allowed: Iterable[int]) -> int:
        allowed_list = list(allowed)
        for index in allowed_list:
            row = _mapping(choices[index])
            if _field(row) == FIELD_BOOLEAN and _scalar(row) != 0:
                return int(index)
            if _choice_kind(row) == _CHOICE_KIND["SCALAR"] and _scalar(row) != 0:
                return int(index)
            if _int(row.get("type_id")) in self._anglerfish_ids:
                if _scalar(row) != 0 or _field(row) != FIELD_DECLINE:
                    return int(index)
        for index in allowed_list:
            row = _mapping(choices[index])
            if _choice_kind(row) in (
                _CHOICE_KIND["READY"],
                _CHOICE_KIND["USE_DEFAULT"],
                _CHOICE_KIND["DECLINE"],
            ) or _field(row) in (FIELD_READY, FIELD_USE_DEFAULT, FIELD_DECLINE):
                return int(index)
        return int(allowed_list[0])

    # -- command type selection ------------------------------------------
    def _pick_command(
        self, command_names: Sequence[str], choices: Sequence[Any], allowed: Iterable[int]
    ) -> int:
        allowed_list = list(allowed)
        by_type: dict[int, int] = {}
        for index in allowed_list:
            row = _mapping(choices[index])
            if _field(row) == FIELD_TYPE:
                by_type.setdefault(_type_id(row), int(index))
        for name in command_names:
            command_id = _COMMAND.get(name)
            if command_id in by_type:
                return by_type[command_id]
        return int(allowed_list[0])

    def _normal_type(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Iterable[int]
    ) -> int:
        phase = _phase(observation)
        seat = _own_seat(observation)
        own = [animal for animal in _animals(observation) if _animal_is_own(animal, seat)]

        if phase == PHASE_DEPLOY:
            if not own:
                return self._pick_command(
                    ("gameDeployAnimalRequest", "gameEndTurnRequest"), choices, allowed
                )
            return self._pick_command(
                ("gameDeployTraitRequest", "gameEndTurnRequest"), choices, allowed
            )

        if phase == PHASE_FEEDING:
            need = any(_animal_need(animal) > 0 for animal in own)
            shared_food = _int(_mapping(observation.get("global")).get("food")) > 0
            plant_food = any(
                _int(_mapping(plant).get("food")) > 0 for plant in _plants(observation)
            )
            if need and self._sharing_available(own):
                # Auto-sharing can satisfy the visible need before the feed
                # request is validated; a stateless greedy policy would retry
                # the rejected attempt forever.
                return self._pick_command(("gameEndTurnRequest",), choices, allowed)
            if need and (plant_food if _plants(observation) else shared_food):
                return self._pick_command(
                    ("traitTakeFoodRequest", "gameEndTurnRequest"), choices, allowed
                )
            if self._fat_available(own):
                return self._pick_command(
                    ("traitActivateRequest", "gameEndTurnRequest"), choices, allowed
                )
            return self._pick_command(("gameEndTurnRequest",), choices, allowed)

        return self._pick_command(("gameEndTurnRequest",), choices, allowed)

    def _regeneration_choice(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Iterable[int]
    ) -> int:
        allowed_list = list(allowed)
        if not allowed_list:
            return 0
        field = _field(choices[allowed_list[0]])
        if field in (FIELD_USE_DEFAULT, FIELD_DECLINE):
            return int(allowed_list[0])
        if field in (FIELD_ANIMAL, FIELD_ENTITY):
            return self._first_regeneration_animal(observation, choices, allowed_list)
        # Hand card level: first card locally; the animal was chosen already.
        return int(allowed_list[0])

    def _fat_available(self, own: Sequence[Any]) -> bool:
        return any(self._animal_has_fat_capacity(animal) for animal in own)

    def _sharing_available(self, own: Sequence[Any]) -> bool:
        if not self._sharing_ids:
            return False
        hungry = any(_animal_need(animal) > 0 for animal in own)
        if not hungry:
            return False
        for animal in own:
            if _animal_need(animal) > 0:
                continue
            for trait in _rows(animal, "traits"):
                if _int(_mapping(trait).get("type")) in self._sharing_ids:
                    return True
        return False

    # -- prefix fields ----------------------------------------------------
    def _field_choice(
        self,
        observation: Mapping[str, Any],
        command: str,
        choices: Sequence[Any],
        allowed: Iterable[int],
    ) -> int:
        allowed_list = list(allowed)
        if not allowed_list:
            return 0
        field = _field(choices[allowed_list[0]])

        if field == FIELD_HAND_CARD:
            if command == "gameDeployTraitRequest":
                return self._min_demand_card(observation, choices, allowed_list)
            return int(allowed_list[0])

        if field == FIELD_POSITION:
            for index in allowed_list:
                if _scalar(choices[index]) == 0:
                    return int(index)
            return int(min(allowed_list, key=lambda index: _scalar(choices[index])))

        if field == FIELD_BOOLEAN:
            if command == "gameDeployTraitRequest":
                return self._min_demand_side(observation, choices, allowed_list)
            for index in allowed_list:
                if _scalar(choices[index]) != 0:
                    return int(index)
            return int(allowed_list[0])

        if field in (FIELD_ENTITY, FIELD_ANIMAL):
            if command == "traitTakeFoodRequest":
                return self._min_need_animal(observation, choices, allowed_list)
            if command == "traitActivateRequest":
                return self._first_fat_animal(observation, choices, allowed_list)
            if command == "gameDeployRegeneratedAnimalRequest":
                return self._first_regeneration_animal(observation, choices, allowed_list)
            return int(allowed_list[0])

        if field in (FIELD_TARGET, FIELD_PLANT):
            if command == "traitTakeFoodRequest":
                return self._first_food_plant(observation, choices, allowed_list)
            return int(allowed_list[0])

        if field == FIELD_TRAIT:
            if command == "traitActivateRequest":
                return self._first_fat_trait(observation, choices, allowed_list)
            return int(allowed_list[0])

        if field in (FIELD_DECLINE, FIELD_USE_DEFAULT, FIELD_READY):
            return int(allowed_list[0])

        return int(allowed_list[0])

    # -- leaf helpers -----------------------------------------------------
    def _min_demand_card(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Sequence[int]
    ) -> int:
        hand = _hand(observation)
        best_index: int | None = None
        best_demand: int | None = None
        for index in allowed:
            row = _mapping(choices[index])
            slot = _entity_slot(row)
            if not 0 <= slot < len(hand):
                slot = index if index < len(hand) else -1
            demand = 0
            if 0 <= slot < len(hand):
                card = _mapping(hand[slot])
                sides = [
                    _int(card.get("trait1_type")),
                    _int(card.get("trait2_type")),
                ]
                sides = [side for side in sides if side > 0]
                demand = min(
                    (_demand_of_type(side, self.catalog) for side in sides),
                    default=0,
                )
            if best_demand is None or demand < best_demand:
                best_demand = demand
                best_index = index
        return int(best_index if best_index is not None else allowed[0])

    def _min_demand_side(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Sequence[int]
    ) -> int:
        hand = _hand(observation)
        card_slot = -1
        for row in _prefix(observation):
            if _field(row) == FIELD_HAND_CARD:
                card_slot = _entity_slot(row)
                break
        if not 0 <= card_slot < len(hand):
            card_slot = 0
        card = _mapping(hand[card_slot]) if hand else {}
        type1 = _int(card.get("trait1_type"))
        type2 = _int(card.get("trait2_type"))
        demand1 = _demand_of_type(type1, self.catalog)
        demand2 = _demand_of_type(type2, self.catalog)
        if type1 <= 0 and type2 > 0:
            want = 1
        elif type2 <= 0 or demand1 <= demand2:
            want = 0
        else:
            want = 1
        for index in allowed:
            if _scalar(choices[index]) == want:
                return int(index)
        return int(allowed[0])

    def _min_need_animal(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Sequence[int]
    ) -> int:
        animals = _animals(observation)
        seat = _own_seat(observation)
        best_index: int | None = None
        best_need: int | None = None
        for index in allowed:
            slot = _entity_slot(choices[index])
            if not 0 <= slot < len(animals):
                slot = index if index < len(animals) else -1
            if not 0 <= slot < len(animals):
                continue
            animal = animals[slot]
            if not _animal_is_own(animal, seat):
                continue
            need = _animal_need(animal)
            if need <= 0:
                continue
            if best_need is None or need < best_need:
                best_need = need
                best_index = index
        return int(best_index if best_index is not None else allowed[0])

    def _first_regeneration_animal(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Sequence[int]
    ) -> int:
        animals = _animals(observation)
        seat = _own_seat(observation)
        for index in allowed:
            slot = _entity_slot(choices[index])
            if not 0 <= slot < len(animals):
                slot = index if index < len(animals) else -1
            if 0 <= slot < len(animals):
                animal = animals[slot]
                if _animal_is_own(animal, seat) and _animal_has_regeneration(animal):
                    return int(index)
        return int(allowed[0])

    def _first_food_plant(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Sequence[int]
    ) -> int:
        plants = _plants(observation)
        for index in allowed:
            slot = _entity_slot(choices[index])
            if not 0 <= slot < len(plants):
                slot = index if index < len(plants) else -1
            if 0 <= slot < len(plants) and _int(_mapping(plants[slot]).get("food")) > 0:
                return int(index)
        return int(allowed[0])

    def _first_fat_animal(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Sequence[int]
    ) -> int:
        animals = _animals(observation)
        for index in allowed:
            slot = _entity_slot(choices[index])
            if not 0 <= slot < len(animals):
                continue
            animal = animals[slot]
            if not self._animal_has_fat_capacity(animal):
                continue
            return int(index)
        return int(allowed[0])

    def _animal_has_fat_capacity(self, animal: Any) -> bool:
        if _int(_mapping(animal).get("fat_size")) <= 0:
            return False
        for trait in _rows(animal, "traits"):
            if _int(_mapping(trait).get("type")) in self._fat_ids:
                return True
        return False

    def _first_fat_trait(
        self, observation: Mapping[str, Any], choices: Sequence[Any], allowed: Sequence[int]
    ) -> int:
        animals = _animals(observation)
        for index in allowed:
            row = _mapping(choices[index])
            trait_type = _type_id(row)
            if trait_type in self._fat_ids:
                return int(index)
            slot = _entity_slot(row)
            trait_slot = _trait_slot(row)
            if 0 <= slot < len(animals):
                traits = _rows(animals[slot], "traits")
                if 0 <= trait_slot < len(traits):
                    if _int(_mapping(traits[trait_slot]).get("type")) in self._fat_ids:
                        return int(index)
        return int(allowed[0])


_DEFAULT_RANDOM = RandomPolicy(rng=np.random.default_rng())
_DEFAULT_SURVIVAL = SurvivalPolicy()


def random_policy(observation: Mapping[str, Any], rng: Any = None) -> int:
    """Uniform per-prefix choice with the supplied policy RNG."""
    return _DEFAULT_RANDOM.select(observation, rng=rng)


def survival_policy(observation: Mapping[str, Any]) -> int:
    """Deterministic visible-only greedy choice."""
    return _DEFAULT_SURVIVAL.select(observation)


def make_policy(name: str, rng: Any = None, catalog: Any = None) -> RandomPolicy | SurvivalPolicy:
    """Create a baseline policy by name (``random`` or ``survival``)."""
    key = str(name).strip().lower()
    if key == "random":
        policy = RandomPolicy(rng=rng)
        policy.catalog = catalog
        return policy
    if key == "survival":
        return SurvivalPolicy(catalog=catalog)
    raise ValueError(f"unknown policy {name!r}; expected 'random' or 'survival'")
