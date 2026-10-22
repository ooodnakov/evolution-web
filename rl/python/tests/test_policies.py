"""Baseline policy contract tests (synthetic observations, no engine)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evolution_rl import policies as P

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((REPO_ROOT / "rl" / "schema.json").read_text(encoding="utf-8"))

FIELD = {name: index for index, name in enumerate(SCHEMA["choice_field_kinds"])}
KIND = {name: index for index, name in enumerate(SCHEMA["choice_kinds"])}
DECISION = {name: index for index, name in enumerate(SCHEMA["decision_kinds"])}
PHASE = {name: index for index, name in enumerate(SCHEMA["phase_values"])}
COMMAND = {
    "gameEndTurnRequest": 1,
    "gameDeployAnimalRequest": 2,
    "gameDeployTraitRequest": 3,
    "traitTakeFoodRequest": 4,
    "traitTakeShellRequest": 5,
    "traitTakeCoverRequest": 6,
    "traitActivateRequest": 7,
    "gamePlantAttackRequest": 8,
    "plantTraitActivateRequest": 9,
}

CATALOG = {
    "namespaces": {
        "all_trait_types": {
            "NONE": 0,
            "TraitFatTissue": 5,
            "TraitSwimming": 6,
            "TraitParasite": 7,
            "TraitCarnivorous": 8,
            "TraitAnglerfish": 9,
        },
        "trait_types": {
            "NONE": 0,
            "TraitFatTissue": 5,
            "TraitSwimming": 6,
            "TraitParasite": 7,
            "TraitCarnivorous": 8,
            "TraitAnglerfish": 9,
        },
    }
}


def trait_row(trait_type: int = 0, **overrides):
    row = {
        "type": trait_type,
        "disabled": 0,
        "covers": 0,
        "link_source": -1,
        "linked_entity_kind": 0,
        "linked_entity_slot": -1,
        "linked_trait_slot": -1,
        "value_kind": 0,
        "value_number": 0,
        "value_entity_kind": 0,
        "value_entity_slot": -1,
        "value_trait_slot": -1,
        "source_type": 0,
        "source_slot": -1,
        "auto_share": 0,
        "cooldowns": [],
    }
    row.update(overrides)
    return row


def choice(
    field: str,
    kind: str = "SCALAR",
    type_id: int = 0,
    entity_kind: int = 0,
    entity_slot: int = -1,
    trait_slot: int = -1,
    scalar: int = 0,
):
    return {
        "field_kind": FIELD[field],
        "choice_kind": KIND[kind],
        "type_id": int(type_id),
        "entity_kind": int(entity_kind),
        "entity_slot": int(entity_slot),
        "trait_slot": int(trait_slot),
        "scalar": int(scalar),
    }


def animal(
    owner: int = 0,
    position: int = 0,
    food: int = 0,
    food_size: int = 1,
    fat_size: int = 0,
    flags=None,
    traits=(),
):
    return {
        "owner": owner,
        "position": position,
        "food": food,
        "food_size": food_size,
        "fat_size": fat_size,
        "flags": list(flags or [0] * len(SCHEMA["animal_flags"])),
        "traits": list(traits),
    }


def observation(
    *,
    phase: str = "DEPLOY",
    seat: int = 0,
    animals=(),
    plants=(),
    hand=(),
    context_kind: str = "NORMAL",
    prefix=(),
    choices=(),
    mask=None,
):
    return {
        "global": {
            "phase": PHASE[phase],
            "turn": 1,
            "round": 1,
            "seat": seat,
            "current_player": seat,
            "round_player": seat,
            "food": 0,
            "deck_count": 0,
            "active_players": 2,
            "addons": [0] * len(SCHEMA["addons"]),
            "half_deck": 0,
        },
        "players": [],
        "hand": list(hand),
        "animals": list(animals),
        "plants": list(plants),
        "shells": [],
        "context": {
            "kind": DECISION[context_kind],
            "source_kind": 0,
            "source_slot": -1,
            "target_kind": 0,
            "target_slot": -1,
            "attack_trait_type": 0,
            "prefix": list(prefix),
        },
        "choices": list(choices),
        "action_mask": list(mask if mask is not None else [1] * len(choices)),
    }


class RecordingRng:
    def __init__(self, answer: int) -> None:
        self.answer = answer
        self.calls = []

    def integers(self, low, high):
        self.calls.append((low, high))
        return self.answer


# ------------------------------------------------------------------- random
def test_random_respects_mask_and_uses_given_rng():
    choices = [choice("HAND_CARD", "ENTITY", entity_slot=index) for index in range(5)]
    obs = observation(choices=choices, mask=[1, 0, 1, 0, 1])
    rng = RecordingRng(1)
    policy = P.make_policy("random", rng=np.random.default_rng(0))
    assert policy.select(obs, rng=rng) == 2
    assert rng.calls == [(0, 3)]


def test_random_empty_mask_falls_back_to_zero():
    choices = [choice("HAND_CARD", "ENTITY", entity_slot=0)]
    obs = observation(choices=choices, mask=[0])
    assert P.make_policy("random", rng=np.random.default_rng(1)).select(obs) == 0
    assert P.make_policy("random", rng=np.random.default_rng(1)).select(observation()) == 0


def test_random_is_per_prefix_not_compound():
    choices = [choice("HAND_CARD", "ENTITY", entity_slot=index) for index in range(3)]
    obs = observation(choices=choices, mask=[1, 1, 1])
    policy = P.make_policy("random", rng=np.random.default_rng(42))
    first = [policy.select(obs) for _ in range(20)]
    assert set(first) <= {0, 1, 2}
    assert len(set(first)) > 1


# ----------------------------------------------------------------- survival
def test_survival_deploy_animal_when_no_own_animals():
    obs = observation(
        phase="DEPLOY",
        choices=[
            choice("TYPE", "TYPE", type_id=COMMAND["gameEndTurnRequest"]),
            choice("TYPE", "TYPE", type_id=COMMAND["gameDeployAnimalRequest"]),
        ],
    )
    assert P.make_policy("survival").select(obs) == 1

    obs = observation(
        phase="DEPLOY",
        prefix=[
            choice(
                "TYPE",
                "TYPE",
                type_id=COMMAND["gameDeployAnimalRequest"],
            )
        ],
        choices=[
            choice("HAND_CARD", "ENTITY", entity_slot=0),
            choice("HAND_CARD", "ENTITY", entity_slot=1),
        ],
        hand=[
            {"card_type": 1, "trait1_type": 6, "trait2_type": 0},
            {"card_type": 1, "trait1_type": 5, "trait2_type": 0},
        ],
    )
    assert P.make_policy("survival").select(obs) == 0


def test_survival_deploy_trait_minimal_demand_and_side():
    hand = [
        {"card_type": 1, "trait1_type": 7, "trait2_type": 0},  # Parasite: +2
        {"card_type": 1, "trait1_type": 6, "trait2_type": 5},  # Swimming 0 / Fat 0
    ]
    obs = observation(
        phase="DEPLOY",
        animals=[animal(owner=0, position=0)],
        hand=hand,
        prefix=[
            choice("TYPE", "TYPE", type_id=COMMAND["gameDeployTraitRequest"]),
        ],
        choices=[choice("HAND_CARD", "ENTITY", entity_slot=0), choice("HAND_CARD", "ENTITY", entity_slot=1)],
    )
    policy = P.make_policy("survival", catalog=CATALOG)
    assert policy.select(obs) == 1

    obs = observation(
        phase="DEPLOY",
        animals=[animal(owner=0, position=0)],
        hand=hand,
        prefix=[
            choice("TYPE", "TYPE", type_id=COMMAND["gameDeployTraitRequest"]),
            choice("HAND_CARD", "ENTITY", entity_slot=1),
        ],
        choices=[
            choice("BOOLEAN", "SCALAR", scalar=0),
            choice("BOOLEAN", "SCALAR", scalar=1),
        ],
    )
    assert policy.select(obs) == 0


def test_survival_deploy_passes_without_trait_choice():
    obs = observation(
        phase="DEPLOY",
        animals=[animal(owner=0)],
        choices=[choice("TYPE", "TYPE", type_id=COMMAND["gameEndTurnRequest"])],
    )
    assert P.make_policy("survival").select(obs) == 0


def test_survival_feeding_food_then_min_need_then_plant():
    animals = [
        animal(owner=0, position=0, food=0, food_size=3),
        animal(owner=0, position=1, food=1, food_size=2),
    ]
    plants = [
        {"type": 1, "food": 0, "covers": 0, "cover_slots": 0, "fruit": 0, "traits": []},
        {"type": 1, "food": 2, "covers": 0, "cover_slots": 0, "fruit": 0, "traits": []},
    ]
    obs = observation(
        phase="FEEDING",
        animals=animals,
        plants=plants,
        choices=[
            choice("TYPE", "TYPE", type_id=COMMAND["gameEndTurnRequest"]),
            choice("TYPE", "TYPE", type_id=COMMAND["traitTakeFoodRequest"]),
        ],
    )
    policy = P.make_policy("survival", catalog=CATALOG)
    assert policy.select(obs) == 1

    obs = observation(
        phase="FEEDING",
        animals=animals,
        plants=plants,
        prefix=[choice("TYPE", "TYPE", type_id=COMMAND["traitTakeFoodRequest"])],
        choices=[
            choice("ANIMAL", "ENTITY", entity_kind=1, entity_slot=0),
            choice("ANIMAL", "ENTITY", entity_kind=1, entity_slot=1),
        ],
    )
    assert policy.select(obs) == 1

    obs = observation(
        phase="FEEDING",
        animals=animals,
        plants=plants,
        prefix=[
            choice("TYPE", "TYPE", type_id=COMMAND["traitTakeFoodRequest"]),
            choice("ANIMAL", "ENTITY", entity_kind=1, entity_slot=1),
        ],
        choices=[
            choice("TARGET", "ENTITY", entity_kind=2, entity_slot=0),
            choice("TARGET", "ENTITY", entity_kind=2, entity_slot=1),
        ],
    )
    assert policy.select(obs) == 1


def test_survival_feeding_fat_tissue_then_pass():
    fat_animal = animal(
        owner=0,
        food=0,
        food_size=2,
        fat_size=1,
        traits=[trait_row(5)],
    )
    obs = observation(
        phase="FEEDING",
        animals=[fat_animal],
        plants=[],
        choices=[
            choice("TYPE", "TYPE", type_id=COMMAND["gameEndTurnRequest"]),
            choice("TYPE", "TYPE", type_id=COMMAND["traitActivateRequest"]),
        ],
    )
    policy = P.make_policy("survival", catalog=CATALOG)
    assert policy.select(obs) == 1

    obs = observation(
        phase="FEEDING",
        animals=[fat_animal],
        prefix=[
            choice("TYPE", "TYPE", type_id=COMMAND["traitActivateRequest"]),
            choice("ANIMAL", "ENTITY", entity_kind=1, entity_slot=0),
        ],
        choices=[choice("TRAIT", "TRAIT", entity_kind=1, entity_slot=0, trait_slot=0, type_id=5)],
    )
    assert policy.select(obs) == 0

    # No fat trait and no food -> pass.
    obs = observation(
        phase="FEEDING",
        animals=[animal(owner=0, food=0, food_size=2)],
        plants=[],
        choices=[
            choice("TYPE", "TYPE", type_id=COMMAND["gameEndTurnRequest"]),
            choice("TYPE", "TYPE", type_id=COMMAND["traitActivateRequest"]),
        ],
    )
    assert policy.select(obs) == 0


# ------------------------------------------------------------------ questions
def test_defense_and_intellect_first_non_decline():
    trait_choice = choice("TRAIT", "TRAIT", entity_kind=1, entity_slot=0, trait_slot=0)
    decline = choice("DECLINE", "DECLINE")
    for kind in ("DEFENSE", "INTELLECT"):
        obs = observation(
            context_kind=kind,
            choices=[decline, trait_choice],
            mask=[1, 1],
        )
        assert P.make_policy("survival").select(obs) == 1
        obs = observation(context_kind=kind, choices=[decline], mask=[1])
        assert P.make_policy("survival").select(obs) == 0


def test_ambush_yes_then_continue():
    policy = P.make_policy("survival")
    obs = observation(
        context_kind="AMBUSH",
        choices=[
            choice("ANIMAL", "ENTITY", entity_kind=1, entity_slot=0),
            choice("DECLINE", "DECLINE"),
        ],
    )
    assert policy.select(obs) == 0

    obs = observation(
        context_kind="AMBUSH",
        choices=[
            choice("BOOLEAN", "SCALAR", scalar=0),
            choice("BOOLEAN", "SCALAR", scalar=1),
        ],
    )
    assert policy.select(obs) == 1

    obs = observation(
        context_kind="AMBUSH",
        choices=[choice("DECLINE", "DECLINE")],
    )
    assert policy.select(obs) == 0


def test_regeneration_first_card_then_default():
    policy = P.make_policy("survival")
    obs = observation(
        context_kind="REGENERATION",
        animals=[
            animal(owner=0, position=0, flags=[0, 0, 0, 0, 0, 0, 0, 0]),
            animal(owner=0, position=1, flags=[0, 0, 0, 1, 0, 0, 0, 0]),
        ],
        choices=[
            choice("ANIMAL", "ENTITY", entity_kind=1, entity_slot=0),
            choice("ANIMAL", "ENTITY", entity_kind=1, entity_slot=1),
            choice("USE_DEFAULT", "USE_DEFAULT"),
        ],
    )
    assert policy.select(obs) == 1

    obs = observation(
        context_kind="REGENERATION",
        choices=[choice("USE_DEFAULT", "USE_DEFAULT")],
    )
    assert policy.select(obs) == 0

    obs = observation(
        context_kind="REGENERATION",
        animals=[
            animal(owner=0, position=0, flags=[0, 0, 0, 1, 0, 0, 0, 0]),
        ],
        prefix=[
            choice("ANIMAL", "ENTITY", entity_kind=1, entity_slot=0),
        ],
        choices=[
            choice("HAND_CARD", "ENTITY", entity_slot=0),
            choice("HAND_CARD", "ENTITY", entity_slot=1),
        ],
        hand=[
            {"card_type": 1, "trait1_type": 6, "trait2_type": 0},
            {"card_type": 1, "trait1_type": 5, "trait2_type": 0},
        ],
    )
    assert policy.select(obs) == 0


def test_transient_true_then_ready():
    policy = P.make_policy("survival", catalog=CATALOG)
    obs = observation(
        context_kind="TRANSIENT",
        choices=[
            choice("BOOLEAN", "SCALAR", scalar=0),
            choice("BOOLEAN", "SCALAR", scalar=1, type_id=9),
            choice("READY", "READY"),
        ],
    )
    assert policy.select(obs) == 1

    obs = observation(context_kind="TRANSIENT", choices=[choice("READY", "READY")])
    assert policy.select(obs) == 0


# ------------------------------------------------------------------- privacy
def test_policies_ignore_non_whitelisted_keys():
    choices = [choice("HAND_CARD", "ENTITY", entity_slot=index) for index in range(3)]
    obs = observation(choices=choices, mask=[1, 1, 1])
    hidden = json.loads(json.dumps(obs))
    hidden["authoritative"] = {"deck": [1, 2, 3], "seed": 99}
    hidden["global"]["seed"] = 12345
    hidden["players"] = [{"seat": 0, "score": 99}]
    hidden["animals"].append({"owner": 1, "hidden_trait": 5})

    random_a = P.make_policy("random", rng=np.random.default_rng(7))
    random_b = P.make_policy("random", rng=np.random.default_rng(7))
    assert [random_a.select(obs) for _ in range(10)] == [
        random_b.select(hidden) for _ in range(10)
    ]
    assert P.make_policy("survival").select(obs) == P.make_policy("survival").select(hidden)


def test_make_policy_rejects_unknown_name():
    with pytest.raises(ValueError):
        P.make_policy("oracle")
    assert P.make_policy("random", rng=np.random.default_rng(0)).name == "random"
    assert P.make_policy("survival").name == "survival"
