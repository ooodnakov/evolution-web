"""Contract tests for ``evolution_rl.spaces`` against ``rl/schema.json``."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from evolution_rl import spaces as rl_spaces

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((REPO_ROOT / "rl" / "schema.json").read_text(encoding="utf-8"))

EXPECTED_KEYS = {
    "global",
    "players",
    "hand",
    "animals",
    "plants",
    "shells",
    "context",
    "choices",
    "action_mask",
}


def trait_row(**overrides):
    row = {
        "type": 1,
        "disabled": np.zeros(1, dtype=np.int8),
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


def choice_row(**overrides):
    row = {
        "field_kind": 0,
        "choice_kind": 1,
        "type_id": 0,
        "entity_kind": 0,
        "entity_slot": -1,
        "trait_slot": -1,
        "scalar": 0,
    }
    row.update(overrides)
    return row


def animal(traits=()):
    return {
        "owner": 0,
        "position": 0,
        "food": 0,
        "food_size": 1,
        "fat_size": 0,
        "flags": np.zeros(len(SCHEMA["animal_flags"]), dtype=np.int8),
        "traits": list(traits),
    }


def observation(animals=None, choices=None, action_mask=None):
    return {
        "global": {
            "phase": 0,
            "turn": 1,
            "round": 1,
            "seat": 0,
            "current_player": 0,
            "round_player": 0,
            "food": 10,
            "deck_count": 4,
            "active_players": 2,
            "addons": [0] * len(SCHEMA["addons"]),
            "half_deck": np.zeros(1, dtype=np.int8),
        },
        "players": [
            {
                "seat": 0,
                "playing": np.ones(1, dtype=np.int8),
                "ended": np.zeros(1, dtype=np.int8),
                "acted": np.zeros(1, dtype=np.int8),
                "hand_count": 1,
                "animal_count": 1,
            },
            {
                "seat": 1,
                "playing": np.ones(1, dtype=np.int8),
                "ended": np.zeros(1, dtype=np.int8),
                "acted": np.ones(1, dtype=np.int8),
                "hand_count": 3,
                "animal_count": 0,
            },
        ],
        "hand": [{"card_type": 1, "trait1_type": 2, "trait2_type": 0}],
        "animals": [animal([trait_row()])] if animals is None else animals,
        "plants": [
            {
                "type": 1,
                "food": 2,
                "covers": 0,
                "cover_slots": 1,
                "fruit": 0,
                "traits": [],
            }
        ],
        "shells": [{"type": 5, "value": 1, "slot": 0}],
        "context": {
            "kind": 0,
            "source_kind": 1,
            "source_slot": 0,
            "target_kind": 0,
            "target_slot": -1,
            "attack_trait_type": 0,
            "prefix": [],
        },
        "choices": [choice_row()] if choices is None else choices,
        "action_mask": (
            [np.int8(1)] if action_mask is None else action_mask
        ),
    }


def test_observation_space_top_level_keys():
    space = rl_spaces.observation_space()
    assert isinstance(space, spaces.Dict)
    assert set(space.spaces) == EXPECTED_KEYS


def test_observation_space_is_stable():
    assert rl_spaces.observation_space() is rl_spaces.observation_space()
    assert rl_spaces.space_for() == rl_spaces.observation_space()


def test_minimal_sample_is_contained():
    space = rl_spaces.observation_space()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert space.contains(observation())
    assert rl_spaces.validate_observation(space, observation()) is True


def test_validate_accepts_plain_lists():
    """Wire observations arrive as JSON lists; validation coerces them."""
    space = rl_spaces.observation_space()
    sample = observation()
    sample["global"]["half_deck"] = 0
    sample["players"][0]["playing"] = 1
    sample["players"][0]["ended"] = 0
    sample["players"][0]["acted"] = 0
    sample["animals"][0]["flags"] = [0] * len(SCHEMA["animal_flags"])
    sample["animals"][0]["traits"][0]["disabled"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert space.contains(sample) is False
    assert rl_spaces.validate_observation(space, sample) is True


def test_invalid_samples_rejected():
    space = rl_spaces.observation_space()

    missing_key = observation()
    del missing_key["global"]
    assert rl_spaces.validate_observation(space, missing_key) is False

    bad_enum = observation()
    bad_enum["global"]["phase"] = len(SCHEMA["phase_values"])
    assert rl_spaces.validate_observation(space, bad_enum) is False

    bad_sequence_item = observation(animals=[{"owner": 0}])
    assert rl_spaces.validate_observation(space, bad_sequence_item) is False

    assert rl_spaces.validate_observation(space, None) is False


@pytest.mark.parametrize("count", [0, 1, 3])
def test_animal_sequence_arity(count):
    space = rl_spaces.observation_space()
    animals = [
        animal([trait_row(type=index + 1) for index in range(index % 3)])
        for index in range(count)
    ]
    assert rl_spaces.validate_observation(space, observation(animals=animals)) is True


@pytest.mark.parametrize("trait_count", [0, 1, 2])
def test_trait_sequence_arity(trait_count):
    space = rl_spaces.observation_space()
    animals = [animal([trait_row(type=index + 1) for index in range(trait_count)])]
    assert rl_spaces.validate_observation(space, observation(animals=animals)) is True


def test_enums_sized_from_schema():
    space = rl_spaces.observation_space()
    assert space["global"]["phase"].n == len(SCHEMA["phase_values"])
    assert space["context"]["kind"].n == len(SCHEMA["decision_kinds"])
    assert space["context"]["source_kind"].n == len(SCHEMA["entity_kinds"])
    assert space["context"]["target_kind"].n == len(SCHEMA["entity_kinds"])
    assert space["choices"].feature_space["field_kind"].n == len(
        SCHEMA["choice_field_kinds"]
    )
    assert space["choices"].feature_space["choice_kind"].n == len(SCHEMA["choice_kinds"])
    trait_space = space["animals"].feature_space["traits"].feature_space
    assert trait_space["linked_entity_kind"].n == len(SCHEMA["entity_kinds"])
    assert trait_space["value_kind"].n == len(SCHEMA["value_kinds"])
    assert trait_space["source_type"].n == len(SCHEMA["feeding_source_types"])


def test_scalar_and_multibinary_shapes():
    space = rl_spaces.observation_space()
    assert space["global"]["food"].dtype == np.int64
    assert space["global"]["phase"].dtype == np.int64
    assert space["global"]["addons"].n == len(SCHEMA["addons"])
    assert space["global"]["half_deck"].n == 1
    flags = space["animals"].feature_space["flags"]
    assert flags.n == len(SCHEMA["animal_flags"])
    assert space["action_mask"].feature_space.dtype == np.int8
    mask = observation(action_mask=[np.int8(1), np.int8(0), np.int8(1)])
    assert rl_spaces.validate_observation(space, mask) is True


def test_schema_version_matches_json():
    assert rl_spaces.SCHEMA_VERSION == SCHEMA["schema_version"]


def test_change_space():
    assert rl_spaces.change_space(0) == spaces.Discrete(1)
    assert rl_spaces.change_space(4) == spaces.Discrete(4)
    assert rl_spaces.change_space(-3) == spaces.Discrete(1)
