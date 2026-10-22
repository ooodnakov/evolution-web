"""Gymnasium observation spaces for the Evolution RL environment.

The authoritative structure lives in ``rl/schema.json`` (repository root).
This module only translates that JSON into Gymnasium spaces; it never
duplicates the field list by hand.

Mapping rules (mirrors the schema contract):

* nested named blocks            -> ``spaces.Dict``
* ``{"sequence": node}``         -> ``spaces.Sequence(node)``
* ``{"enum": name}``             -> ``spaces.Discrete(len(schema[name]))``
* ``{"int": true}``              -> ``Box(-1, 2**31 - 1, (), int64)``
* ``{"bool": true}``             -> ``MultiBinary(1)``
* ``{"multibinary": name}``      -> ``MultiBinary(len(schema[name]))``
* ``{"int8": true}``             -> ``Box(0, 1, (), int8)``

Keys and their order inside ``Dict`` spaces come from the schema; Gymnasium
itself sorts ``Dict`` keys internally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

__all__ = [
    "INT_MAX",
    "SCHEMA_VERSION",
    "SCHEMA_PATH",
    "observation_space",
    "space_for",
    "validate_observation",
    "change_space",
    "VariableSequence",
]

INT_MAX = 2**31 - 1


class VariableSequence(spaces.Sequence):
    """``Sequence`` that accepts empty and list-typed collections.

    Gymnasium's ``Sequence.contains`` requires a ``tuple`` and rejects ``[]``,
    yet zero players, animals, plants, shells, traits, cooldowns or choices are
    valid states of the Evolution table, and observations arrive as lists. The
    schema contract is therefore any finite (including empty) collection of
    valid rows.
    """

    def contains(self, x: Any) -> bool:
        if isinstance(x, (list, tuple)):
            return all(self.feature_space.contains(item) for item in x)
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return True
            return all(self.feature_space.contains(item) for item in x.tolist())
        return super().contains(x)


def _find_schema_path() -> Path:
    """Locate ``rl/schema.json`` without any network or package data copy."""
    here = Path(__file__).resolve()
    candidates = [parent / "rl" / "schema.json" for parent in here.parents]
    cwd = Path.cwd().resolve()
    candidates.extend(parent / "rl" / "schema.json" for parent in (cwd, *cwd.parents))
    candidates.append(here.parent / "schema.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "rl/schema.json not found; the Evolution RL schema is required to build spaces"
    )


SCHEMA_PATH = _find_schema_path()


def _load_schema(path: Path | None = None) -> dict[str, Any]:
    with open(path or SCHEMA_PATH, encoding="utf-8") as schema_file:
        return json.load(schema_file)


_SCHEMA = _load_schema()
SCHEMA_VERSION: int = _SCHEMA["schema_version"]


def _build_node(schema: Mapping[str, Any], node: Any) -> spaces.Space:
    if isinstance(node, str):
        # Named block reference, e.g. "trait_row" / "choice_row".
        return _build_node(schema, schema[node])

    if not isinstance(node, dict):
        raise TypeError(f"unsupported schema node: {node!r}")

    if "enum" in node:
        return spaces.Discrete(len(schema[node["enum"]]))
    if "int" in node:
        return spaces.Box(low=-1, high=INT_MAX, shape=(), dtype=np.int64)
    if "bool" in node:
        return spaces.MultiBinary(1)
    if "int8" in node:
        return spaces.Box(low=0, high=1, shape=(), dtype=np.int8)
    if "multibinary" in node:
        return spaces.MultiBinary(len(schema[node["multibinary"]]))
    if "sequence" in node:
        return VariableSequence(_build_node(schema, node["sequence"]))

    return spaces.Dict(
        {key: _build_node(schema, value) for key, value in node.items()}
    )


def space_for(schema_path: str | Path | None = None) -> spaces.Dict:
    """Build the observation space for a schema file (default: repo schema)."""
    schema = _SCHEMA if schema_path is None else _load_schema(Path(schema_path))
    space = _build_node(schema, schema["observation"])
    if not isinstance(space, spaces.Dict):
        raise TypeError("schema observation must be a named block of fields")
    return space


def observation_space() -> spaces.Dict:
    """Return the one stable observation space built from ``rl/schema.json``."""
    return _OBSERVATION_SPACE


_OBSERVATION_SPACE = space_for()


def _coerce_data(space: spaces.Space, data: Any) -> Any:
    """Convert plain Python containers into the arrays Gymnasium expects."""
    if isinstance(space, spaces.Dict):
        if not isinstance(data, Mapping):
            return data
        return {
            key: _coerce_data(sub_space, data[key])
            for key, sub_space in space.spaces.items()
            if key in data
        }
    if isinstance(space, spaces.Sequence):
        try:
            items = list(data)
        except TypeError:
            return data
        return [_coerce_data(space.feature_space, item) for item in items]
    if isinstance(space, spaces.MultiBinary):
        return np.asarray(data, dtype=np.int8).reshape(space.n)
    if isinstance(space, spaces.Box):
        return np.asarray(data, dtype=space.dtype)
    return data


def validate_observation(space: spaces.Space, obs: Any) -> bool:
    """True when ``obs`` (ordinary lists allowed) belongs to ``space``."""
    try:
        return bool(space.contains(_coerce_data(space, obs)))
    except (TypeError, ValueError, KeyError):
        return False


def change_space(n: int) -> spaces.Discrete:
    """Action space for one choice prefix level with ``n`` live options."""
    return spaces.Discrete(max(1, int(n)))
