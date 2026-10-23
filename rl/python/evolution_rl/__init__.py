"""Evolution RL: Gymnasium/PettingZoo access to the headless Evolution engine.

Public API::

    from evolution_rl import env, raw_env

    aec = env()                 # OrderEnforcingWrapper
    aec = raw_env(config={"numPlayers": 4})

Importing this package is cheap: neither PettingZoo nor the Node engine is
loaded until an environment is actually created.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import bridge

__all__ = [
    "EvolutionAECEnv",
    "env",
    "raw_env",
    "SCHEMA_VERSION",
]


def raw_env(
    config: Mapping[str, Any] | None = None,
    render_mode: str | None = None,
    runtime_executable: str = bridge.DEFAULT_RUNTIME_EXECUTABLE,
) -> Any:
    """Create the unwrapped :class:`evolution_rl.aec.EvolutionAECEnv`."""
    from .aec import EvolutionAECEnv

    return EvolutionAECEnv(
        config=config, render_mode=render_mode, runtime_executable=runtime_executable
    )


def env(
    config: Mapping[str, Any] | None = None,
    render_mode: str | None = None,
    runtime_executable: str = bridge.DEFAULT_RUNTIME_EXECUTABLE,
) -> Any:
    """Create the AEC environment wrapped in ``OrderEnforcingWrapper``."""
    from pettingzoo.utils import OrderEnforcingWrapper

    return OrderEnforcingWrapper(
        raw_env(
            config=config, render_mode=render_mode, runtime_executable=runtime_executable
        )
    )


def __getattr__(name: str) -> Any:
    if name == "EvolutionAECEnv":
        from .aec import EvolutionAECEnv

        return EvolutionAECEnv
    if name == "SCHEMA_VERSION":
        from .spaces import SCHEMA_VERSION

        return SCHEMA_VERSION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
