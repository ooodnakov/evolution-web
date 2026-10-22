"""JSONL transport to the headless Evolution engine (``rl/node/worker.cjs``).

The Node worker owns the authoritative game. This module is deliberately
dumb: one long-lived child process per environment, one request in flight at
a time, ids must match, and protocol corruption is fatal. It never retries or
replays an episode on its own.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "EngineProcess",
    "EngineProcessError",
    "find_repo_root",
    "LAUNCHER_RELPATH",
    "SCHEMA_RELPATH",
]

LAUNCHER_RELPATH = Path("rl") / "node" / "worker.cjs"
SCHEMA_RELPATH = Path("rl") / "schema.json"

_EOF = object()


class EngineProcessError(RuntimeError):
    """Raised when the worker cannot be spawned or breaks the protocol."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = str(code)
        self.message = message or self.code
        super().__init__(f"{self.code}: {self.message}")


def _has_engine(root: Path) -> bool:
    return (root / LAUNCHER_RELPATH).is_file() and (root / SCHEMA_RELPATH).is_file()


def find_repo_root(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate the repository checkout that owns ``rl/node/worker.cjs``."""
    if start is not None:
        candidate = Path(start).expanduser().resolve()
        return candidate if _has_engine(candidate) else None

    here = Path(__file__).resolve()
    for parent in here.parents:
        if _has_engine(parent):
            return parent

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if _has_engine(parent):
            return parent
    return None


class EngineProcess:
    """One long-lived engine subprocess speaking newline-delimited JSON."""

    def __init__(
        self,
        node_executable: str = "node",
        repo_root: str | os.PathLike[str] | None = None,
        config: dict[str, Any] | None = None,
        *,
        argv: Sequence[str] | None = None,
        timeout: float = 30.0,
        forward_stderr: bool = True,
        stderr_limit: int = 2000,
    ) -> None:
        self.node_executable = node_executable
        self.timeout = float(timeout)
        self.forward_stderr = bool(forward_stderr)
        self.stderr_lines: deque[str] = deque(maxlen=max(1, int(stderr_limit)))
        self.config = dict(config) if config is not None else None

        if repo_root is not None:
            root = Path(repo_root).expanduser().resolve()
            if not root.is_dir():
                raise EngineProcessError(
                    "ENGINE_NOT_FOUND", f"repository root does not exist: {root}"
                )
        else:
            root = find_repo_root()
            if root is None:
                raise EngineProcessError(
                    "ENGINE_NOT_FOUND", "rl/node/worker.cjs not found in any parent directory"
                )

        if argv is None:
            if not (root / LAUNCHER_RELPATH).is_file():
                raise EngineProcessError(
                    "ENGINE_NOT_FOUND", f"missing launcher: {root / LAUNCHER_RELPATH}"
                )
            command = [node_executable, str(LAUNCHER_RELPATH)]
        else:
            command = [str(part) for part in argv]
            if not command:
                raise EngineProcessError("ENGINE_NOT_FOUND", "empty engine command")

        self.repo_root = root
        self._command: tuple[str, ...] = tuple(command)
        self._next_id = 1
        self._lock = threading.Lock()
        self._responses: queue.Queue[Any] = queue.Queue()
        self._closed = False

        try:
            self._proc = subprocess.Popen(
                list(self._command),
                cwd=str(root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                shell=False,
            )
        except OSError as exc:
            raise EngineProcessError(
                "ENGINE_NOT_FOUND", f"cannot start engine process: {exc}"
            ) from exc

        self._stdout_thread = threading.Thread(
            target=self._drain_stdout, name="evolution-engine-stdout", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="evolution-engine-stderr", daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        if self.config is not None:
            try:
                self.hello(self.config)
            except BaseException:
                self.close()
                raise

    # ------------------------------------------------------------------ io
    @property
    def process(self) -> subprocess.Popen[str]:
        return self._proc

    @property
    def stderr_tail(self) -> list[str]:
        return list(self.stderr_lines)

    def _drain_stdout(self) -> None:
        stream = self._proc.stdout
        try:
            if stream is None:
                return
            for line in stream:
                if line:
                    self._responses.put(line)
        except (ValueError, OSError):
            pass
        finally:
            self._responses.put(_EOF)

    def _drain_stderr(self) -> None:
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            for line in stream:
                if not line:
                    continue
                text = line.rstrip("\n")
                self.stderr_lines.append(text)
                if self.forward_stderr:
                    try:
                        err = sys.stderr
                        if err is not None:
                            err.write(text + "\n")
                            err.flush()
                    except (ValueError, OSError):
                        pass
        except (ValueError, OSError):
            pass

    def _roundtrip(self, op: str, payload: dict[str, Any] | None, timeout: float) -> Any:
        if self._closed:
            raise EngineProcessError("ENGINE_CLOSED", "engine process is closed")

        request_id = self._next_id
        self._next_id += 1
        request = {"id": request_id, "op": op, "payload": payload or {}}
        line = json.dumps(request, separators=(",", ":"), ensure_ascii=False)

        stdin = self._proc.stdin
        if stdin is None:
            raise EngineProcessError("ENGINE_DEAD", "engine stdin is not available")
        try:
            stdin.write(line + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise EngineProcessError(
                "ENGINE_DEAD", f"engine process is not accepting input: {exc}"
            ) from exc

        try:
            item = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            tail = "\n".join(self.stderr_tail[-20:])
            hint = f"\nstderr tail:\n{tail}" if tail else ""
            raise EngineProcessError(
                "ENGINE_TIMEOUT", f"no response to {op!r} within {timeout}s{hint}"
            ) from exc

        if item is _EOF:
            returncode = self._proc.poll()
            raise EngineProcessError(
                "ENGINE_EOF",
                f"engine process closed the protocol stream (exit code {returncode})",
            )

        raw = item.strip()
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EngineProcessError(
                "ENGINE_PROTOCOL", f"invalid JSON from engine: {raw[:500]!r}"
            ) from exc

        if not isinstance(response, dict):
            raise EngineProcessError(
                "ENGINE_PROTOCOL", f"non-object response from engine: {raw[:500]!r}"
            )
        if response.get("id") != request_id:
            raise EngineProcessError(
                "ENGINE_PROTOCOL",
                f"response id mismatch: expected {request_id}, got {response.get('id')!r}",
            )
        if not response.get("ok"):
            error = response.get("error")
            if isinstance(error, dict):
                code = str(error.get("code") or "INTERNAL")
                message = str(error.get("message") or code)
            else:
                code, message = "INTERNAL", str(error)
            raise EngineProcessError(code, message)
        return response.get("result")

    def request(
        self, op: str, payload: dict[str, Any] | None = None, timeout: float | None = None
    ) -> Any:
        """Send one request and return its ``result`` payload."""
        with self._lock:
            return self._roundtrip(op, payload, self.timeout if timeout is None else timeout)

    # ------------------------------------------------------------ protocol
    def hello(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"config": self.config if config is None else dict(config)}
        result = self.request("hello", payload)
        if not isinstance(result, dict):
            raise EngineProcessError("ENGINE_PROTOCOL", "hello result must be an object")
        return result

    def reset(
        self, seed: int | None = None, scenario: str | None = None
    ) -> dict[str, Any]:
        return _as_frame(self.request("reset", {"seed": seed, "scenario": scenario}))

    def observe(self, agent: str) -> dict[str, Any]:
        result = self.request("observe", {"agent": str(agent)})
        if not isinstance(result, dict):
            raise EngineProcessError("ENGINE_PROTOCOL", "observation must be an object")
        return result

    def step(self, decision_id: int, choice_index: int) -> dict[str, Any]:
        payload = {
            "decision_id": int(decision_id),
            "choice_index": int(choice_index),
        }
        return _as_frame(self.request("step", payload))

    def snapshot(self) -> int:
        result = self.request("snapshot", {})
        if isinstance(result, bool):
            raise EngineProcessError("ENGINE_PROTOCOL", "snapshot handle must be an integer")
        if isinstance(result, int):
            return int(result)
        if isinstance(result, dict) and isinstance(result.get("handle"), int):
            return int(result["handle"])
        raise EngineProcessError(
            "ENGINE_PROTOCOL", f"unexpected snapshot result: {result!r}"
        )

    def restore(self, handle: int) -> dict[str, Any]:
        return _as_frame(self.request("restore", {"handle": int(handle)}))

    def export_replay(self) -> dict[str, Any]:
        result = self.request("export_replay", {})
        if not isinstance(result, dict):
            raise EngineProcessError("ENGINE_PROTOCOL", "replay must be an object")
        return result

    # --------------------------------------------------------------- close
    def close(self) -> None:
        """Stop the worker; idempotent and safe from ``__del__``."""
        if self._closed:
            return

        proc = self._proc
        try:
            if proc.poll() is None:
                try:
                    # _roundtrip still works because _closed flips only after
                    # the courtesy close request (or its failure).
                    with self._lock:
                        self._roundtrip("close", {}, 1.0)
                except Exception:
                    # Best effort: the worker may already be gone or shutting
                    # down at interpreter exit.
                    pass
        finally:
            self._closed = True
            self._reap(proc)

    @staticmethod
    def _reap(proc: subprocess.Popen[str]) -> None:
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass

    # -------------------------------------------------------------- dunder
    def __enter__(self) -> "EngineProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _as_frame(result: Any) -> dict[str, Any]:
    """Normalise a Frame from the worker into snake_case Python keys."""
    if not isinstance(result, dict):
        raise EngineProcessError("ENGINE_PROTOCOL", f"Frame must be an object, got {type(result)!r}")
    frame = dict(result)
    if "decisionId" in frame:
        decision_id = frame.pop("decisionId")
        frame.setdefault("decision_id", decision_id)
    for key in ("observations", "rewards", "terminated", "truncated", "info"):
        frame.setdefault(key, {})
    frame.setdefault("committed", False)
    frame.setdefault("actor", None)
    return frame
