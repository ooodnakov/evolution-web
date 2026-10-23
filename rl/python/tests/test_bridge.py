"""Unit tests for the JSONL bridge using fake child processes.

These tests intentionally do not need ``rl/node/environment.js``: every child
is a tiny ``python -c`` script speaking the worker protocol. End-to-end AEC
checks belong to the integration owner once the engine exists.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from evolution_rl import bridge
from evolution_rl.bridge import EngineProcess, EngineProcessError

REPO_ROOT = bridge.find_repo_root() or Path(__file__).resolve().parents[3]

ECHO_SCRIPT = r"""
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    result = {"op": req["op"], "id": req["id"], "payload": req.get("payload")}
    sys.stdout.write(json.dumps({"id": req["id"], "ok": True, "result": result}) + "\n")
    sys.stdout.flush()
    if req["op"] == "close":
        break
"""

FRAME_SCRIPT = r"""
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    payload = req.get("payload") or {}
    if req["op"] == "snapshot":
        result = {"handle": 5}
    elif req["op"] == "close":
        result = {"closed": True}
    else:
        result = {
            "actor": "player_0",
            "decisionId": 7,
            "observations": {"player_0": {"choices": [{}]}},
            "rewards": {},
            "terminated": {},
            "truncated": {},
            "info": {},
            "committed": True,
        }
    sys.stdout.write(json.dumps({"id": req["id"], "ok": True, "result": result}) + "\n")
    sys.stdout.flush()
    if req["op"] == "close":
        break
"""

PARTIAL_SCRIPT = r"""
import json, sys, time
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    payload = json.dumps({"id": req["id"], "ok": True, "result": {"partial": True}})
    midpoint = len(payload) // 2
    sys.stdout.write(payload[:midpoint])
    sys.stdout.flush()
    time.sleep(0.2)
    sys.stdout.write(payload[midpoint:] + "\n")
    sys.stdout.flush()
"""

WRONG_ID_SCRIPT = r"""
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    response = {"id": int(req["id"]) + 1000, "ok": True, "result": {}}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
"""

MALFORMED_SCRIPT = r"""
import sys
sys.stdin.readline()
sys.stdout.write("this is not json\n")
sys.stdout.flush()
"""

ERROR_SCRIPT = r"""
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    error = {"code": "INVALID_CONFIG", "message": "bad config from fake child"}
    sys.stdout.write(json.dumps({"id": req["id"], "ok": False, "error": error}) + "\n")
    sys.stdout.flush()
"""

EOF_SCRIPT = r"""
import sys
sys.exit(3)
"""

STDERR_SCRIPT = r"""
import json, sys
sys.stderr.write("fake-child-stderr-line\n")
sys.stderr.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    sys.stdout.write(json.dumps({"id": req["id"], "ok": True, "result": {}}) + "\n")
    sys.stdout.flush()
    if req["op"] == "close":
        break
"""

SLOW_SCRIPT = r"""
import sys, time
for line in sys.stdin:
    time.sleep(0.5)
"""


def make_process(script: str, **kwargs) -> EngineProcess:
    return EngineProcess(
        repo_root=REPO_ROOT,
        argv=[sys.executable, "-u", "-c", script],
        timeout=5.0,
        **kwargs,
    )


def test_request_ids_are_monotonic() -> None:
    process = make_process(ECHO_SCRIPT)
    try:
        first = process.request("observe", {"agent": "player_0"})
        second = process.request("hello", {"config": {}})
        assert first["id"] == 1
        assert first["op"] == "observe"
        assert first["payload"] == {"agent": "player_0"}
        assert second["id"] == 2
    finally:
        process.close()


def test_partial_reads_are_reassembled() -> None:
    process = make_process(PARTIAL_SCRIPT)
    try:
        result = process.request("observe", {"agent": "player_0"})
        assert result == {"partial": True}
    finally:
        process.close()


def test_frame_is_normalised_to_snake_case() -> None:
    process = make_process(FRAME_SCRIPT)
    try:
        frame = process.reset(seed=0)
        assert frame["actor"] == "player_0"
        assert frame["decision_id"] == 7
        assert "decisionId" not in frame
        assert frame["committed"] is True
        assert process.snapshot() == 5
        assert process.restore(5)["decision_id"] == 7
    finally:
        process.close()


def test_wrong_response_id_is_a_protocol_error() -> None:
    process = make_process(WRONG_ID_SCRIPT)
    try:
        with pytest.raises(EngineProcessError) as excinfo:
            process.request("observe", {"agent": "player_0"})
        assert excinfo.value.code == "ENGINE_PROTOCOL"
        assert "id mismatch" in str(excinfo.value)
    finally:
        process.close()


def test_malformed_response_json_is_a_protocol_error() -> None:
    process = make_process(MALFORMED_SCRIPT)
    try:
        with pytest.raises(EngineProcessError) as excinfo:
            process.request("observe", {"agent": "player_0"})
        assert excinfo.value.code == "ENGINE_PROTOCOL"
    finally:
        process.close()


def test_worker_error_code_is_preserved() -> None:
    process = make_process(ERROR_SCRIPT)
    try:
        with pytest.raises(EngineProcessError) as excinfo:
            process.hello({"numPlayers": 2})
        assert excinfo.value.code == "INVALID_CONFIG"
        assert "bad config from fake child" in str(excinfo.value)
    finally:
        process.close()


def test_eof_raises_instead_of_retrying() -> None:
    process = make_process(EOF_SCRIPT)
    try:
        with pytest.raises(EngineProcessError) as excinfo:
            process.request("hello", {"config": {}})
        assert excinfo.value.code in {"ENGINE_EOF", "ENGINE_DEAD"}
    finally:
        process.close()


def test_timeout_raises_engine_timeout() -> None:
    process = make_process(SLOW_SCRIPT)
    try:
        with pytest.raises(EngineProcessError) as excinfo:
            process.request("observe", {"agent": "player_0"}, timeout=0.15)
        assert excinfo.value.code == "ENGINE_TIMEOUT"
    finally:
        process.close()


def test_stderr_is_drained_into_tail() -> None:
    process = make_process(STDERR_SCRIPT, forward_stderr=False)
    try:
        process.request("hello", {"config": {}})
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if any("fake-child-stderr-line" in line for line in process.stderr_tail):
                break
            time.sleep(0.02)
        assert any("fake-child-stderr-line" in line for line in process.stderr_tail)
    finally:
        process.close()


def test_close_is_idempotent_and_requests_are_rejected_afterwards() -> None:
    process = make_process(ECHO_SCRIPT)
    process.close()
    assert process.process.poll() is not None
    process.close()
    assert process.process.poll() is not None
    with pytest.raises(EngineProcessError) as excinfo:
        process.request("hello", {"config": {}})
    assert excinfo.value.code == "ENGINE_CLOSED"


def test_context_manager_closes_process() -> None:
    with make_process(ECHO_SCRIPT) as process:
        process.request("observe", {"agent": "player_0"})
        proc = process.process
    assert proc.poll() is not None


def test_missing_launcher_raises_engine_not_found(tmp_path: Path) -> None:
    with pytest.raises(EngineProcessError) as excinfo:
        EngineProcess(repo_root=tmp_path)
    assert excinfo.value.code == "ENGINE_NOT_FOUND"


def test_default_runtime_and_launcher_target_the_compiled_build() -> None:
    assert bridge.DEFAULT_RUNTIME_EXECUTABLE == "bun"
    assert bridge.LAUNCHER_RELPATH == Path("dist") / "rl" / "node" / "worker.cjs"


def test_missing_build_points_at_the_build_command(tmp_path: Path) -> None:
    with pytest.raises(EngineProcessError) as excinfo:
        EngineProcess(repo_root=tmp_path)
    assert "npm run rl:build" in excinfo.value.message


def test_missing_runtime_raises_engine_runtime_not_found(tmp_path: Path) -> None:
    launcher = tmp_path / bridge.LAUNCHER_RELPATH
    launcher.parent.mkdir(parents=True)
    launcher.write_text("// compiled worker stub\n", encoding="utf-8")
    schema = tmp_path / bridge.SCHEMA_RELPATH
    schema.parent.mkdir(parents=True)
    schema.write_text("{}", encoding="utf-8")
    with pytest.raises(EngineProcessError) as excinfo:
        EngineProcess(repo_root=tmp_path, runtime_executable="definitely-not-bun-1-4-4")
    assert excinfo.value.code == "ENGINE_RUNTIME_NOT_FOUND"
    assert "npm run rl:build" in excinfo.value.message


def test_missing_repo_root_raises_engine_not_found(tmp_path: Path) -> None:
    with pytest.raises(EngineProcessError) as excinfo:
        EngineProcess(repo_root=tmp_path / "does-not-exist")
    assert excinfo.value.code == "ENGINE_NOT_FOUND"
