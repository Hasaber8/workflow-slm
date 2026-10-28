import hashlib
import json
from pathlib import Path
import random
import subprocess
import threading
from types import SimpleNamespace

import pytest

from exp.data.split import scenario_to_dict
from exp.runtime import codex
from exp.teacher import run_traces
from exp.world.scenarios import generate_scenario


@pytest.fixture
def cli(monkeypatch):
    state = {
        "login": "Logged in using ChatGPT",
        "returncode": 0,
        "response": {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate",
                     "window_minutes": 30, "text": None},
        "events": [{"type": "turn.completed", "usage": {
            "input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 40,
        }}],
        "calls": [],
    }
    monkeypatch.setattr(codex.shutil, "which", lambda command: "/fake/codex")

    def run(command, **kwargs):
        state["calls"].append((command, kwargs))
        if command[1:] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="", stderr=state["login"])
        if command[1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="codex-cli test", stderr="")
        if state.get("timeout"):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        state["schema"] = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        if state["response"] is not None:
            Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(state["response"]))
        return SimpleNamespace(
            returncode=state["returncode"], stderr="test diagnostic",
            stdout="\n".join(json.dumps(event) for event in state["events"]),
        )

    monkeypatch.setattr(codex.subprocess, "run", run)
    return state


def test_codex_structured_step_and_usage(cli, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("CODEX_API_KEY", "not-a-real-key")
    client = codex.CodexClient(model="gpt-6.1-terra")
    messages = [{"role": "system", "content": "Follow the SOP"}, {"role": "user", "content": "test alert"}]
    text, usage = client.complete(messages, {})
    assert json.loads(text) == {
        "step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30,
    }
    assert usage["input_tokens"] == 100
    assert usage["cache_read_input_tokens"] == 40
    assert usage["usage_available"] is True
    command, kwargs = cli["calls"][-1]
    assert command[command.index("--model") + 1] == "gpt-6.1-terra"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "--ephemeral" in command
    assert 'forced_login_method="chatgpt"' in command
    assert "shell_tool" in command and "multi_agent" in command
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "CODEX_API_KEY" not in kwargs["env"]
    assert json.dumps(messages) in kwargs["input"]
    assert kwargs["cwd"] != Path.cwd()
    assert cli["schema"]["type"] == "object"
    assert cli["schema"]["additionalProperties"] is False
    assert set(cli["schema"]["required"]) == set(cli["schema"]["properties"])
    assert all("default" not in prop for prop in cli["schema"]["properties"].values())


@pytest.mark.parametrize("login", ["Not logged in", "Logged in using an API key"])
def test_codex_requires_chatgpt_login(cli, login):
    cli["login"] = login
    with pytest.raises(codex.CodexError, match="ChatGPT subscription login"):
        codex.CodexClient().validate_setup()
    assert len(cli["calls"]) == 1


def test_codex_missing_cli_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(codex.shutil, "which", lambda command: None)
    with pytest.raises(codex.CodexError, match="CLI not found"):
        codex.CodexClient(command=str(tmp_path / "missing-codex")).validate_setup()


@pytest.mark.parametrize("failure", ["exit", "event", "missing", "invalid", "tool", "timeout"])
def test_codex_failures_stop_the_batch(cli, failure):
    if failure == "exit":
        cli["returncode"] = 1
    elif failure == "event":
        cli["events"] = [{"type": "turn.failed", "error": {"message": "usage limit reached"}}]
    elif failure == "missing":
        cli["response"] = None
    elif failure == "invalid":
        cli["response"] = {"step": "get_metrics"}
    elif failure == "tool":
        cli["events"] = [{"type": "item.completed", "item": {"type": "command_execution"}}]
    else:
        cli["timeout"] = True
    with pytest.raises(codex.CodexError):
        codex.CodexClient().complete([], {})


@pytest.mark.parametrize("diagnostic", [
    {"type": "error", "message": "Reconnecting... 2/5 (403 Forbidden on WebSocket)"},
    {"type": "turn.failed", "error": {"message": "Recoverable earlier turn failure"}},
])
def test_codex_accepts_a_valid_completed_turn_after_recovery(cli, diagnostic):
    cli["events"].insert(0, diagnostic)
    text, usage = codex.CodexClient().complete([], {})
    assert json.loads(text)["step"] == "get_metrics"
    assert usage["input_tokens"] == 100
    assert usage["recovered_cli_warnings"] == 1


def test_codex_rejects_diagnostics_without_a_completed_turn(cli):
    cli["events"] = [{"type": "error", "message": "403 Forbidden"}]
    with pytest.raises(codex.CodexError, match="403 Forbidden"):
        codex.CodexClient().complete([], {})


def test_codex_requires_the_final_turn_to_succeed(cli):
    cli["events"].append({"type": "turn.failed", "error": {"message": "final failure"}})
    with pytest.raises(codex.CodexError, match="final failure"):
        codex.CodexClient().complete([], {})


@pytest.mark.parametrize("tool", ["command_execution", "mcp_tool_call", "web_search", "file_change"])
def test_codex_never_ignores_external_tool_use_even_if_the_turn_completes(cli, tool):
    cli["events"].insert(0, {"type": "item.completed", "item": {"type": tool}})
    with pytest.raises(codex.CodexError, match="external tool"):
        codex.CodexClient().complete([], {})


@pytest.fixture
def teacher_batch(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(run_traces, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_traces, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.setattr(run_traces, "load_config", lambda: {"teacher": {
        "provider": "codex", "model": "gpt-6.1-terra", "concurrency": 1,
    }})
    scenarios = [generate_scenario(f"test_{i}", random.Random(i), force_root_cause="transient_noise")
                 for i in range(3)]
    (data_dir / "train.jsonl").write_text("".join(json.dumps(scenario_to_dict(s)) + "\n" for s in scenarios))
    client = SimpleNamespace(version="codex-test", validate_setup=lambda: None)
    monkeypatch.setattr(run_traces, "CodexClient", lambda **kwargs: client)
    original = run_traces.run_single_trace
    state = {"calls": [], "fail_id": None}

    def trace(scen, sop, client, use_oracle):
        assert use_oracle is False
        state["calls"].append(scen.id)
        if scen.id == state["fail_id"]:
            raise codex.CodexError("subscription usage limit")
        return original(scen, sop, client, True)

    monkeypatch.setattr(run_traces, "run_single_trace", trace)
    return state


def test_codex_resume_skips_accepted_traces_and_preserves_larger_runs(teacher_batch):
    first = run_traces.run_teacher_traces(limit=1, resume=True)
    assert first["pass_count"] == 1
    rendered = (run_traces.TRACES_DIR / "sop_rendered.md").read_bytes()
    assert first["sop_sha256"] == hashlib.sha256(rendered).hexdigest()
    assert b"{{ALIAS_CATALOG}}" not in rendered
    second = run_traces.run_teacher_traces(resume=True)
    assert second["resumed_count"] == 1
    assert second["pass_count"] == 3
    assert teacher_batch["calls"] == ["test_0", "test_1", "test_2"]
    third = run_traces.run_teacher_traces(limit=1, resume=True)
    assert third["resumed_count"] == 1
    lines = (run_traces.TRACES_DIR / "train_pass.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["teacher"]["model"] == "gpt-6.1-terra" for line in lines)


def test_codex_failure_checkpoints_and_stops_scheduling(teacher_batch):
    teacher_batch["fail_id"] = "test_1"
    with pytest.raises(codex.CodexError, match="rerun to resume"):
        run_traces.run_teacher_traces(resume=True)
    assert teacher_batch["calls"] == ["test_0", "test_1"]
    assert len((run_traces.TRACES_DIR / "train_pass.jsonl").read_text().splitlines()) == 1
    teacher_batch["fail_id"] = None
    summary = run_traces.run_teacher_traces(resume=True)
    assert summary["resumed_count"] == 1
    assert summary["pass_count"] == 3


def test_resume_refuses_to_mix_teachers(teacher_batch):
    run_traces.run_teacher_traces(limit=1, resume=True)
    path = run_traces.TRACES_DIR / "train_pass.jsonl"
    before = path.read_text()
    with pytest.raises(ValueError, match="different teacher"):
        run_traces.run_teacher_traces(use_oracle=True, resume=True)
    assert path.read_text() == before


def test_resume_refuses_changed_scenarios(teacher_batch):
    run_traces.run_teacher_traces(limit=1, resume=True)
    data = run_traces.DATA_DIR / "train.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    rows[0]["rendered_alert"] = "changed alert"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="different teacher, SOP, or dataset"):
        run_traces.run_teacher_traces(resume=True)


def test_resume_fingerprint_includes_rendered_alias_catalog(teacher_batch, monkeypatch):
    from exp.world.aliases import SERVICE_ALIASES

    run_traces.run_teacher_traces(limit=1, resume=True)
    monkeypatch.setitem(SERVICE_ALIASES, "cart-svc", SERVICE_ALIASES["cart-svc"] + ["new-basket-alias"])
    with pytest.raises(ValueError, match="different teacher, SOP, or dataset"):
        run_traces.run_teacher_traces(resume=True)


def test_resume_revalidates_saved_traces_before_reusing_them(teacher_batch):
    run_traces.run_teacher_traces(limit=1, resume=True)
    path = run_traces.TRACES_DIR / "train_pass.jsonl"
    entry = json.loads(path.read_text())
    assert entry["trace"][1]["step"]["step"] == "get_metrics"
    del entry["trace"][1]
    path.write_text(json.dumps(entry) + "\n")
    before = path.read_text()
    with pytest.raises(ValueError, match="fails current validation"):
        run_traces.run_teacher_traces(limit=1, resume=True)
    assert path.read_text() == before


def test_parallel_failure_drains_and_checkpoints_other_workers(teacher_batch, monkeypatch):
    data = run_traces.DATA_DIR / "train.jsonl"
    with data.open("a") as f:
        for i in range(3, 5):
            scenario = generate_scenario(f"test_{i}", random.Random(i), force_root_cause="transient_noise")
            f.write(json.dumps(scenario_to_dict(scenario)) + "\n")

    barrier = threading.Barrier(3, timeout=10)
    release = threading.Event()
    original = run_traces.run_single_trace

    class ReleaseWorkersError(codex.CodexError):
        def __str__(self):
            release.set()
            return super().__str__()

    def parallel_trace(scenario, *args):
        barrier.wait()
        if scenario.id == "test_0":
            raise ReleaseWorkersError("simulated quota failure")
        assert release.wait(10)
        return original(scenario, *args)

    monkeypatch.setattr(run_traces, "run_single_trace", parallel_trace)
    with pytest.raises(codex.CodexError, match="rerun to resume"):
        run_traces.run_teacher_traces(max_workers=3, resume=True)

    accepted = [json.loads(line) for line in (run_traces.TRACES_DIR / "train_pass.jsonl").read_text().splitlines()]
    assert {entry["id"] for entry in accepted} == {"test_1", "test_2"}
    assert set(teacher_batch["calls"]) == {"test_1", "test_2"}
    summary = json.loads((run_traces.TRACES_DIR / "summary.json").read_text())
    assert summary["total"] == 5
    assert summary["pass_count"] == 2
    assert summary["fail_count"] == 1
    assert summary["incomplete"] is True
    assert summary["pass_rate"] is None
    assert summary["checkpoint_coverage"] == 0.4
    assert summary["attempted_count"] == 3
    assert summary["new_attempt_count"] == 3
    assert summary["not_attempted_count"] == 2
    assert summary["request_error_count"] == 1
    assert summary["validation_fail_count"] == 0


@pytest.mark.parametrize("workers", [2, 10])
def test_parallel_workers_checkpoint_every_scenario_once(teacher_batch, monkeypatch, workers):
    scenarios = [generate_scenario(f"test_{i}", random.Random(i), force_root_cause="transient_noise")
                 for i in range(20)]
    (run_traces.DATA_DIR / "train.jsonl").write_text(
        "".join(json.dumps(scenario_to_dict(scenario)) + "\n" for scenario in scenarios)
    )
    barrier = threading.Barrier(workers, timeout=10)
    original = run_traces.run_single_trace

    def parallel_trace(*args):
        barrier.wait()
        return original(*args)

    monkeypatch.setattr(run_traces, "run_single_trace", parallel_trace)
    summary = run_traces.run_teacher_traces(max_workers=workers, resume=True)
    assert summary["pass_count"] == 20
    accepted = [json.loads(line) for line in (run_traces.TRACES_DIR / "train_pass.jsonl").read_text().splitlines()]
    assert len(accepted) == len({entry["id"] for entry in accepted}) == 20
    assert sorted(teacher_batch["calls"]) == sorted(scenario.id for scenario in scenarios)
