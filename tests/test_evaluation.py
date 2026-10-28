import json
import random

import pytest

from exp.baselines import run_baseline
from exp.data.split import scenario_to_dict
from exp.eval import run_eval
from exp.eval.teacher_forced import are_steps_equal, evaluate_teacher_forced
from exp.runtime.clients import FakeClient
from exp.teacher.run_traces import create_oracle_teacher_responder
from exp.world.scenarios import generate_scenario


def make_scenario(index=0, cause="credential_failure", **kwargs):
    return generate_scenario(
        f"eval_{index}", random.Random(index), force_root_cause=cause,
        transient_error_fraction=0, persistent_error_fraction=0, **kwargs,
    )


def oracle_client(scenario):
    return FakeClient(response_fn=create_oracle_teacher_responder(scenario))


def test_teacher_forced_scores_every_phase_and_matches_runtime_messages():
    scenario = make_scenario()
    responder = create_oracle_teacher_responder(scenario)

    def complete(messages):
        assert messages[-1]["role"] == "user"
        for message in messages:
            if message["role"] == "user" and message["content"].startswith("TOOL_RESULT: {"):
                assert "_sig" not in json.loads(message["content"][13:])
        if len(messages) == 4:
            assert messages[-1]["content"] == "TOOL_RESULT: continue"
        return responder(messages)

    summary = evaluate_teacher_forced(FakeClient(response_fn=complete), [scenario], "test SOP")
    assert summary["overall_accuracy"] == 1.0
    for phase in ("extract_alert", "tool_calls", "conclude", "draft_summary"):
        assert summary["phases"][phase]["total"] > 0
        assert summary["phases"][phase]["accuracy"] == 1.0


@pytest.mark.parametrize("wrong_phase", ["extract_alert", "conclude", "draft_summary"])
def test_teacher_forced_counts_model_mistakes_independently(wrong_phase):
    scenario = make_scenario()
    responder = create_oracle_teacher_responder(scenario)

    def complete(messages):
        response = json.loads(responder(messages))
        if response["step"] == wrong_phase:
            if wrong_phase == "extract_alert":
                response["service"] = None
            elif wrong_phase == "conclude":
                response = {"step": "conclude", "root_cause": "unknown", "runbook_id": "RB-900"}
            else:
                response["text"] = "Everything is fine."
        return json.dumps(response)

    summary = evaluate_teacher_forced(FakeClient(response_fn=complete), [scenario], "test SOP")
    assert summary["overall_accuracy"] < 1.0
    assert summary["phases"][wrong_phase]["accuracy"] == 0.0
    for phase in ("extract_alert", "tool_calls", "conclude", "draft_summary"):
        if phase != wrong_phase:
            assert summary["phases"][phase]["accuracy"] == 1.0


def test_teacher_forced_handles_an_unresolvable_alert():
    scenario = make_scenario(cause="unknown", force_subvariant="unresolvable")
    summary = evaluate_teacher_forced(oracle_client(scenario), [scenario], "test SOP")
    assert summary["overall_accuracy"] == 1.0
    assert summary["phases"]["tool_calls"]["total"] == 0
    assert summary["phases"]["conclude"]["total"] == 1
    assert summary["phases"]["draft_summary"]["total"] == 1


def test_teacher_forced_compares_tool_windows():
    expected = {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}
    assert are_steps_equal(expected, expected)
    assert not are_steps_equal({**expected, "window_minutes": 120}, expected)


def test_oracle_evaluation_uses_a_separate_responder_for_every_scenario(tmp_path, monkeypatch):
    scenarios = [make_scenario(0), make_scenario(1, "transient_noise")]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dev.jsonl").write_text("".join(json.dumps(scenario_to_dict(s)) + "\n" for s in scenarios))
    monkeypatch.setattr(run_eval, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(run_baseline, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_baseline, "BASELINES_DIR", tmp_path / "baselines")
    summary = run_eval.run_full_evaluation(model_name="oracle", split_name="dev", limit=2)
    assert summary["free_run"]["n_scenarios"] == 2
    assert summary["free_run"]["e2e_ok_rate"] == 1.0
    assert summary["teacher_forced"]["overall_accuracy"] == 1.0
    assert summary["teacher_forced"]["phases"]["conclude"]["total"] == 2


def test_evaluation_requires_a_nonempty_split(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path / "eval")
    with pytest.raises(ValueError, match="No scenarios found"):
        run_eval.run_full_evaluation()


def test_compiled_model_uses_the_same_prompt_mode_in_both_evaluations(tmp_path, monkeypatch):
    (tmp_path / "dev.jsonl").write_text(json.dumps(scenario_to_dict(make_scenario())) + "\n")
    monkeypatch.setattr(run_eval, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(run_eval, "VllmClient", lambda **kwargs: object())
    prompts = []

    def evaluate(**kwargs):
        prompts.append(kwargs["system_prompt"])
        return {}

    monkeypatch.setattr(run_eval, "run_baseline_on_split", evaluate)
    monkeypatch.setattr(run_eval, "evaluate_teacher_forced", evaluate)
    run_eval.run_full_evaluation(model_name="Model_D", split_name="dev", limit=1)
    assert prompts == ["incident_triage_v1", "incident_triage_v1"]
