import json
import random
from types import SimpleNamespace

import pytest

from exp.data.split import scenario_to_dict
from exp.runtime.clients import FakeClient
from exp.teacher import run_traces
from exp.train import build_sft
from exp.world.scenarios import generate_scenario


def test_missing_teacher_key_fails_before_writing_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(run_traces, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.setattr(run_traces, "GeminiClient", lambda **kwargs: SimpleNamespace(api_key=None))

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        run_traces.run_teacher_traces(provider="gemini")

    assert not (tmp_path / "traces").exists()


@pytest.mark.parametrize("trace_contents", [None, "", "\n  \n"])
def test_sft_rejects_missing_or_empty_traces_without_overwriting(tmp_path, monkeypatch, trace_contents):
    monkeypatch.setattr(build_sft, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.setattr(build_sft, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(build_sft, "SFT_DIR", tmp_path / "sft")
    build_sft.SFT_DIR.mkdir()
    output = build_sft.SFT_DIR / "train.jsonl"
    output.write_text("existing training data\n")
    if trace_contents is not None:
        build_sft.TRACES_DIR.mkdir()
        (build_sft.TRACES_DIR / "train_pass.jsonl").write_text(trace_contents)

    with pytest.raises(ValueError, match="traces"):
        build_sft.build_sft_dataset()

    assert output.read_text() == "existing training data\n"


@pytest.mark.parametrize("contents", [None, "", "  \n"])
def test_training_data_validation_rejects_missing_or_empty(tmp_path, contents):
    path = tmp_path / "train.jsonl"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(ValueError, match="missing or empty"):
        build_sft.validate_sft_file(path)


def test_oracle_traces_and_sft_work_without_gemini(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    traces_dir = tmp_path / "traces"
    data_dir.mkdir()
    monkeypatch.setattr(run_traces, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_traces, "TRACES_DIR", traces_dir)
    monkeypatch.setattr(build_sft, "DATA_DIR", data_dir)
    monkeypatch.setattr(build_sft, "TRACES_DIR", traces_dir)
    monkeypatch.setattr(build_sft, "SFT_DIR", tmp_path / "sft")

    def unexpected_client(**kwargs):
        pytest.fail("Oracle traces must not initialize a Gemini client")

    monkeypatch.setattr(run_traces, "GeminiClient", unexpected_client)
    scenario = generate_scenario("test_oracle", random.Random(1234), force_root_cause="transient_noise")
    (data_dir / "train.jsonl").write_text(json.dumps(scenario_to_dict(scenario)) + "\n")

    summary = run_traces.run_teacher_traces(use_oracle=True, max_workers=1)
    assert summary["pass_count"] == 1
    rows = build_sft.build_sft_dataset()
    assert rows
    build_sft.validate_sft_file(build_sft.SFT_DIR / "train.jsonl")


def test_empty_dataset_stops_training_before_model_download(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from exp.train import train_lora

    monkeypatch.setattr(train_lora, "SFT_DIR", tmp_path)
    monkeypatch.setattr(train_lora, "ADAPTERS_DIR", tmp_path / "adapters")
    (tmp_path / "train.jsonl").write_text("")

    def unexpected_download(*args, **kwargs):
        pytest.fail("Empty datasets must be rejected before downloading a model")

    monkeypatch.setattr(train_lora.AutoTokenizer, "from_pretrained", unexpected_download)
    with pytest.raises(ValueError, match="missing or empty"):
        train_lora.train_qlora()
    assert not train_lora.ADAPTERS_DIR.exists()


@pytest.mark.parametrize("field", ["service", "region"])
def test_teacher_does_not_silently_repair_missing_extraction_fields(field):
    from exp.prompts import load_sop_prompt

    scenario = generate_scenario("missing_field", random.Random(1234), force_root_cause="transient_noise")
    extraction = {"step": "extract_alert", **scenario.expected_extraction}
    assert extraction[field] is not None
    extraction[field] = None
    client = FakeClient(responses=[
        json.dumps(extraction),
        json.dumps({"step": "conclude", "root_cause": "unknown", "runbook_id": "RB-900"}),
        json.dumps({"step": "draft_summary", "text": "Unknown cause. Follow RB-900."}),
    ])
    ok, report, trace, _ = run_traces.run_single_trace(scenario, load_sop_prompt(), client)
    assert ok is False
    assert trace[0]["step"][field] is None
    assert report["V3"]["ok"] is False
    assert f"extraction_mismatch_{field}" in report["V3"]["msg"]
