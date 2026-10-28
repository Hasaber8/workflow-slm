from pathlib import Path
import pytest
from exp.data import log_bank, split
from exp.data.log_bank import build_log_bank, validate_log_line
from exp.data.render import render_alert_templated
from exp.data.split import generate_all_splits
from exp.world.services import HELDOUT_SERVICES, TRAIN_SERVICES


@pytest.fixture(autouse=True)
def isolate_generated_artifacts(tmp_path, monkeypatch):
    # Generator tests must never replace real scenario splits or log banks.
    monkeypatch.setattr(log_bank, "DATA_DIR", tmp_path)
    monkeypatch.setattr(log_bank, "LOG_BANK_FILE", tmp_path / "log_bank.json")
    monkeypatch.setattr(split, "DATA_DIR", tmp_path)


def test_log_bank_validity():
    bank = build_log_bank(force=True)
    signatures = [
        "unhandled_exception", "upstream_timeout", "connection_refused",
        "oom_killed", "cpu_throttled", "db_pool_exhausted",
        "auth_failure", "config_parse_error", "benign"
    ]
    for sig in signatures:
        assert sig in bank
        lines = bank[sig]
        assert len(lines) >= (60 if sig == "benign" else 40)
        assert len(set(lines)) == len(lines), f"Duplicates in {sig}"
        for l in lines:
            assert validate_log_line(l), f"Forbidden word in {l}"


def test_render_alert_styles_and_aliases():
    styles = ["pagerduty_style", "slack_message", "email_notification", "oncall_free_text", "terse_sms", "handoff_note"]
    for st in styles:
        rendered = render_alert_templated(
            service="checkout-api",
            symptom="high_error_rate",
            region="us-east",
            started_at=None,
            started_minutes_ago=15,
            received_at="2026-10-22T12:15:00Z",
            style=st
        )
        assert "Received: 2026-10-22T12:15:00Z" in rendered
        assert len(rendered.splitlines()) >= 2


def test_splits_generation_and_leakage():
    splits = generate_all_splits(profile="smoke", force=True)
    all_ids = set()
    all_texts = set()

    assert "train" in splits
    assert "dev" in splits
    assert "test" in splits
    assert "shift" in splits
    assert "probe" in splits

    for s_name, scen_list in splits.items():
        for s in scen_list:
            # No duplicate IDs
            assert s.id not in all_ids
            all_ids.add(s.id)

            # No duplicate texts
            assert s.rendered_alert not in all_texts
            all_texts.add(s.rendered_alert)

            # Leakage checks
            if s_name in ("train", "dev"):
                assert s.service not in HELDOUT_SERVICES
                for d in s.dependencies_state.get(s.service or "", []):
                    assert d not in HELDOUT_SERVICES
                assert "Handoff note" not in (s.rendered_alert or "")
