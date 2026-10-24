import pytest
from pydantic import ValidationError
from exp.schema import STEP_ADAPTER, Step


def test_valid_step_types():
    valid_examples = [
        {"step": "extract_alert", "service": "checkout-api", "symptom": "high_error_rate", "region": "us-east", "started_at": "2026-10-22T12:00:00Z", "started_minutes_ago": None},
        {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30},
        {"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120},
        {"step": "query_logs", "service": "checkout-api", "window_minutes": 30},
        {"step": "get_dependencies", "service": "checkout-api"},
        {"step": "check_dependency_health", "dependency": "payments-svc"},
        {"step": "conclude", "root_cause": "bad_deploy", "runbook_id": "RB-101", "culprit": "d-1234", "log_signature": "unhandled_exception"},
        {"step": "draft_summary", "text": "Everything is running smoothly again."},
    ]
    for ex in valid_examples:
        parsed = STEP_ADAPTER.validate_python(ex)
        assert parsed.step == ex["step"]


def test_invalid_step_value():
    with pytest.raises(ValidationError):
        STEP_ADAPTER.validate_python({"step": "unknown_step_action", "service": "checkout-api"})


def test_wrong_window_minutes():
    # get_metrics requires window_minutes: 30
    with pytest.raises(ValidationError):
        STEP_ADAPTER.validate_python({
            "step": "get_metrics",
            "service": "checkout-api",
            "metric": "error_rate",
            "window_minutes": 60  # invalid window
        })

    # get_recent_deploys requires window_minutes: 120
    with pytest.raises(ValidationError):
        STEP_ADAPTER.validate_python({
            "step": "get_recent_deploys",
            "service": "checkout-api",
            "window_minutes": 30  # invalid window
        })


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        STEP_ADAPTER.validate_python({
            "step": "get_dependencies",
            "service": "checkout-api",
            "extra_field": "disallowed"
        })


def test_summary_length_limit():
    long_text = "x" * 701
    with pytest.raises(ValidationError):
        STEP_ADAPTER.validate_python({
            "step": "draft_summary",
            "text": long_text
        })
