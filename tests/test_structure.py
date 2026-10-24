import pytest
from exp.runtime.structure import structural_problem
from exp.schema import (
    CheckDependencyHealth,
    Conclude,
    DraftSummary,
    ExtractAlert,
    GetDependencies,
    GetMetrics,
)


def test_first_step_must_be_extract_alert():
    metric_step = GetMetrics(step="get_metrics", service="checkout-api", metric="error_rate", window_minutes=30)
    prob = structural_problem("START", metric_step, None, [])
    assert prob == "first_step_must_be_extract_alert"

    extract_step = ExtractAlert(step="extract_alert", service="checkout-api", symptom="high_error_rate", region="us-east")
    prob_ok = structural_problem("START", extract_step, None, [])
    assert prob_ok is None


def test_extract_alert_can_appear_only_once():
    ext = ExtractAlert(step="extract_alert", service="checkout-api", symptom="high_error_rate", region="us-east")
    prob = structural_problem("extract_alert", ext, ext, [{"step": ext.model_dump()}])
    assert prob == "extract_alert_can_appear_only_once"


def test_unresolvable_alert_requires_conclude():
    # service is None
    ext = ExtractAlert(step="extract_alert", service=None, symptom="high_error_rate", region="us-east")
    metric_step = GetMetrics(step="get_metrics", service="checkout-api", metric="error_rate", window_minutes=30)
    prob = structural_problem("extract_alert", metric_step, ext, [{"step": ext.model_dump()}])
    assert prob == "unresolvable_alert_requires_conclude"

    conclude_step = Conclude(step="conclude", root_cause="unknown", runbook_id="RB-900")
    prob_ok = structural_problem("extract_alert", conclude_step, ext, [{"step": ext.model_dump()}])
    assert prob_ok is None


def test_tool_service_mismatch():
    ext = ExtractAlert(step="extract_alert", service="checkout-api", symptom="high_error_rate", region="us-east")
    # Tool queries payments-svc instead of extracted checkout-api
    metric_step = GetMetrics(step="get_metrics", service="payments-svc", metric="error_rate", window_minutes=30)
    prob = structural_problem("extract_alert", metric_step, ext, [{"step": ext.model_dump()}])
    assert "tool_service_mismatch" in prob


def test_dependency_health_requires_prior_get_dependencies():
    ext = ExtractAlert(step="extract_alert", service="checkout-api", symptom="high_error_rate", region="us-east")
    health_step = CheckDependencyHealth(step="check_dependency_health", dependency="payments-svc")

    # Without prior get_dependencies
    prob = structural_problem("extract_alert", health_step, ext, [{"step": ext.model_dump()}])
    assert "not_in_prior_get_dependencies" in prob

    # With prior get_dependencies returning ["payments-svc", "auth-gateway"]
    trace = [
        {"step": ext.model_dump()},
        {"step": {"step": "get_dependencies", "service": "checkout-api"}, "result": {"dependencies": ["payments-svc", "auth-gateway"]}}
    ]
    prob_ok = structural_problem("get_dependencies", health_step, ext, trace)
    assert prob_ok is None


def test_repetition_and_retry_constraints():
    ext = ExtractAlert(step="extract_alert", service="checkout-api", symptom="high_error_rate", region="us-east")
    metric_step = GetMetrics(step="get_metrics", service="checkout-api", metric="error_rate", window_minutes=30)

    # 1. Repeated successful call is disallowed
    trace_success = [
        {"step": ext.model_dump()},
        {"step": metric_step.model_dump(), "result": {"status": "critical", "change_pct": 150}}
    ]
    prob_rep = structural_problem("get_metrics", metric_step, ext, trace_success)
    assert prob_rep == "repeated_successful_call"

    # 2. Call after single error may be repeated once
    trace_err1 = [
        {"step": ext.model_dump()},
        {"step": metric_step.model_dump(), "result": {"tool": "get_metrics", "error": "timeout"}}
    ]
    prob_retry1 = structural_problem("get_metrics", metric_step, ext, trace_err1)
    assert prob_retry1 is None  # Allowed 1 retry!

    # 3. Call after second error cannot be repeated again
    trace_err2 = [
        {"step": ext.model_dump()},
        {"step": metric_step.model_dump(), "result": {"tool": "get_metrics", "error": "timeout"}},
        {"step": metric_step.model_dump(), "result": {"tool": "get_metrics", "error": "timeout"}}
    ]
    prob_retry2 = structural_problem("get_metrics", metric_step, ext, trace_err2)
    assert prob_retry2 == "exceeded_retry_limit_max_1_retry"


def test_conclude_and_summary_sequence():
    ext = ExtractAlert(step="extract_alert", service="checkout-api", symptom="high_error_rate", region="us-east")
    conclude_step = Conclude(step="conclude", root_cause="unknown", runbook_id="RB-900")
    summary_step = DraftSummary(step="draft_summary", text="Summary text")

    # summary must immediately follow conclude
    prob1 = structural_problem("extract_alert", summary_step, ext, [{"step": ext.model_dump()}])
    assert prob1 == "draft_summary_must_immediately_follow_conclude"

    prob_conclude = structural_problem("conclude", summary_step, ext, [
        {"step": ext.model_dump()},
        {"step": conclude_step.model_dump()}
    ])
    assert prob_conclude is None

    # Nothing may follow draft_summary
    prob_after = structural_problem("draft_summary", conclude_step, ext, [
        {"step": ext.model_dump()},
        {"step": conclude_step.model_dump()},
        {"step": summary_step.model_dump()}
    ])
    assert prob_after == "nothing_may_follow_draft_summary"
