import pytest
from exp.validators import (
    validate_v1_schema,
    validate_v2_trajectory,
    validate_v3_extraction,
    validate_v4_conclusion,
    validate_v5_decision,
    validate_v6_summary_facts,
    validate_v7_summary_safe,
    validate_v8_summary_length,
)


# --- V1 Schema ---
def test_v1_schema_pass_and_fails():
    trace_pass = [
        {"step": {"step": "extract_alert", "service": "checkout-api", "symptom": "high_error_rate", "region": "us-east"}},
        {"step": {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}},
    ]
    ok, _ = validate_v1_schema(trace_pass)
    assert ok is True

    # Fail 1: Empty trace
    ok1, msg1 = validate_v1_schema([])
    assert ok1 is False

    # Fail 2: Invalid step schema
    trace_fail = [
        {"step": {"step": "invalid_step_name"}}
    ]
    ok2, msg2 = validate_v1_schema(trace_fail)
    assert ok2 is False


# --- V2 Trajectory ---
def test_v2_trajectory_pass_and_fails():
    ext_dict = {"service": "checkout-api", "symptom": "high_error_rate", "region": "us-east"}
    trace_pass = [
        {"step": {"step": "extract_alert", **ext_dict}},
        {"step": {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}, "result": {"status": "critical"}},
        {"step": {"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120}, "result": {"correlated": False, "correlated_deploy_id": None, "correlated_kind": None}},
    ]
    ok, _ = validate_v2_trajectory(trace_pass, ext_dict)
    assert ok is True

    # Fail 1: Wrong first tool call
    trace_fail1 = [
        {"step": {"step": "extract_alert", **ext_dict}},
        {"step": {"step": "query_logs", "service": "checkout-api", "window_minutes": 30}},
    ]
    ok1, _ = validate_v2_trajectory(trace_fail1, ext_dict)
    assert ok1 is False

    # Fail 2: Wrong tool arguments
    trace_fail2 = [
        {"step": {"step": "extract_alert", **ext_dict}},
        {"step": {"step": "get_metrics", "service": "checkout-api", "metric": "latency_p99", "window_minutes": 30}},
    ]
    ok2, _ = validate_v2_trajectory(trace_fail2, ext_dict)
    assert ok2 is False


# --- V3 Extraction ---
def test_v3_extraction_pass_and_fails():
    gt = {"service": "checkout-api", "symptom": "high_error_rate", "region": "us-east", "started_at": "2026-10-22T12:00:00Z", "started_minutes_ago": None}
    trace_pass = [{"step": {"step": "extract_alert", **gt}}]
    ok, _ = validate_v3_extraction(trace_pass, gt)
    assert ok is True

    # Fail 1: Wrong service
    trace_f1 = [{"step": {"step": "extract_alert", **gt, "service": "payments-svc"}}]
    ok1, _ = validate_v3_extraction(trace_f1, gt)
    assert ok1 is False

    # Fail 2: Wrong symptom
    trace_f2 = [{"step": {"step": "extract_alert", **gt, "symptom": "high_latency"}}]
    ok2, _ = validate_v3_extraction(trace_f2, gt)
    assert ok2 is False


# --- V4 Conclusion ---
def test_v4_conclusion_pass_and_fails():
    exp_conc = {"root_cause": "bad_deploy", "runbook_id": "RB-101", "culprit": "d-123", "log_signature": "unhandled_exception"}
    trace_pass = [{"step": {"step": "conclude", **exp_conc}}]
    ok, details = validate_v4_conclusion(trace_pass, exp_conc)
    assert ok is True

    # Fail 1: Missing culprit
    trace_f1 = [{"step": {"step": "conclude", **exp_conc, "culprit": None}}]
    ok1, det1 = validate_v4_conclusion(trace_f1, exp_conc)
    assert ok1 is False
    assert det1["culprit"] is False

    # Fail 2: Wrong root cause
    trace_f2 = [{"step": {"step": "conclude", **exp_conc, "root_cause": "unknown"}}]
    ok2, det2 = validate_v4_conclusion(trace_f2, exp_conc)
    assert ok2 is False
    assert det2["root_cause"] is False


# --- V5 Decision ---
def test_v5_decision_pass_and_fails():
    ext = {"service": "checkout-api", "symptom": "high_error_rate"}
    dec = {"severity": "sev1", "action": "page_oncall_immediately", "team": "team-checkout", "runbook_title": "Roll back the deploy"}
    trace_pass = [
        {"step": {"step": "conclude", "root_cause": "bad_deploy", "culprit": "d-1"}, "decision": dec}
    ]
    ok, _ = validate_v5_decision(trace_pass, ext, "critical")
    assert ok is True

    # Fail 1: Wrong severity
    dec_bad_sev = dict(dec, severity="sev4")
    trace_f1 = [{"step": {"step": "conclude", "root_cause": "bad_deploy", "culprit": "d-1"}, "decision": dec_bad_sev}]
    ok1, _ = validate_v5_decision(trace_f1, ext, "critical")
    assert ok1 is False

    # Fail 2: Missing decision
    trace_f2 = [{"step": {"step": "conclude", "root_cause": "bad_deploy", "culprit": "d-1"}}]
    ok2, _ = validate_v5_decision(trace_f2, ext, "critical")
    assert ok2 is False


# --- V6 Summary Facts ---
def test_v6_summary_facts_pass_and_fails():
    trace_pass = [
        {"step": {"step": "conclude", "root_cause": "bad_deploy", "runbook_id": "RB-101", "culprit": "d-7731"},
         "decision": {"severity": "sev1", "team": "team-checkout"}},
        {"step": {"step": "draft_summary", "text": "SEV1: bad deploy d-7731 in checkout-api. Team team-checkout follow RB-101."}}
    ]
    ok, _ = validate_v6_summary_facts(trace_pass)
    assert ok is True

    # Fail 1: Missing culprit d-7731
    trace_f1 = [
        {"step": {"step": "conclude", "root_cause": "bad_deploy", "runbook_id": "RB-101", "culprit": "d-7731"},
         "decision": {"severity": "sev1", "team": "team-checkout"}},
        {"step": {"step": "draft_summary", "text": "SEV1: bad deploy in checkout-api. Team team-checkout follow RB-101."}}
    ]
    ok1, _ = validate_v6_summary_facts(trace_f1)
    assert ok1 is False

    # Fail 2: Missing runbook ID
    trace_f2 = [
        {"step": {"step": "conclude", "root_cause": "bad_deploy", "runbook_id": "RB-101", "culprit": "d-7731"},
         "decision": {"severity": "sev1", "team": "team-checkout"}},
        {"step": {"step": "draft_summary", "text": "SEV1: bad deploy d-7731. Team team-checkout on call."}}
    ]
    ok2, _ = validate_v6_summary_facts(trace_f2)
    assert ok2 is False


# --- V7 Summary Safe ---
def test_v7_summary_safe_pass_and_fails():
    trace_pass = [
        {"step": {"step": "extract_alert", "service": "checkout-api", "started_minutes_ago": 12}},
        {"step": {"step": "conclude", "root_cause": "bad_deploy", "runbook_id": "RB-101", "culprit": "d-7731"},
         "decision": {"severity": "sev1", "team": "team-checkout"}},
        {"step": {"step": "draft_summary", "text": "SEV1: checkout-api deploy d-7731 started 12 min ago. Team team-checkout RB-101."}}
    ]
    ok, _ = validate_v7_summary_safe(trace_pass)
    assert ok is True

    # Fail 1: Hallucinated service not in trace
    trace_f1 = [
        {"step": {"step": "extract_alert", "service": "checkout-api"}},
        {"step": {"step": "conclude", "root_cause": "bad_deploy", "runbook_id": "RB-101"},
         "decision": {"severity": "sev1", "team": "team-checkout"}},
        {"step": {"step": "draft_summary", "text": "SEV1: payments-svc failed."}}  # payments-svc not in trace!
    ]
    ok1, msg1 = validate_v7_summary_safe(trace_f1)
    assert ok1 is False

    # Fail 2: Hallucinated number not in trace
    trace_f2 = [
        {"step": {"step": "extract_alert", "service": "checkout-api"}},
        {"step": {"step": "conclude", "root_cause": "bad_deploy", "runbook_id": "RB-101"},
         "decision": {"severity": "sev1", "team": "team-checkout"}},
        {"step": {"step": "draft_summary", "text": "SEV1: checkout-api has 99999 errors."}}  # 99999 not in trace!
    ]
    ok2, msg2 = validate_v7_summary_safe(trace_f2)
    assert ok2 is False


# --- V8 Summary Length ---
def test_v8_summary_length_pass_and_fails():
    trace_pass = [{"step": {"step": "draft_summary", "text": "Valid short summary."}}]
    ok, _ = validate_v8_summary_length(trace_pass)
    assert ok is True

    # Fail 1: Empty text
    trace_f1 = [{"step": {"step": "draft_summary", "text": "   "}}]
    ok1, _ = validate_v8_summary_length(trace_f1)
    assert ok1 is False

    # Fail 2: Longer than 700 chars
    trace_f2 = [{"step": {"step": "draft_summary", "text": "x" * 701}}]
    ok2, _ = validate_v8_summary_length(trace_f2)
    assert ok2 is False
