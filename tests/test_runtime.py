import json
import pytest
from exp.runtime.clients import FakeClient
from exp.runtime.escalate import maybe_escalate
from exp.runtime.loop import run
from exp.validators import validate_trace
from exp.world.simulator import Scenario, Simulator


def test_golden_trace_from_doc():
    raw_input = (
        "Received: 2026-10-22T15:30:00Z\n"
        "[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago. lots of timeouts calling downstream. -- pagerduty bot"
    )

    steps = [
        json.dumps({"step": "extract_alert", "service": "checkout-api", "symptom": "high_error_rate", "region": "us-east", "started_at": None, "started_minutes_ago": 12}),
        json.dumps({"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}),
        json.dumps({"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120}),
        json.dumps({"step": "query_logs", "service": "checkout-api", "window_minutes": 30}),
        json.dumps({"step": "get_dependencies", "service": "checkout-api"}),
        json.dumps({"step": "check_dependency_health", "dependency": "auth-gateway"}),
        json.dumps({"step": "check_dependency_health", "dependency": "auth-gateway"}),
        json.dumps({"step": "check_dependency_health", "dependency": "payments-svc"}),
        json.dumps({"step": "conclude", "root_cause": "upstream_dependency_failure", "runbook_id": "RB-201", "culprit": "payments-svc", "log_signature": "upstream_timeout"}),
        json.dumps({"step": "draft_summary", "text": "SEV1: checkout-api error rate is critical in us-east (started about 12 min ago). Cause: upstream dependency failure, payments-svc is down (INC-4471). Follow RB-201 (escalate to the dependency owner). Paging team-payments immediately."}),
    ]

    scen = Scenario(
        id="golden_scenario",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at=None,
        started_minutes_ago=12,
        received_at="2026-10-22T15:30:00Z",
        root_cause="upstream_dependency_failure",
        culprit="payments-svc",
        log_signature="upstream_timeout",
        metrics_state={"checkout-api": {"error_rate": {"current": 0.18, "baseline": 0.01, "change_pct": 1700, "status": "critical"}}},
        deploys_state={"checkout-api": [{"id": "d-7731", "kind": "code", "version": "v3.8.2", "minutes_before_start": 95}]},
        logs_state={"checkout-api": [
            "15:21:04 WARN payments client: request to payments-svc exceeded 3000ms deadline (attempt 2/3)",
            "15:21:09 ERROR checkout.handler: upstream call failed, returning 502 to client",
            "15:21:11 WARN payments client: request to payments-svc exceeded 3000ms deadline (attempt 3/3)",
            "15:21:15 INFO healthcheck ok",
            "15:21:20 ERROR checkout.handler: upstream call failed, returning 502 to client"
        ]},
        dependencies_state={"checkout-api": ["auth-gateway", "payments-svc", "inventory-svc"]},
        dependency_health_state={
            "auth-gateway": {"health": "healthy", "incident_id": None},
            "payments-svc": {"health": "down", "incident_id": "INC-4471"}
        },
        injected_errors={"check_dependency_health:auth-gateway": 1},
        error_type="timeout",
        expected_extraction={
            "service": "checkout-api", "symptom": "high_error_rate", "region": "us-east", "started_at": None, "started_minutes_ago": 12
        }
    )

    client = FakeClient(responses=steps)
    sim = Simulator(scen)
    result = run(raw_input, client, sim)

    assert result.status == "ok"
    assert len(result.trace) == 10

    # Validate against V1-V8
    truth_dict = {
        "expected_extraction": scen.expected_extraction,
        "root_cause": scen.root_cause,
        "runbook_id": "RB-201",
        "culprit": scen.culprit,
        "log_signature": scen.log_signature,
    }
    all_pass, report = validate_trace(result.trace, truth_dict, primary_status="critical")
    assert all_pass is True, f"Validation failed: {report}"


def test_runtime_unresolvable_path():
    steps = [
        json.dumps({"step": "extract_alert", "service": None, "symptom": "high_error_rate", "region": "us-east", "started_at": None, "started_minutes_ago": 10}),
        json.dumps({"step": "conclude", "root_cause": "unknown", "runbook_id": "RB-900", "culprit": None, "log_signature": None}),
        json.dumps({"step": "draft_summary", "text": "SEV3: unknown service, escalating to sre-oncall under RB-900."}),
    ]
    scen = Scenario(
        id="unres",
        service=None,
        symptom="high_error_rate",
        region="us-east",
        started_at=None,
        started_minutes_ago=10,
        received_at="2026-10-22T12:00:00Z",
        root_cause="unknown",
        culprit=None,
        log_signature="none",
    )
    client = FakeClient(responses=steps)
    sim = Simulator(scen)
    res = run("Received: ...\n[FIRING] unknown alert", client, sim)
    assert res.status == "ok"
    assert res.trace[1]["step"]["step"] == "conclude"


def test_runtime_invalid_json():
    client = FakeClient(responses=["Not valid json"])
    scen = Scenario(
        id="inv",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at=None,
        started_minutes_ago=10,
        received_at="2026-10-22T12:00:00Z",
        root_cause="transient_noise",
        culprit=None,
        log_signature="none",
    )
    sim = Simulator(scen)
    res = run("Received: ...", client, sim)
    assert res.status == "invalid_output"


def test_runtime_illegal_step():
    # Model attempts tool step before extract_alert
    client = FakeClient(responses=[
        json.dumps({"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30})
    ])
    scen = Scenario(
        id="ill",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at=None,
        started_minutes_ago=10,
        received_at="2026-10-22T12:00:00Z",
        root_cause="transient_noise",
        culprit=None,
        log_signature="none",
    )
    sim = Simulator(scen)
    res = run("Received: ...", client, sim)
    assert "illegal:first_step_must_be_extract_alert" in res.status


def test_runtime_max_steps():
    # Model sends extract_alert, then queries multiple distinct valid metrics without concluding, hitting max_steps
    distinct_metrics = ["error_rate", "latency_p99", "cpu", "memory"]
    steps = [
        json.dumps({"step": "extract_alert", "service": "checkout-api", "symptom": "high_error_rate", "region": "us-east", "started_at": None, "started_minutes_ago": 10})
    ] + [
        json.dumps({"step": "get_metrics", "service": "checkout-api", "metric": m, "window_minutes": 30})
        for m in distinct_metrics
    ]
    client = FakeClient(responses=steps)
    scen = Scenario(
        id="max_s",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at=None,
        started_minutes_ago=10,
        received_at="2026-10-22T12:00:00Z",
        root_cause="transient_noise",
        culprit=None,
        log_signature="none",
    )
    sim = Simulator(scen)
    res = run("Received: ...", client, sim, max_steps=5)
    assert res.status == "max_steps"


def test_runtime_escalation():
    # If run returns invalid_output, maybe_escalate runs teacher fallback
    failed_res = run("Received: ...", FakeClient(responses=["bad json"]), Simulator(Scenario(
        id="esc", service="checkout-api", symptom="high_error_rate", region="us-east", started_at=None, started_minutes_ago=10, received_at="2026-10-22T12:00:00Z", root_cause="transient_noise", culprit=None, log_signature="none"
    )))
    assert failed_res.status == "invalid_output"

    teacher_steps = [
        json.dumps({"step": "extract_alert", "service": "checkout-api", "symptom": "high_error_rate", "region": "us-east", "started_at": None, "started_minutes_ago": 10}),
        json.dumps({"step": "conclude", "root_cause": "unknown", "runbook_id": "RB-900", "culprit": None, "log_signature": None}),
        json.dumps({"step": "draft_summary", "text": "SEV3: checkout-api unknown error, team-checkout follow RB-900."}),
    ]
    teacher_client = FakeClient(responses=teacher_steps)
    scen = Scenario(id="esc", service="checkout-api", symptom="high_error_rate", region="us-east", started_at=None, started_minutes_ago=10, received_at="2026-10-22T12:00:00Z", root_cause="unknown", culprit=None, log_signature="none")

    escalated = maybe_escalate(failed_res, "alert input", scen, teacher_client, "SOP")
    assert escalated.escalated is True
    assert escalated.status == "ok"
    assert "status_invalid_output" in escalated.escalate_reason
