import pytest
from exp.world.simulator import Scenario, Simulator, compute_metric_status, make_call_key


def test_metric_threshold_crossing():
    # under 25 percent -> normal
    assert compute_metric_status(0.0) == "normal"
    assert compute_metric_status(24.9) == "normal"
    # 25 up to under 100 -> elevated
    assert compute_metric_status(25.0) == "elevated"
    assert compute_metric_status(99.9) == "elevated"
    # 100 and above -> critical
    assert compute_metric_status(100.0) == "critical"
    assert compute_metric_status(250.0) == "critical"


def test_correlation_window_boundary():
    # Correlated means deploy within 45 minutes before start (<= 45)
    # Test boundary with 44, 45, and 46 minutes
    scen = Scenario(
        id="test_corr",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at="2026-10-22T12:00:00Z",
        started_minutes_ago=None,
        received_at="2026-10-22T12:15:00Z",
        root_cause="bad_deploy",
        culprit="d-1",
        log_signature="unhandled_exception",
        deploys_state={
            "checkout-api": [
                {"id": "d-44", "kind": "code", "version": "v1", "minutes_before_start": 44},
                {"id": "d-45", "kind": "code", "version": "v2", "minutes_before_start": 45},
                {"id": "d-46", "kind": "code", "version": "v3", "minutes_before_start": 46},
            ]
        }
    )
    sim = Simulator(scen)
    pub, truth = sim.call({"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120})
    assert pub["correlated"] is True
    assert pub["correlated_deploy_id"] == "d-44"

    # Only 46 min deploy -> NOT correlated
    scen2 = Scenario(
        id="test_corr2",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at="2026-10-22T12:00:00Z",
        started_minutes_ago=None,
        received_at="2026-10-22T12:15:00Z",
        root_cause="transient_noise",
        culprit=None,
        log_signature="none",
        deploys_state={
            "checkout-api": [
                {"id": "d-46", "kind": "code", "version": "v3", "minutes_before_start": 46},
            ]
        }
    )
    sim2 = Simulator(scen2)
    pub2, truth2 = sim2.call({"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120})
    assert pub2["correlated"] is False
    assert pub2["correlated_deploy_id"] is None


def test_hidden_annotations_never_leak_in_public():
    scen = Scenario(
        id="test_hidden",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at="2026-10-22T12:00:00Z",
        started_minutes_ago=None,
        received_at="2026-10-22T12:15:00Z",
        root_cause="bad_deploy",
        culprit="d-1",
        log_signature="unhandled_exception",
        logs_state={"checkout-api": ["line1", "line2", "line3", "line4", "line5"]}
    )
    sim = Simulator(scen)
    pub, truth = sim.call({"step": "query_logs", "service": "checkout-api", "window_minutes": 30})
    # Public must contain lines, but NOT _sig
    assert "lines" in pub
    assert "_sig" not in pub
    assert truth.get("_sig") == "unhandled_exception"


def test_error_injection_transient_and_persistent():
    call_m = {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}
    call_k = make_call_key(call_m)

    # 1. Transient: fails once, then succeeds
    scen_trans = Scenario(
        id="trans",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at="2026-10-22T12:00:00Z",
        started_minutes_ago=None,
        received_at="2026-10-22T12:15:00Z",
        root_cause="transient_noise",
        culprit=None,
        log_signature="none",
        injected_errors={call_k: 1},
        error_type="timeout",
        metrics_state={"checkout-api": {"error_rate": {"current": 10, "baseline": 10, "change_pct": 0, "status": "normal"}}}
    )
    sim_t = Simulator(scen_trans)
    res1, truth1 = sim_t.call(call_m)
    assert "error" in res1
    assert res1["error"] == "timeout"

    res2, truth2 = sim_t.call(call_m)
    assert "error" not in res2
    assert res2["status"] == "normal"

    # 2. Persistent: fails twice
    scen_pers = Scenario(
        id="pers",
        service="checkout-api",
        symptom="high_error_rate",
        region="us-east",
        started_at="2026-10-22T12:00:00Z",
        started_minutes_ago=None,
        received_at="2026-10-22T12:15:00Z",
        root_cause="unknown",
        culprit=None,
        log_signature="none",
        injected_errors={call_k: 2},
        error_type="rate_limited",
        metrics_state={"checkout-api": {"error_rate": {"current": 10, "baseline": 10, "change_pct": 0, "status": "normal"}}}
    )
    sim_p = Simulator(scen_pers)
    res_p1, _ = sim_p.call(call_m)
    assert res_p1.get("error") == "rate_limited"
    res_p2, _ = sim_p.call(call_m)
    assert res_p2.get("error") == "rate_limited"
    res_p3, _ = sim_p.call(call_m)
    assert "error" not in res_p3
