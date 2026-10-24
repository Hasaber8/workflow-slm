import pytest
from exp.world.procedure import expected_next


def test_oracle_unresolvable_alert():
    # 1. Null service
    action, label = expected_next({"service": None, "symptom": "high_error_rate"}, [])
    assert label == "unresolvable_alert"
    assert action["step"] == "conclude"
    assert action["root_cause"] == "unknown"

    # 2. Null symptom
    action2, label2 = expected_next({"service": "checkout-api", "symptom": None}, [])
    assert label2 == "unresolvable_alert"
    assert action2["step"] == "conclude"


def test_oracle_start_metrics_and_normal():
    ex = {"service": "checkout-api", "symptom": "high_error_rate"}
    action, label = expected_next(ex, [])
    assert label == "start_metrics"
    assert action["step"] == "get_metrics"
    assert action["metric"] == "error_rate"

    # If normal -> metrics_normal
    history = [
        (action, {"tool": "get_metrics", "status": "normal"})
    ]
    action2, label2 = expected_next(ex, history)
    assert label2 == "metrics_normal"
    assert action2["step"] == "conclude"
    assert action2["root_cause"] == "transient_noise"


def test_oracle_need_deploys_and_logs():
    ex = {"service": "checkout-api", "symptom": "high_error_rate"}
    m_call = {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}
    history = [(m_call, {"status": "critical"})]

    action, label = expected_next(ex, history)
    assert label == "need_deploys"
    assert action["step"] == "get_recent_deploys"

    dep_call = {"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120}
    history.append((dep_call, {"correlated": False, "correlated_deploy_id": None, "correlated_kind": None}))

    action2, label2 = expected_next(ex, history)
    assert label2 == "need_logs"
    assert action2["step"] == "query_logs"


def test_oracle_deploy_matches():
    ex = {"service": "checkout-api", "symptom": "high_error_rate"}
    m_call = {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}
    dep_call = {"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120}
    log_call = {"step": "query_logs", "service": "checkout-api", "window_minutes": 30}

    # Code deploy match
    history_code = [
        (m_call, {"status": "critical"}),
        (dep_call, {"correlated": True, "correlated_deploy_id": "d-code", "correlated_kind": "code"}),
        (log_call, {"lines": [], "_sig": "unhandled_exception"}),
    ]
    act_c, label_c = expected_next(ex, history_code)
    assert label_c == "deploy_code_match"
    assert act_c["root_cause"] == "bad_deploy"
    assert act_c["culprit"] == "d-code"

    # Config deploy match
    history_cfg = [
        (m_call, {"status": "critical"}),
        (dep_call, {"correlated": True, "correlated_deploy_id": "d-cfg", "correlated_kind": "config"}),
        (log_call, {"lines": [], "_sig": "config_parse_error"}),
    ]
    act_cfg, label_cfg = expected_next(ex, history_cfg)
    assert label_cfg == "deploy_config_match"
    assert act_cfg["root_cause"] == "config_change_error"
    assert act_cfg["culprit"] == "d-cfg"


def test_oracle_upstream_dependencies_chain_positions():
    ex = {"service": "checkout-api", "symptom": "high_error_rate"}
    m_call = {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}
    dep_call = {"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120}
    log_call = {"step": "query_logs", "service": "checkout-api", "window_minutes": 30}
    get_deps_call = {"step": "get_dependencies", "service": "checkout-api"}

    base_hist = [
        (m_call, {"status": "critical"}),
        (dep_call, {"correlated": False, "correlated_deploy_id": None, "correlated_kind": None}),
        (log_call, {"lines": [], "_sig": "upstream_timeout"}),
    ]
    act, label = expected_next(ex, base_hist)
    assert label == "logs_upstream_need_deps"
    assert act["step"] == "get_dependencies"

    base_hist.append((get_deps_call, {"dependencies": ["auth-gateway", "payments-svc", "inventory-svc"]}))

    # Check next is first dependency
    act1, lbl1 = expected_next(ex, base_hist)
    assert lbl1 == "dep_check_next"
    assert act1["dependency"] == "auth-gateway"

    # Unhealthy at position 1 (auth-gateway)
    hist_pos1 = list(base_hist) + [(act1, {"health": "degraded", "incident_id": "INC-1"})]
    act_u1, lbl_u1 = expected_next(ex, hist_pos1)
    assert lbl_u1 == "dep_unhealthy"
    assert act_u1["culprit"] == "auth-gateway"

    # Healthy at pos 1, unhealthy at pos 2 (payments-svc)
    hist_pos2 = list(base_hist) + [
        (act1, {"health": "healthy", "incident_id": None}),
        ({"step": "check_dependency_health", "dependency": "payments-svc"}, {"health": "down", "incident_id": "INC-2"})
    ]
    act_u2, lbl_u2 = expected_next(ex, hist_pos2)
    assert lbl_u2 == "dep_unhealthy"
    assert act_u2["culprit"] == "payments-svc"

    # Healthy at pos 1 & 2, unhealthy at pos 3 (inventory-svc)
    hist_pos3 = list(base_hist) + [
        (act1, {"health": "healthy", "incident_id": None}),
        ({"step": "check_dependency_health", "dependency": "payments-svc"}, {"health": "healthy", "incident_id": None}),
        ({"step": "check_dependency_health", "dependency": "inventory-svc"}, {"health": "down", "incident_id": "INC-3"})
    ]
    act_u3, lbl_u3 = expected_next(ex, hist_pos3)
    assert lbl_u3 == "dep_unhealthy"
    assert act_u3["culprit"] == "inventory-svc"

    # All healthy -> dep_all_healthy -> unknown
    hist_all_h = list(base_hist) + [
        (act1, {"health": "healthy", "incident_id": None}),
        ({"step": "check_dependency_health", "dependency": "payments-svc"}, {"health": "healthy", "incident_id": None}),
        ({"step": "check_dependency_health", "dependency": "inventory-svc"}, {"health": "healthy", "incident_id": None})
    ]
    act_all, lbl_all = expected_next(ex, hist_all_h)
    assert lbl_all == "dep_all_healthy"
    assert act_all["root_cause"] == "unknown"


def test_oracle_metric_confirm_and_reuse():
    # Test metric confirmation for db_pool_exhausted
    ex = {"service": "checkout-api", "symptom": "high_error_rate"}
    m_call = {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}
    dep_call = {"step": "get_recent_deploys", "service": "checkout-api", "window_minutes": 120}
    log_call = {"step": "query_logs", "service": "checkout-api", "window_minutes": 30}

    hist = [
        (m_call, {"status": "critical"}),
        (dep_call, {"correlated": False, "correlated_deploy_id": None, "correlated_kind": None}),
        (log_call, {"lines": [], "_sig": "db_pool_exhausted"}),
    ]
    act, lbl = expected_next(ex, hist)
    assert lbl == "confirm_metric_db_pool_exhausted"
    assert act["metric"] == "db_pool_utilization"

    # Confirmed
    hist_conf = list(hist) + [(act, {"status": "critical"})]
    act_c, lbl_c = expected_next(ex, hist_conf)
    assert lbl_c == "confirmed_db_pool_exhausted"
    assert act_c["root_cause"] == "db_pool_exhausted"

    # Reuse: if symptom was db_pool_saturation, the primary metric is already db_pool_utilization!
    ex_reuse = {"service": "checkout-api", "symptom": "db_pool_saturation"}
    m_primary = {"step": "get_metrics", "service": "checkout-api", "metric": "db_pool_utilization", "window_minutes": 30}
    hist_reuse = [
        (m_primary, {"status": "critical"}),
        (dep_call, {"correlated": False, "correlated_deploy_id": None, "correlated_kind": None}),
        (log_call, {"lines": [], "_sig": "db_pool_exhausted"}),
    ]
    # Should NOT query db_pool_utilization again, should immediately confirm!
    act_reused, lbl_reused = expected_next(ex_reuse, hist_reuse)
    assert lbl_reused == "confirmed_db_pool_exhausted"
    assert act_reused["root_cause"] == "db_pool_exhausted"


def test_oracle_error_retry_and_giveup():
    ex = {"service": "checkout-api", "symptom": "high_error_rate"}
    m_call = {"step": "get_metrics", "service": "checkout-api", "metric": "error_rate", "window_minutes": 30}

    # First error -> error_retry
    hist_err1 = [(m_call, {"tool": "get_metrics", "error": "timeout"})]
    act_r, lbl_r = expected_next(ex, hist_err1)
    assert lbl_r == "error_retry"
    assert act_r == m_call

    # Second error on same call -> error_giveup
    hist_err2 = [
        (m_call, {"tool": "get_metrics", "error": "timeout"}),
        (m_call, {"tool": "get_metrics", "error": "timeout"}),
    ]
    act_g, lbl_g = expected_next(ex, hist_err2)
    assert lbl_g == "error_giveup"
    assert act_g["step"] == "conclude"
    assert act_g["root_cause"] == "unknown"
