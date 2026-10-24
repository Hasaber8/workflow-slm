from exp.world.services import RUNBOOKS

SYMPTOM_TO_METRIC = {
    "high_error_rate": "error_rate",
    "high_latency": "latency_p99",
    "high_cpu": "cpu",
    "high_memory": "memory",
    "db_pool_saturation": "db_pool_utilization",
    "traffic_surge": "request_rate",
    "service_down": "error_rate",
}


def call(step, **kw):
    return {"step": step, **kw}


def conclude(cause, culprit, sig):
    return {
        "step": "conclude",
        "root_cause": cause,
        "runbook_id": RUNBOOKS[cause][0],
        "culprit": culprit,
        "log_signature": sig,
    }


def first_ok(ok, step, **kw):
    for c, r in ok:
        if c["step"] == step and all(c.get(k) == v for k, v in kw.items()):
            return r
    return None


def expected_next(ex, history):
    """Returns (action_dict, branch_label)."""
    svc, sym = ex.get("service"), ex.get("symptom")
    if svc is None or sym is None:
        return conclude("unknown", None, None), "unresolvable_alert"
    if history:
        last_call, last_res = history[-1]
        if "error" in last_res:
            n = 0
            for c, r in reversed(history):
                if c == last_call and "error" in r:
                    n += 1
                else:
                    break
            if n >= 2:
                logs = first_ok([(c, r) for c, r in history if "error" not in r], "query_logs")
                return conclude("unknown", None, logs["_sig"] if logs else None), "error_giveup"
            return last_call, "error_retry"
    ok = [(c, r) for c, r in history if "error" not in r]
    primary = SYMPTOM_TO_METRIC[sym]
    m1 = first_ok(ok, "get_metrics", metric=primary)
    if m1 is None:
        return call("get_metrics", service=svc, metric=primary, window_minutes=30), "start_metrics"
    if m1["status"] == "normal":
        return conclude("transient_noise", None, None), "metrics_normal"
    dep = first_ok(ok, "get_recent_deploys")
    if dep is None:
        return call("get_recent_deploys", service=svc, window_minutes=120), "need_deploys"
    logs = first_ok(ok, "query_logs")
    if logs is None:
        return call("query_logs", service=svc, window_minutes=30), "need_logs"
    sig = logs["_sig"]
    if dep["correlated"]:
        if dep["correlated_kind"] == "config" and sig == "config_parse_error":
            return conclude("config_change_error", dep["correlated_deploy_id"], sig), "deploy_config_match"
        if dep["correlated_kind"] == "code" and sig == "unhandled_exception":
            return conclude("bad_deploy", dep["correlated_deploy_id"], sig), "deploy_code_match"
    if sig in ("upstream_timeout", "connection_refused"):
        deps = first_ok(ok, "get_dependencies")
        if deps is None:
            return call("get_dependencies", service=svc), "logs_upstream_need_deps"
        for d in deps["dependencies"]:
            h = first_ok(ok, "check_dependency_health", dependency=d)
            if h is None:
                return call("check_dependency_health", dependency=d), "dep_check_next"
            if h["health"] != "healthy":
                return conclude("upstream_dependency_failure", d, sig), "dep_unhealthy"
        return conclude("unknown", None, sig), "dep_all_healthy"
    confirm = {
        "db_pool_exhausted": ("db_pool_utilization", "db_pool_exhausted"),
        "oom_killed": ("memory", "resource_exhaustion"),
        "cpu_throttled": ("cpu", "resource_exhaustion"),
        "none": ("request_rate", "traffic_spike")
    }
    if sig in confirm:
        metric, cause = confirm[sig]
        m = first_ok(ok, "get_metrics", metric=metric)
        if m is None:
            return call("get_metrics", service=svc, metric=metric, window_minutes=30), "confirm_metric_" + sig
        if m["status"] != "normal":
            return conclude(cause, None, sig), "confirmed_" + sig
        return conclude("unknown", None, sig), "confirm_failed_" + sig
    if sig == "auth_failure":
        return conclude("credential_failure", None, sig), "logs_auth"
    return conclude("unknown", None, sig), "logs_no_match"
