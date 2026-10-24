import datetime
import random
from typing import Any, Dict, List, Optional, Tuple
from exp.world.procedure import SYMPTOM_TO_METRIC, expected_next
from exp.world.services import (
    HELDOUT_SERVICES,
    REGIONS,
    SERVICES,
    TRAIN_SERVICES,
)
from exp.world.simulator import Scenario, Simulator, make_call_key

ALL_SERVICES = list(SERVICES.keys())

SYMPTOMS_LIST = [
    "high_error_rate", "high_latency", "high_cpu", "high_memory",
    "db_pool_saturation", "traffic_surge", "service_down"
]

ALL_METRICS = [
    "error_rate", "latency_p99", "cpu", "memory", "request_rate", "db_pool_utilization"
]

DEFAULT_ROOT_CAUSE_WEIGHTS = {
    "bad_deploy": 14,
    "config_change_error": 8,
    "upstream_dependency_failure": 18,
    "resource_exhaustion": 10,
    "db_pool_exhausted": 9,
    "traffic_spike": 9,
    "credential_failure": 6,
    "transient_noise": 12,
    "unknown": 14,
}


def compute_oracle_trajectory(scenario: Scenario) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Runs the oracle loop against a simulator episode and returns (history, branch_labels)."""
    sim = Simulator(scenario)
    history: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    labels: List[str] = []
    ex = {
        "service": scenario.service,
        "symptom": scenario.symptom,
        "region": scenario.region,
        "started_at": scenario.started_at,
        "started_minutes_ago": scenario.started_minutes_ago,
    }

    for _ in range(20):
        action, branch_label = expected_next(ex, history)
        labels.append(branch_label)
        if action["step"] == "conclude":
            return history, labels
        pub, truth = sim.call(action)
        # Store hidden annotations into public copy for oracle history inspection
        pub_with_truth = dict(pub)
        for k, v in truth.items():
            if k.startswith("_"):
                pub_with_truth[k] = v
        history.append((action, pub_with_truth))

    return history, labels


def generate_scenario(
    scenario_id: str,
    rng: random.Random,
    is_eval: bool = False,
    heldout_fraction: float = 0.25,
    root_cause_weights: Optional[Dict[str, int]] = None,
    transient_error_fraction: float = 0.10,
    persistent_error_fraction: float = 0.03,
    distractor_deploy_fraction: float = 0.10,
    force_root_cause: Optional[str] = None,
    force_subvariant: Optional[str] = None,
) -> Scenario:
    weights = root_cause_weights or DEFAULT_ROOT_CAUSE_WEIGHTS
    
    # 1. Determine root cause and subvariant
    if force_root_cause:
        root_cause = force_root_cause
        subvariant = force_subvariant or "standard"
    else:
        causes = list(weights.keys())
        w_values = [weights[c] for c in causes]
        chosen = rng.choices(causes, weights=w_values, k=1)[0]
        if chosen == "unknown":
            # unknown splits into: inconclusive (7), unresolvable (4), persistent_error (3)
            subvariant = rng.choices(["inconclusive", "unresolvable", "persistent_error"], weights=[7, 4, 3], k=1)[0]
            root_cause = "unknown"
        else:
            root_cause = chosen
            subvariant = "standard"

    # 2. Service selection
    if subvariant == "unresolvable":
        # Unresolvable alerts either have an unknown service outside catalog or null symptom
        unres_type = rng.choice(["unknown_service", "missing_symptom", "both_missing"])
        if unres_type == "unknown_service":
            service = None  # Outside catalog
            symptom = rng.choice(SYMPTOMS_LIST)
        elif unres_type == "missing_symptom":
            service = rng.choice(ALL_SERVICES if is_eval else TRAIN_SERVICES)
            symptom = None
        else:
            service = None
            symptom = None
    else:
        # If upstream_dependency_failure, ensure selected service has dependencies
        if root_cause == "upstream_dependency_failure":
            if is_eval and (rng.random() < heldout_fraction):
                heldout_with_deps = [s for s in HELDOUT_SERVICES if len(SERVICES[s]["deps"]) > 0]
                service = rng.choice(heldout_with_deps)
            else:
                train_with_deps = [s for s in TRAIN_SERVICES if len(SERVICES[s]["deps"]) > 0]
                service = rng.choice(train_with_deps)
        else:
            if is_eval and (rng.random() < heldout_fraction):
                service = rng.choice(HELDOUT_SERVICES)
            else:
                service = rng.choice(TRAIN_SERVICES)

        # Consistent symptom
        if root_cause == "bad_deploy":
            symptom = rng.choice(["high_error_rate", "service_down", "high_latency"])
        elif root_cause == "config_change_error":
            symptom = rng.choice(["high_error_rate", "service_down"])
        elif root_cause == "upstream_dependency_failure":
            symptom = rng.choice(["high_error_rate", "high_latency"])
        elif root_cause == "resource_exhaustion":
            symptom = rng.choice(["high_memory", "high_cpu", "high_latency"])
        elif root_cause == "db_pool_exhausted":
            symptom = rng.choice(["db_pool_saturation", "high_error_rate", "high_latency"])
        elif root_cause == "traffic_spike":
            symptom = rng.choice(["traffic_surge", "high_latency", "high_error_rate"])
        elif root_cause == "credential_failure":
            symptom = "high_error_rate"
        elif root_cause == "transient_noise":
            symptom = rng.choice(SYMPTOMS_LIST)
        else:  # unknown / inconclusive / persistent_error
            symptom = rng.choice(SYMPTOMS_LIST)

    region = rng.choice(REGIONS)

    # 3. Timing: start time and received time
    base_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    start_offset_minutes = rng.randint(10, 1000)
    start_dt = base_time + datetime.timedelta(minutes=start_offset_minutes)
    received_delta = rng.randint(2, 40)
    received_dt = start_dt + datetime.timedelta(minutes=received_delta)
    received_at = received_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    use_relative_time = rng.choice([True, False])
    if use_relative_time:
        started_at = None
        started_minutes_ago = received_delta
    else:
        started_at = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        started_minutes_ago = None

    # 4. Scenario state: metrics, deploys, logs, dependencies
    metrics_state: Dict[str, Dict[str, Dict[str, Any]]] = {}
    deploys_state: Dict[str, List[Dict[str, Any]]] = {}
    logs_state: Dict[str, List[str]] = {}
    dependencies_state: Dict[str, List[str]] = {}
    dependency_health_state: Dict[str, Dict[str, Any]] = {}
    culprit: Optional[str] = None
    log_signature: str = "none"

    if service:
        # Default all services metrics to normal
        for s in ALL_SERVICES:
            metrics_state[s] = {}
            for m in ALL_METRICS:
                base = round(rng.uniform(10.0, 50.0), 2)
                pct = round(rng.uniform(0.0, 20.0), 1)  # normal is < 25%
                metrics_state[s][m] = {
                    "current": round(base * (1 + pct / 100.0), 2),
                    "baseline": base,
                    "change_pct": pct,
                    "status": "normal"
                }

        # Dependencies for the service
        svc_deps = list(SERVICES.get(service, {}).get("deps", []))[:3]
        dependencies_state[service] = svc_deps
        for d in svc_deps:
            dependency_health_state[d] = {"health": "healthy", "incident_id": None}

        # Build cause-specific state
        primary_metric = SYMPTOM_TO_METRIC[symptom] if symptom else "error_rate"

        # Elevated or critical primary metric
        if root_cause != "transient_noise" and subvariant != "unresolvable":
            is_critical = rng.choice([True, False])
            pct = round(rng.uniform(110.0, 500.0) if is_critical else rng.uniform(30.0, 90.0), 1)
            b = metrics_state[service][primary_metric]["baseline"]
            metrics_state[service][primary_metric] = {
                "current": round(b * (1 + pct / 100.0), 2),
                "baseline": b,
                "change_pct": pct,
                "status": "critical" if is_critical else "elevated"
            }

        # Deploys
        deploys: List[Dict[str, Any]] = []
        if root_cause in ("bad_deploy", "config_change_error"):
            kind = "code" if root_cause == "bad_deploy" else "config"
            dep_min = rng.randint(3, 40)  # within 45 min -> correlated
            dep_id = f"d-{rng.randint(1000, 9999)}"
            version = f"v{rng.randint(1, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
            deploys.append({"id": dep_id, "kind": kind, "version": version, "minutes_before_start": dep_min})
            culprit = dep_id
            log_signature = "unhandled_exception" if root_cause == "bad_deploy" else "config_parse_error"
        else:
            # Uncorrelated older deploy (e.g. 50-110 min ago)
            if rng.random() < 0.5:
                deploys.append({
                    "id": f"d-{rng.randint(1000, 9999)}",
                    "kind": rng.choice(["code", "config"]),
                    "version": "v1.0.0",
                    "minutes_before_start": rng.randint(46, 115)
                })

            # Distractor deploy
            if root_cause not in ("bad_deploy", "config_change_error") and (rng.random() < distractor_deploy_fraction):
                # Correlated deploy whose kind does not match the logs
                d_kind = rng.choice(["code", "config"])
                deploys.append({
                    "id": f"distractor-{rng.randint(1000, 9999)}",
                    "kind": d_kind,
                    "version": "v2.0.0",
                    "minutes_before_start": rng.randint(5, 40)
                })

        deploys_state[service] = deploys

        # Logs signature and confirming states
        if root_cause == "upstream_dependency_failure":
            log_signature = rng.choice(["upstream_timeout", "connection_refused"])
            if svc_deps:
                # One dependency is degraded or down
                unhealthy_pos = rng.randint(0, len(svc_deps) - 1)
                culprit = svc_deps[unhealthy_pos]
                inc_id = f"INC-{rng.randint(1000, 9999)}"
                h_status = rng.choice(["degraded", "down"])
                dependency_health_state[culprit] = {"health": h_status, "incident_id": inc_id}
            else:
                # If service has no deps, fallback to inconclusive
                log_signature = "none"

        elif root_cause == "resource_exhaustion":
            log_signature = rng.choice(["oom_killed", "cpu_throttled"])
            target_m = "memory" if log_signature == "oom_killed" else "cpu"
            pct = round(rng.uniform(35.0, 120.0), 1)
            b = metrics_state[service][target_m]["baseline"]
            metrics_state[service][target_m] = {
                "current": round(b * (1 + pct / 100.0), 2),
                "baseline": b,
                "change_pct": pct,
                "status": "critical" if pct >= 100 else "elevated"
            }

        elif root_cause == "db_pool_exhausted":
            log_signature = "db_pool_exhausted"
            pct = round(rng.uniform(35.0, 150.0), 1)
            b = metrics_state[service]["db_pool_utilization"]["baseline"]
            metrics_state[service]["db_pool_utilization"] = {
                "current": round(b * (1 + pct / 100.0), 2),
                "baseline": b,
                "change_pct": pct,
                "status": "critical" if pct >= 100 else "elevated"
            }

        elif root_cause == "traffic_spike":
            log_signature = "none"
            pct = round(rng.uniform(35.0, 200.0), 1)
            b = metrics_state[service]["request_rate"]["baseline"]
            metrics_state[service]["request_rate"] = {
                "current": round(b * (1 + pct / 100.0), 2),
                "baseline": b,
                "change_pct": pct,
                "status": "critical" if pct >= 100 else "elevated"
            }

        elif root_cause == "credential_failure":
            log_signature = "auth_failure"

        elif root_cause == "transient_noise":
            # Primary metric normal (< 25%)
            pct = round(rng.uniform(0.0, 20.0), 1)
            b = metrics_state[service][primary_metric]["baseline"]
            metrics_state[service][primary_metric] = {
                "current": round(b * (1 + pct / 100.0), 2),
                "baseline": b,
                "change_pct": pct,
                "status": "normal"
            }
            log_signature = rng.choice([
                "none", "auth_failure", "oom_killed", "cpu_throttled", "upstream_timeout"
            ])

        elif subvariant == "inconclusive":
            # One of 4 inconclusive patterns
            inconclusive_choice = rng.choice([
                "upstream_all_healthy", "unhandled_no_deploy", "oom_normal_memory", "none_normal_req_rate"
            ])
            if inconclusive_choice == "upstream_all_healthy":
                log_signature = rng.choice(["upstream_timeout", "connection_refused"])
                for d in svc_deps:
                    dependency_health_state[d] = {"health": "healthy", "incident_id": None}
            elif inconclusive_choice == "unhandled_no_deploy":
                log_signature = "unhandled_exception"
                deploys_state[service] = []  # No deploy
            elif inconclusive_choice == "oom_normal_memory":
                log_signature = "oom_killed"
                metrics_state[service]["memory"]["change_pct"] = 10.0
                metrics_state[service]["memory"]["status"] = "normal"
            else:
                log_signature = "none"
                metrics_state[service]["request_rate"]["change_pct"] = 10.0
                metrics_state[service]["request_rate"]["status"] = "normal"

        # Generate placeholder raw log lines for the signature
        logs_state[service] = [
            f"12:00:{10+i:02d} INFO {service}: routine event {i}" for i in range(5)
        ]

    # Construct the baseline Scenario
    scen = Scenario(
        id=scenario_id,
        service=service,
        symptom=symptom,
        region=region,
        started_at=started_at,
        started_minutes_ago=started_minutes_ago,
        received_at=received_at,
        root_cause=root_cause,
        culprit=culprit,
        log_signature=log_signature,
        metrics_state=metrics_state,
        deploys_state=deploys_state,
        logs_state=logs_state,
        dependencies_state=dependencies_state,
        dependency_health_state=dependency_health_state,
        injected_errors={},
        error_type=rng.choice(["timeout", "rate_limited", "internal_error"]),
        expected_extraction={
            "step": "extract_alert",
            "service": service,
            "symptom": symptom,
            "region": region,
            "started_at": started_at,
            "started_minutes_ago": started_minutes_ago,
        }
    )

    # 5. Error injection planning
    # First, run error-free oracle to find possible call targets
    hist, labels = compute_oracle_trajectory(scen)
    tool_calls = [c for c, _ in hist if c.get("step") != "conclude"]

    if tool_calls:
        if subvariant == "persistent_error":
            # Persistent error fails twice
            target_call = rng.choice(tool_calls)
            call_k = make_call_key(target_call)
            scen.injected_errors[call_k] = 2
        elif rng.random() < persistent_error_fraction:
            target_call = rng.choice(tool_calls)
            call_k = make_call_key(target_call)
            scen.injected_errors[call_k] = 2
        elif rng.random() < transient_error_fraction:
            target_call = rng.choice(tool_calls)
            call_k = make_call_key(target_call)
            scen.injected_errors[call_k] = 1

    # Re-run oracle with injected errors to capture actual ground truth labels
    final_hist, final_labels = compute_oracle_trajectory(scen)
    scen.oracle_branch_labels = final_labels

    return scen


def generate_scenarios(
    count: int,
    seed: int = 1234,
    is_eval: bool = False,
    heldout_fraction: float = 0.25,
    root_cause_weights: Optional[Dict[str, int]] = None,
    transient_error_fraction: float = 0.10,
    persistent_error_fraction: float = 0.03,
    distractor_deploy_fraction: float = 0.10,
) -> List[Scenario]:
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    for i in range(count):
        scen_id = f"scen_{'eval' if is_eval else 'train'}_{i:05d}"
        scen = generate_scenario(
            scen_id,
            rng,
            is_eval=is_eval,
            heldout_fraction=heldout_fraction,
            root_cause_weights=root_cause_weights,
            transient_error_fraction=transient_error_fraction,
            persistent_error_fraction=persistent_error_fraction,
            distractor_deploy_fraction=distractor_deploy_fraction,
        )
        scenarios.append(scen)
    return scenarios
