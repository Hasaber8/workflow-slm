from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from exp.world.services import SERVICES


def compute_metric_status(change_pct: float) -> str:
    """Status: normal under 25 percent, elevated under 100, else critical."""
    if change_pct < 25.0:
        return "normal"
    elif change_pct < 100.0:
        return "elevated"
    else:
        return "critical"


def make_call_key(step: Any) -> str:
    """Generate a canonical key for error matching."""
    if isinstance(step, dict):
        s_type = step.get("step") or step.get("tool")
        if s_type == "get_metrics":
            return f"get_metrics:{step.get('service')}:{step.get('metric')}"
        elif s_type == "get_recent_deploys":
            return f"get_recent_deploys:{step.get('service')}"
        elif s_type == "query_logs":
            return f"query_logs:{step.get('service')}"
        elif s_type == "get_dependencies":
            return f"get_dependencies:{step.get('service')}"
        elif s_type == "check_dependency_health":
            return f"check_dependency_health:{step.get('dependency')}"
        return f"{s_type}"
    else:
        s_type = getattr(step, "step", None)
        if s_type == "get_metrics":
            return f"get_metrics:{getattr(step, 'service')}:{getattr(step, 'metric')}"
        elif s_type == "get_recent_deploys":
            return f"get_recent_deploys:{getattr(step, 'service')}"
        elif s_type == "query_logs":
            return f"query_logs:{getattr(step, 'service')}"
        elif s_type == "get_dependencies":
            return f"get_dependencies:{getattr(step, 'service')}"
        elif s_type == "check_dependency_health":
            return f"check_dependency_health:{getattr(step, 'dependency')}"
        return f"{s_type}"


@dataclass
class Scenario:
    id: str
    service: Optional[str]
    symptom: Optional[str]
    region: Optional[str]
    started_at: Optional[str]
    started_minutes_ago: Optional[int]
    received_at: str
    root_cause: str
    culprit: Optional[str]
    log_signature: str
    metrics_state: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    deploys_state: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    logs_state: Dict[str, List[str]] = field(default_factory=dict)
    dependencies_state: Dict[str, List[str]] = field(default_factory=dict)
    dependency_health_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    injected_errors: Dict[str, int] = field(default_factory=dict)
    error_type: str = "timeout"
    rendered_alert: Optional[str] = None
    expected_extraction: Dict[str, Any] = field(default_factory=dict)
    oracle_branch_labels: List[str] = field(default_factory=list)


class Simulator:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        # Deepcopy or dict copy error counters so episodes are independent
        self.error_counts = dict(scenario.injected_errors)

    def call(self, step: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Executes a tool step, returning (public_result, truth_annotations)."""
        call_key = make_call_key(step)
        step_dict = step if isinstance(step, dict) else step.model_dump()
        step_name = step_dict.get("step")

        # Check injected errors
        if self.error_counts.get(call_key, 0) > 0:
            self.error_counts[call_key] -= 1
            error_res = {"tool": step_name, "error": self.scenario.error_type}
            truth_res = {"error": self.scenario.error_type}
            return error_res, truth_res

        if step_name == "get_metrics":
            svc = step_dict["service"]
            metric = step_dict["metric"]
            svc_metrics = self.scenario.metrics_state.get(svc, {})
            m_data = svc_metrics.get(metric, {
                "current": 10.0,
                "baseline": 10.0,
                "change_pct": 0.0,
                "status": "normal"
            })
            change_pct = float(m_data["change_pct"])
            status = compute_metric_status(change_pct)
            public = {
                "tool": "get_metrics",
                "service": svc,
                "metric": metric,
                "window_minutes": 30,
                "current": round(float(m_data["current"]), 4),
                "baseline": round(float(m_data["baseline"]), 4),
                "change_pct": round(change_pct, 1),
                "status": status,
            }
            truth = {"status": status, "change_pct": change_pct}
            return public, truth

        elif step_name == "get_recent_deploys":
            svc = step_dict["service"]
            deploys = self.scenario.deploys_state.get(svc, [])
            # Window is 120 minutes
            in_window = [d for d in deploys if d["minutes_before_start"] <= 120]
            # Correlated means a deploy within 45 minutes before the start (<= 45)
            correlated_deploys = [d for d in in_window if d["minutes_before_start"] <= 45]
            is_correlated = len(correlated_deploys) > 0
            corr_id = correlated_deploys[0]["id"] if is_correlated else None
            corr_kind = correlated_deploys[0]["kind"] if is_correlated else None

            public = {
                "tool": "get_recent_deploys",
                "service": svc,
                "window_minutes": 120,
                "deploys": in_window,
                "correlated": is_correlated,
                "correlated_deploy_id": corr_id,
                "correlated_kind": corr_kind,
            }
            truth = {
                "correlated": is_correlated,
                "correlated_deploy_id": corr_id,
                "correlated_kind": corr_kind,
            }
            return public, truth

        elif step_name == "query_logs":
            svc = step_dict["service"]
            lines = self.scenario.logs_state.get(svc, [])
            public = {
                "tool": "query_logs",
                "service": svc,
                "window_minutes": 30,
                "lines": lines,
            }
            truth = {
                "_sig": self.scenario.log_signature
            }
            return public, truth

        elif step_name == "get_dependencies":
            svc = step_dict["service"]
            deps = self.scenario.dependencies_state.get(
                svc, SERVICES.get(svc, {}).get("deps", [])[:3]
            )
            public = {
                "tool": "get_dependencies",
                "service": svc,
                "dependencies": deps,
            }
            truth = {"dependencies": list(deps)}
            return public, truth

        elif step_name == "check_dependency_health":
            dep = step_dict["dependency"]
            health_info = self.scenario.dependency_health_state.get(dep, {
                "health": "healthy",
                "incident_id": None
            })
            public = {
                "tool": "check_dependency_health",
                "dependency": dep,
                "health": health_info["health"],
                "incident_id": health_info.get("incident_id"),
            }
            truth = {
                "health": health_info["health"],
                "incident_id": health_info.get("incident_id"),
            }
            return public, truth

        else:
            raise ValueError(f"Unknown tool step: {step_name}")
