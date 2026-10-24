from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class MetricSummary:
    e2e_ok_rate: float
    system_ok_rate: float
    extract_exact_rate: float
    extract_field_acc: Dict[str, float]
    conclusion_exact_rate: float
    conclusion_field_acc: Dict[str, float]
    summary_ok_rate: float
    traj_exact_rate: float
    over_call_rate: float
    under_call_rate: float
    redundant_call_rate: float
    mean_steps_vs_oracle: float
    invalid_output_rate: float
    illegal_transition_rate: float
    escalation_rate: float
    avg_tokens_in: float
    avg_tokens_out: float
    avg_cached_tokens: float
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None


def compute_metrics(traces: List[Dict[str, Any]], oracle_trajectories: List[List[Dict[str, Any]]]) -> MetricSummary:
    n = len(traces)
    if n == 0:
        return MetricSummary(
            0.0, 0.0, 0.0, {}, 0.0, {}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    e2e_ok_cnt = 0
    system_ok_cnt = 0
    extract_exact_cnt = 0
    extract_fields = {"service": 0, "symptom": 0, "region": 0, "started_at": 0, "started_minutes_ago": 0}

    conclusion_exact_cnt = 0
    conclusion_fields = {"root_cause": 0, "runbook_id": 0, "culprit": 0, "log_signature": 0}

    summary_ok_cnt = 0
    traj_exact_cnt = 0
    over_call_cnt = 0
    under_call_cnt = 0
    redundant_call_cnt = 0
    steps_diffs = []

    invalid_cnt = 0
    illegal_cnt = 0
    escalation_cnt = 0

    tokens_in = []
    tokens_out = []
    cached_tokens = []

    for t_item, oracle_hist in zip(traces, oracle_trajectories):
        v = t_item.get("validation", {})
        status = t_item.get("status", "ok")

        if t_item.get("e2e_ok", False):
            e2e_ok_cnt += 1
        if t_item.get("system_ok", False):
            system_ok_cnt += 1
        if t_item.get("escalated", False):
            escalation_cnt += 1

        if status == "invalid_output":
            invalid_cnt += 1
        elif status.startswith("illegal:"):
            illegal_cnt += 1

        # V3 Extraction
        if v.get("V3", {}).get("ok", False):
            extract_exact_cnt += 1

        # V4 Conclusion
        v4_det = v.get("V4", {}).get("details", {})
        if v.get("V4", {}).get("ok", False):
            conclusion_exact_cnt += 1
        for k in conclusion_fields.keys():
            if v4_det.get(k, False):
                conclusion_fields[k] += 1

        # Summary ok (V6-V8)
        if v.get("V6", {}).get("ok", False) and v.get("V7", {}).get("ok", False) and v.get("V8", {}).get("ok", False):
            summary_ok_cnt += 1

        # Trajectory metrics vs oracle
        trace = t_item.get("trace", [])
        model_tools = [e["step"] for e in trace if e.get("step", {}).get("step") not in ("extract_alert", "conclude", "draft_summary")]
        oracle_tools = [c for c, _ in oracle_hist if c.get("step") not in ("extract_alert", "conclude", "draft_summary")]

        diff = len(model_tools) - len(oracle_tools)
        steps_diffs.append(diff)

        if model_tools == oracle_tools:
            traj_exact_cnt += 1
        if len(model_tools) > len(oracle_tools):
            over_call_cnt += 1
        elif len(model_tools) < len(oracle_tools):
            under_call_cnt += 1

        tokens_in.append(t_item.get("tokens_in", 0))
        tokens_out.append(t_item.get("tokens_out", 0))
        cached_tokens.append(t_item.get("cached_tokens", 0))

    return MetricSummary(
        e2e_ok_rate=round(e2e_ok_cnt / n, 4),
        system_ok_rate=round(system_ok_cnt / n, 4),
        extract_exact_rate=round(extract_exact_cnt / n, 4),
        extract_field_acc={k: round(v / n, 4) for k, v in extract_fields.items()},
        conclusion_exact_rate=round(conclusion_exact_cnt / n, 4),
        conclusion_field_acc={k: round(v / n, 4) for k, v in conclusion_fields.items()},
        summary_ok_rate=round(summary_ok_cnt / n, 4),
        traj_exact_rate=round(traj_exact_cnt / n, 4),
        over_call_rate=round(over_call_cnt / n, 4),
        under_call_rate=round(under_call_cnt / n, 4),
        redundant_call_rate=round(redundant_call_cnt / n, 4),
        mean_steps_vs_oracle=round(float(np.mean(steps_diffs)), 2) if steps_diffs else 0.0,
        invalid_output_rate=round(invalid_cnt / n, 4),
        illegal_transition_rate=round(illegal_cnt / n, 4),
        escalation_rate=round(escalation_cnt / n, 4),
        avg_tokens_in=round(float(np.mean(tokens_in)), 1) if tokens_in else 0.0,
        avg_tokens_out=round(float(np.mean(tokens_out)), 1) if tokens_out else 0.0,
        avg_cached_tokens=round(float(np.mean(cached_tokens)), 1) if cached_tokens else 0.0,
    )
