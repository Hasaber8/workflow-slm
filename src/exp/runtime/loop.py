from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from exp.runtime.structure import TOOL_STEPS, structural_problem
from exp.schema import STEP_ADAPTER, STEP_JSON_SCHEMA, ExtractAlert, Step
from exp.world.policy import triage_decision
from exp.world.procedure import SYMPTOM_TO_METRIC


def canonical_json(obj: Any) -> str:
    """Formats an object or model into compact canonical JSON with no unnecessary whitespace."""
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
    elif isinstance(obj, dict):
        data = obj
    else:
        data = obj
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def try_parse(text: str) -> Optional[Step]:
    """Tries to parse model text into a validated Step object."""
    try:
        clean_text = text.strip()
        # Handle possible markdown fencing if returned
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        data = json.loads(clean_text)
        try:
            return STEP_ADAPTER.validate_python(data)
        except Exception:
            # If from flattened schema or minor key alias, clean up fields
            cleaned = {k: v for k, v in data.items() if v is not None}
            if "started_minutes" in cleaned and "started_minutes_ago" not in cleaned:
                cleaned["started_minutes_ago"] = cleaned.pop("started_minutes")
            # For window_minutes, ensure default if missing
            if cleaned.get("step") == "get_metrics":
                if "window_minutes" not in cleaned:
                    cleaned["window_minutes"] = 30
                m = str(cleaned.get("metric", ""))
                if "latency" in m or "p95" in m or "p99" in m:
                    cleaned["metric"] = "latency_p99"
                elif "error" in m or "5xx" in m:
                    cleaned["metric"] = "error_rate"
                elif "cpu" in m:
                    cleaned["metric"] = "cpu"
                elif "mem" in m:
                    cleaned["metric"] = "memory"
                elif "pool" in m or "db" in m:
                    cleaned["metric"] = "db_pool_utilization"
                elif "req" in m or "traffic" in m:
                    cleaned["metric"] = "request_rate"
            elif cleaned.get("step") == "get_recent_deploys" and "window_minutes" not in cleaned:
                cleaned["window_minutes"] = 120
            elif cleaned.get("step") == "query_logs" and "window_minutes" not in cleaned:
                cleaned["window_minutes"] = 30
            elif cleaned.get("step") == "conclude":
                from exp.world.services import RUNBOOKS
                rc = cleaned.get("root_cause")
                if rc in RUNBOOKS:
                    valid_rbs = {rb[0] for rb in RUNBOOKS.values()}
                    if cleaned.get("runbook_id") not in valid_rbs:
                        cleaned["runbook_id"] = RUNBOOKS[rc][0]
            return STEP_ADAPTER.validate_python(cleaned)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return None


def is_primary_metric_ok(step: Any, public: Dict[str, Any], extracted: Optional[ExtractAlert]) -> bool:
    """Checks if a step was a successful query for the primary metric."""
    if extracted is None or extracted.symptom is None:
        return False
    step_dict = step if isinstance(step, dict) else step.model_dump()
    if step_dict.get("step") != "get_metrics":
        return False
    if "error" in public:
        return False
    expected_primary = SYMPTOM_TO_METRIC.get(extracted.symptom)
    return (
        step_dict.get("metric") == expected_primary and
        step_dict.get("service") == extracted.service
    )


@dataclass
class Result:
    status: str
    trace: List[Dict[str, Any]] = field(default_factory=list)
    extracted: Optional[ExtractAlert] = None
    escalated: bool = False
    escalate_reason: Optional[str] = None


def run(
    raw_input: str,
    client: Any,
    world: Any,
    workflow_id: str = "incident_triage_v1",
    system_prompt: Optional[str] = None,
    max_steps: int = 16
) -> Result:
    """Runs the incident triage interaction loop."""
    sys_msg = system_prompt if system_prompt else workflow_id
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": raw_input}
    ]
    state: str = "START"
    extracted: Optional[ExtractAlert] = None
    trace: List[Dict[str, Any]] = []
    primary_status: Optional[str] = None

    for _ in range(max_steps):
        text, usage = client.complete(messages, STEP_JSON_SCHEMA)
        step = try_parse(text)
        if step is None:
            return Result(status="invalid_output", trace=trace)

        problem = structural_problem(state, step, extracted, trace)
        if problem:
            return Result(status="illegal:" + problem, trace=trace)

        messages.append({"role": "assistant", "content": canonical_json(step)})
        entry: Dict[str, Any] = {"step": step.model_dump(), "usage": usage}

        if step.step == "extract_alert":
            extracted = step
            messages.append({"role": "user", "content": "TOOL_RESULT: continue"})
        elif step.step in TOOL_STEPS:
            public, truth = world.call(step)
            entry.update(result=public, truth=truth)
            if primary_status is None and is_primary_metric_ok(step, public, extracted):
                primary_status = public.get("status")
            messages.append({"role": "user", "content": "TOOL_RESULT: " + canonical_json(public)})
        elif step.step == "conclude":
            service_name = extracted.service if extracted else None
            decision = triage_decision(service_name, primary_status, step.root_cause, step.culprit)
            entry.update(decision=decision)
            messages.append({
                "role": "user",
                "content": "TOOL_RESULT: " + canonical_json({"tool": "decision", **decision})
            })

        trace.append(entry)

        if step.step == "draft_summary":
            return Result(status="ok", trace=trace, extracted=extracted)

        state = step.step

    return Result(status="max_steps", trace=trace)
