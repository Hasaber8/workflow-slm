from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from exp.data.split import scenario_from_dict
from exp.runtime.loop import canonical_json, try_parse
from exp.schema import STEP_JSON_SCHEMA
from exp.world.policy import triage_decision
from exp.world.procedure import SYMPTOM_TO_METRIC, expected_next
from exp.world.scenarios import compute_oracle_trajectory
from exp.world.simulator import Scenario
from exp.validators import validate_v6_summary_facts, validate_v7_summary_safe, validate_v8_summary_length


def are_steps_equal(s1: Dict[str, Any], s2: Dict[str, Any]) -> bool:
    """Checks if model step matches oracle step semantically."""
    if s1.get("step") != s2.get("step"):
        return False
    s_type = s1.get("step")
    if s_type == "extract_alert":
        return all(s1.get(k) == s2.get(k) for k in ("service", "symptom", "region", "started_at", "started_minutes_ago"))
    elif s_type in ("get_metrics", "get_recent_deploys", "query_logs", "get_dependencies", "check_dependency_health"):
        return all(s1.get(key) == value for key, value in s2.items())
    elif s_type == "conclude":
        return all(s1.get(k) == s2.get(k) for k in ("root_cause", "runbook_id", "culprit", "log_signature"))
    elif s_type == "draft_summary":
        # Draft summary text does not need exact matching in teacher-forced
        return True
    return s1 == s2


def evaluate_teacher_forced(
    client: Any,
    scenarios: List[Scenario],
    system_prompt: str,
    client_factory: Optional[Callable[[Scenario], Any]] = None,
) -> Dict[str, Any]:
    """Runs teacher-forced evaluation, breaking down accuracy by branch label."""
    branch_total = defaultdict(int)
    branch_correct = defaultdict(int)
    phase_total = defaultdict(int)
    phase_correct = defaultdict(int)

    def record(label, phase, correct):
        branch_total[label] += 1
        phase_total[phase] += 1
        if correct:
            branch_correct[label] += 1
            phase_correct[phase] += 1

    for scen in scenarios:
        target_client = client_factory(scen) if client_factory else client
        hist, labels = compute_oracle_trajectory(scen)
        ex = {"step": "extract_alert", **scen.expected_extraction}
        raw_input = scen.rendered_alert or f"Received: {scen.received_at}\nAlert on {scen.service}"

        # 1. First step check: extract_alert
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_input}
        ]
        text, _ = target_client.complete(messages, STEP_JSON_SCHEMA)
        step = try_parse(text)
        record("extract_alert", "extract_alert", step is not None and are_steps_equal(step.model_dump(), ex))

        # Reconstruct message chain from oracle history
        messages.append({"role": "assistant", "content": canonical_json(ex)})
        messages.append({"role": "user", "content": "TOOL_RESULT: continue"})
        oracle_trace = [{"step": ex}]

        for i, (oracle_call, oracle_res) in enumerate(hist):
            _, branch_label = expected_next(ex, hist[:i])
            text, _ = target_client.complete(messages, STEP_JSON_SCHEMA)
            model_step = try_parse(text)
            record(branch_label, "tool_calls", model_step is not None and are_steps_equal(model_step.model_dump(), oracle_call))

            # Advance messages with oracle truth
            messages.append({"role": "assistant", "content": canonical_json(oracle_call)})
            pub_res = {k: v for k, v in oracle_res.items() if not k.startswith("_")}
            messages.append({"role": "user", "content": f"TOOL_RESULT: {canonical_json(pub_res)}"})
            oracle_trace.append({"step": oracle_call, "result": pub_res})

        # Score the conclusion too, independently of earlier model mistakes.
        conclusion, branch_label = expected_next(ex, hist)
        text, _ = target_client.complete(messages, STEP_JSON_SCHEMA)
        model_step = try_parse(text)
        record(branch_label, "conclude", model_step is not None and are_steps_equal(model_step.model_dump(), conclusion))

        primary_metric = SYMPTOM_TO_METRIC.get(ex.get("symptom"))
        primary_status = next((
            result.get("status") for call, result in hist
            if call["step"] == "get_metrics" and call.get("metric") == primary_metric and "error" not in result
        ), None)
        decision = triage_decision(ex.get("service"), primary_status, conclusion["root_cause"], conclusion.get("culprit"))
        messages.append({"role": "assistant", "content": canonical_json(conclusion)})
        messages.append({"role": "user", "content": "TOOL_RESULT: " + canonical_json({"tool": "decision", **decision})})
        oracle_trace.append({"step": conclusion, "decision": decision})

        text, _ = target_client.complete(messages, STEP_JSON_SCHEMA)
        summary = try_parse(text)
        summary_ok = False
        if summary is not None and summary.step == "draft_summary":
            summary_trace = oracle_trace + [{"step": summary.model_dump()}]
            summary_ok = all(check(summary_trace)[0] for check in (
                validate_v6_summary_facts, validate_v7_summary_safe, validate_v8_summary_length,
            ))
        record("draft_summary", "draft_summary", summary_ok)

    # Aggregate accuracies
    branch_accuracies = {}
    for label, tot in branch_total.items():
        corr = branch_correct[label]
        acc = round(corr / tot, 4) if tot > 0 else 0.0
        branch_accuracies[label] = {
            "correct": corr,
            "total": tot,
            "accuracy": acc
        }

    overall_total = sum(branch_total.values())
    overall_correct = sum(branch_correct.values())
    overall_acc = round(overall_correct / overall_total, 4) if overall_total > 0 else 0.0

    return {
        "overall_accuracy": overall_acc,
        "branches": branch_accuracies,
        "phases": {
            phase: {
                "correct": phase_correct[phase], "total": phase_total[phase],
                "accuracy": round(phase_correct[phase] / phase_total[phase], 4) if phase_total[phase] else 0.0,
            }
            for phase in ("extract_alert", "tool_calls", "conclude", "draft_summary")
        },
    }
