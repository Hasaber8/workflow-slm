from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from exp.data.split import scenario_from_dict
from exp.runtime.loop import canonical_json, try_parse
from exp.schema import STEP_JSON_SCHEMA
from exp.world.procedure import expected_next
from exp.world.scenarios import compute_oracle_trajectory
from exp.world.simulator import Scenario


def are_steps_equal(s1: Dict[str, Any], s2: Dict[str, Any]) -> bool:
    """Checks if model step matches oracle step semantically."""
    if s1.get("step") != s2.get("step"):
        return False
    s_type = s1.get("step")
    if s_type == "extract_alert":
        return all(s1.get(k) == s2.get(k) for k in ("service", "symptom", "region", "started_at", "started_minutes_ago"))
    elif s_type == "get_metrics":
        return s1.get("service") == s2.get("service") and s1.get("metric") == s2.get("metric")
    elif s_type in ("get_recent_deploys", "query_logs", "get_dependencies"):
        return s1.get("service") == s2.get("service")
    elif s_type == "check_dependency_health":
        return s1.get("dependency") == s2.get("dependency")
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
) -> Dict[str, Any]:
    """Runs teacher-forced evaluation, breaking down accuracy by branch label."""
    branch_total = defaultdict(int)
    branch_correct = defaultdict(int)

    for scen in scenarios:
        hist, labels = compute_oracle_trajectory(scen)
        ex = scen.expected_extraction
        raw_input = scen.rendered_alert or f"Received: {scen.received_at}\nAlert on {scen.service}"

        # 1. First step check: extract_alert
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_input}
        ]
        text, _ = client.complete(messages, STEP_JSON_SCHEMA)
        step = try_parse(text)
        branch_total["extract_alert"] += 1
        if step and are_steps_equal(step.model_dump(), ex):
            branch_correct["extract_alert"] += 1

        # Reconstruct message chain from oracle history
        messages.append({"role": "assistant", "content": canonical_json(ex)})

        for i, (oracle_call, oracle_res) in enumerate(hist):
            _, branch_label = expected_next(ex, hist[:i])
            branch_total[branch_label] += 1

            text, _ = client.complete(messages, STEP_JSON_SCHEMA)
            model_step = try_parse(text)

            if model_step and are_steps_equal(model_step.model_dump(), oracle_call):
                branch_correct[branch_label] += 1

            # Advance messages with oracle truth
            messages.append({"role": "assistant", "content": canonical_json(oracle_call)})
            pub_res = {k: v for k, v in oracle_res.items() if not k.startswith("_")}
            messages.append({"role": "user", "content": f"TOOL_RESULT: {canonical_json(pub_res)}"})

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
        "branches": branch_accuracies
    }
