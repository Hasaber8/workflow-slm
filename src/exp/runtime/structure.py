from typing import Any, Dict, List, Optional
from exp.schema import Step

TOOL_STEPS = {
    "get_metrics", "get_recent_deploys", "query_logs", "get_dependencies", "check_dependency_health"
}


def are_calls_equal(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
    """Compare two step dictionaries for functional equality."""
    if c1.get("step") != c2.get("step"):
        return False
    step_type = c1.get("step")
    if step_type == "get_metrics":
        return c1.get("service") == c2.get("service") and c1.get("metric") == c2.get("metric")
    elif step_type in ("get_recent_deploys", "query_logs", "get_dependencies"):
        return c1.get("service") == c2.get("service")
    elif step_type == "check_dependency_health":
        return c1.get("dependency") == c2.get("dependency")
    return c1 == c2


def structural_problem(
    state: str,
    step: Step,
    extracted: Optional[Any],
    trace: List[Dict[str, Any]],
) -> Optional[str]:
    """Enforces structural legality checks that runtime can verify without oracle."""
    step_dict = step.model_dump()
    s_name = step.step

    # 1. First step must be extract_alert, and can only appear once
    if state == "START":
        if s_name != "extract_alert":
            return "first_step_must_be_extract_alert"
        return None

    if s_name == "extract_alert":
        return "extract_alert_can_appear_only_once"

    # 2. Conclude and draft_summary flow
    if state == "conclude":
        if s_name != "draft_summary":
            return "draft_summary_must_immediately_follow_conclude"
        return None

    if state == "draft_summary":
        return "nothing_may_follow_draft_summary"

    if s_name == "draft_summary":
        if state != "conclude":
            return "draft_summary_must_immediately_follow_conclude"

    # 3. Unresolvable alert constraint
    if extracted is not None and (extracted.service is None or extracted.symptom is None):
        if s_name != "conclude":
            return "unresolvable_alert_requires_conclude"

    # 4. Tool step validations
    if s_name in TOOL_STEPS:
        if extracted is None or extracted.service is None:
            return "cannot_call_tools_without_extracted_service"

        # Check service match
        if s_name in ("get_metrics", "get_recent_deploys", "query_logs", "get_dependencies"):
            if step_dict.get("service") != extracted.service:
                return f"tool_service_mismatch: {step_dict.get('service')} != {extracted.service}"

        # Check check_dependency_health.dependency
        if s_name == "check_dependency_health":
            dep = step_dict.get("dependency")
            seen_deps: List[str] = []
            for entry in trace:
                res = entry.get("result", {})
                if isinstance(res, dict) and "dependencies" in res:
                    seen_deps.extend(res.get("dependencies", []))
            if dep not in seen_deps:
                return f"dependency_{dep}_not_in_prior_get_dependencies"

        # 5. Repetition and retry checks
        # Check if identical call previously succeeded
        for entry in trace:
            prev_step = entry.get("step", {})
            prev_res = entry.get("result", {})
            if are_calls_equal(step_dict, prev_step) and "error" not in prev_res:
                return "repeated_successful_call"

        # Check error retry constraints
        # If the call is repeated after error:
        # Check if the immediate last call had an error
        if trace:
            last_entry = trace[-1]
            last_step = last_entry.get("step", {})
            last_res = last_entry.get("result", {})

            if are_calls_equal(step_dict, last_step):
                # We are repeating the immediately preceding call
                if "error" not in last_res:
                    return "repeated_successful_call"
                # Count consecutive errors for this call
                error_count = 0
                for entry in reversed(trace):
                    if are_calls_equal(step_dict, entry.get("step", {})) and "error" in entry.get("result", {}):
                        error_count += 1
                    else:
                        break
                if error_count >= 2:
                    return "exceeded_retry_limit_max_1_retry"
            else:
                # If we are repeating a call that errored in the past but not immediately before
                for entry in trace[:-1]:
                    if are_calls_equal(step_dict, entry.get("step", {})):
                        return "non_consecutive_retry"

    return None
