from typing import Any, Dict, Optional
from exp.runtime.loop import Result, run
from exp.validators import validate_v6_summary_facts, validate_v7_summary_safe, validate_v8_summary_length
from exp.world.simulator import Simulator


def maybe_escalate(
    result: Result,
    raw_input: str,
    scenario: Any,
    teacher_client: Any,
    sop_prompt: str,
) -> Result:
    """Escalates to the teacher model if the primary run failed or summary was invalid."""
    needs_escalation = False
    reason = None

    if result.status != "ok":
        needs_escalation = True
        reason = f"status_{result.status}"
    else:
        # Check summary validators
        v6_ok, v6_msg = validate_v6_summary_facts(result.trace)
        v7_ok, v7_msg = validate_v7_summary_safe(result.trace)
        v8_ok, v8_msg = validate_v8_summary_length(result.trace)
        if not (v6_ok and v7_ok and v8_ok):
            needs_escalation = True
            reason = f"summary_failed: v6={v6_ok}, v7={v7_ok}, v8={v8_ok}"

    if not needs_escalation:
        return result

    # Fresh episode for escalation fallback
    fresh_sim = Simulator(scenario)
    escalated_res = run(
        raw_input=raw_input,
        client=teacher_client,
        world=fresh_sim,
        system_prompt=sop_prompt,
    )
    escalated_res.escalated = True
    escalated_res.escalate_reason = reason
    return escalated_res
