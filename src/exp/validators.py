import re
from typing import Any, Dict, List, Optional, Tuple
from exp.schema import STEP_ADAPTER
from exp.world.policy import triage_decision
from exp.world.procedure import expected_next
from exp.world.services import SERVICES

SUMMARY_KEYWORDS = {
    "bad_deploy": ["deploy"],
    "config_change_error": ["config"],
    "upstream_dependency_failure": ["dependency"],  # Or culprit name
    "resource_exhaustion": ["resource", "memory", "cpu", "capacity"],
    "db_pool_exhausted": ["pool"],
    "traffic_spike": ["traffic"],
    "credential_failure": ["credential", "auth"],
    "transient_noise": ["noise", "transient", "recovered"],
    "unknown": ["unknown", "inconclusive", "escalat"],
}


def validate_v1_schema(trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """V1: Every step parses against Step schema."""
    if not trace:
        return False, "empty_trace"
    for i, entry in enumerate(trace):
        step_dict = entry.get("step")
        if not step_dict:
            return False, f"step_{i}_missing_step_data"
        try:
            STEP_ADAPTER.validate_python(step_dict)
        except Exception as e:
            return False, f"step_{i}_schema_error: {e}"
    return True, "ok"


def validate_v2_trajectory(trace: List[Dict[str, Any]], expected_extraction: Dict[str, Any]) -> Tuple[bool, str]:
    """V2: Follow the complete procedure, then conclude and immediately summarize."""
    if len(trace) < 2:
        return False, "trace_too_short"

    first_step = trace[0].get("step", {})
    if first_step.get("step") != "extract_alert":
        return False, "first_step_not_extract_alert"

    history: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for i in range(1, len(trace)):
        curr_step = trace[i].get("step", {})
        exp_action, branch_label = expected_next(expected_extraction, history)
        # Check action equivalence
        if curr_step.get("step") != exp_action.get("step"):
            return False, f"step_{i}_type_mismatch: got {curr_step.get('step')}, expected {exp_action.get('step')} ({branch_label})"

        if curr_step.get("step") == "conclude":
            # V4 checks the conclusion fields. V2 checks that the procedure has
            # actually reached a conclusion and the episode terminates correctly.
            if i + 1 >= len(trace) or trace[i + 1].get("step", {}).get("step") != "draft_summary":
                return False, "conclude_must_be_followed_by_summary"
            if i + 2 != len(trace):
                return False, "unexpected_steps_after_summary"
            return True, "ok"

        for k, v in exp_action.items():
            if curr_step.get(k) != v:
                return False, f"step_{i}_arg_mismatch for {k}: got {curr_step.get(k)}, expected {v}"

        res = trace[i].get("result", {})
        truth = trace[i].get("truth", {})
        pub_with_truth = dict(res)
        for tk, tv in truth.items():
            if tk.startswith("_"):
                pub_with_truth[tk] = tv
        history.append((curr_step, pub_with_truth))

    return False, "missing_conclude_and_summary"


def validate_v3_extraction(trace: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Tuple[bool, str]:
    """V3: Fields equal ground truth."""
    if not trace:
        return False, "empty_trace"
    ext = trace[0].get("step", {})
    if ext.get("step") != "extract_alert":
        return False, "first_step_not_extract_alert"

    for field in ("service", "symptom", "region", "started_at", "started_minutes_ago"):
        expected_val = ground_truth.get(field)
        actual_val = ext.get(field)
        if actual_val != expected_val:
            return False, f"extraction_mismatch_{field}: got {actual_val}, expected {expected_val}"

    return True, "ok"


def validate_v4_conclusion(trace: List[Dict[str, Any]], expected_conclusion: Dict[str, Any]) -> Tuple[bool, Dict[str, bool]]:
    """V4: Conclude step equals the oracle's root_cause, runbook_id, culprit, log_signature."""
    field_results = {
        "root_cause": False,
        "runbook_id": False,
        "culprit": False,
        "log_signature": False,
    }

    conclude_step = None
    for entry in trace:
        step = entry.get("step", {})
        if step.get("step") == "conclude":
            conclude_step = step
            break

    if not conclude_step:
        return False, field_results

    for f in field_results.keys():
        exp_val = expected_conclusion.get(f)
        act_val = conclude_step.get(f)
        if act_val == exp_val:
            field_results[f] = True

    all_pass = all(field_results.values())
    return all_pass, field_results


def validate_v5_decision(trace: List[Dict[str, Any]], extracted: Dict[str, Any], primary_status: Optional[str] = None) -> Tuple[bool, str]:
    """V5: The decision in the trace equals triage_decision() on the trace's facts."""
    conclude_entry = None
    for entry in trace:
        if entry.get("step", {}).get("step") == "conclude":
            conclude_entry = entry
            break

    if not conclude_entry or "decision" not in conclude_entry:
        return False, "missing_conclude_decision"

    actual_dec = conclude_entry["decision"]
    conclude_step = conclude_entry["step"]

    # If primary_status is not passed, find the status of the first successful primary metric call
    if primary_status is None and extracted:
        svc = extracted.get("service")
        sym = extracted.get("symptom")
        from exp.world.procedure import SYMPTOM_TO_METRIC
        primary_metric = SYMPTOM_TO_METRIC.get(sym)
        for entry in trace:
            s_dict = entry.get("step", {})
            r_dict = entry.get("result", {})
            if s_dict.get("step") == "get_metrics" and s_dict.get("service") == svc and s_dict.get("metric") == primary_metric:
                if "error" not in r_dict and "status" in r_dict:
                    primary_status = r_dict["status"]
                    break

    expected_dec = triage_decision(
        extracted.get("service"),
        primary_status,
        conclude_step.get("root_cause"),
        conclude_step.get("culprit")
    )

    if actual_dec != expected_dec:
        return False, f"decision_mismatch: got {actual_dec}, expected {expected_dec}"

    return True, "ok"


def validate_v6_summary_facts(trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """V6: Summary contains severity, runbook ID, team, a root cause keyword, and culprit when not null."""
    summary_text = None
    conclude_step = None
    decision = None

    for entry in trace:
        step = entry.get("step", {})
        if step.get("step") == "conclude":
            conclude_step = step
            decision = entry.get("decision")
        elif step.get("step") == "draft_summary":
            summary_text = step.get("text")

    if not summary_text or not conclude_step or not decision:
        return False, "missing_summary_or_conclude_or_decision"

    text_lower = summary_text.lower()

    # Severity
    sev = decision.get("severity", "").lower()
    if sev and sev not in text_lower:
        return False, f"summary_missing_severity_{sev}"

    # Runbook ID
    rb_id = conclude_step.get("runbook_id", "").lower()
    if rb_id and rb_id not in text_lower:
        return False, f"summary_missing_runbook_{rb_id}"

    # Team
    team = decision.get("team", "").lower()
    if team and team not in text_lower:
        return False, f"summary_missing_team_{team}"

    # Culprit when not null
    culprit = conclude_step.get("culprit")
    if culprit and culprit.lower() not in text_lower:
        return False, f"summary_missing_culprit_{culprit}"

    # Root cause keyword
    rc = conclude_step.get("root_cause")
    keywords = list(SUMMARY_KEYWORDS.get(rc, []))
    if rc == "upstream_dependency_failure" and culprit:
        keywords.append(culprit.lower())

    if not any(kw.lower() in text_lower for kw in keywords):
        return False, f"summary_missing_root_cause_keyword_for_{rc}"

    return True, "ok"


def _extract_all_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+\b", text)


def validate_v7_summary_safe(trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """V7: Every ID, service name, and number in summary appears in extraction, tool results, or decision."""
    summary_text = None
    for entry in trace:
        step = entry.get("step", {})
        if step.get("step") == "draft_summary":
            summary_text = step.get("text")
            break

    if not summary_text:
        return False, "missing_summary"

    # Collect source corpus text (excluding draft_summary itself)
    source_corpus: List[str] = []
    for entry in trace:
        step_dict = entry.get("step", {})
        if step_dict.get("step") == "draft_summary":
            continue
        source_corpus.append(str(step_dict))
        if "result" in entry:
            source_corpus.append(str(entry.get("result", {})))
        if "decision" in entry:
            source_corpus.append(str(entry.get("decision", {})))

    combined_corpus = " ".join(source_corpus)
    combined_corpus_lower = combined_corpus.lower()

    # 1. Check service names in catalog
    for svc in SERVICES.keys():
        if re.search(r"\b" + re.escape(svc) + r"\b", summary_text, re.IGNORECASE):
            if svc.lower() not in combined_corpus_lower:
                return False, f"hallucinated_service_{svc}"

    # 2. Check IDs (runbooks, deploys, incidents)
    # IDs look like RB-\d+, d-\d+, INC-\d+, distractor-\d+
    summary_ids = re.findall(r"\b(?:RB|INC|d|distractor)-\w+\b", summary_text, re.IGNORECASE)
    for sid in summary_ids:
        if sid.lower() not in combined_corpus_lower:
            return False, f"hallucinated_id_{sid}"

    # 3. Check standalone numbers
    summary_numbers = _extract_all_numbers(summary_text)
    corpus_numbers = set(_extract_all_numbers(combined_corpus))
    for num in summary_numbers:
        if num not in corpus_numbers:
            return False, f"hallucinated_number_{num}"

    return True, "ok"


def validate_v8_summary_length(trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """V8: Summary text is within schema limits (<= 700 chars) and not empty."""
    summary_text = None
    for entry in trace:
        step = entry.get("step", {})
        if step.get("step") == "draft_summary":
            summary_text = step.get("text")
            break

    if summary_text is None:
        return False, "missing_summary"
    if len(summary_text.strip()) == 0:
        return False, "empty_summary"
    if len(summary_text) > 700:
        return False, f"summary_too_long: {len(summary_text)} > 700"

    return True, "ok"


def validate_trace(trace: List[Dict[str, Any]], scenario_truth: Dict[str, Any], primary_status: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """Runs V1-V8 validators on a complete trace."""
    report: Dict[str, Any] = {}

    v1_ok, v1_msg = validate_v1_schema(trace)
    report["V1"] = {"ok": v1_ok, "msg": v1_msg}

    v2_ok, v2_msg = validate_v2_trajectory(trace, scenario_truth.get("expected_extraction", {}))
    report["V2"] = {"ok": v2_ok, "msg": v2_msg}

    v3_ok, v3_msg = validate_v3_extraction(trace, scenario_truth.get("expected_extraction", {}))
    report["V3"] = {"ok": v3_ok, "msg": v3_msg}

    exp_conclude = {
        "root_cause": scenario_truth.get("root_cause"),
        "runbook_id": scenario_truth.get("runbook_id"),
        "culprit": scenario_truth.get("culprit"),
        "log_signature": scenario_truth.get("log_signature"),
    }
    v4_ok, v4_details = validate_v4_conclusion(trace, exp_conclude)
    report["V4"] = {"ok": v4_ok, "details": v4_details}

    v5_ok, v5_msg = validate_v5_decision(trace, scenario_truth.get("expected_extraction", {}), primary_status)
    report["V5"] = {"ok": v5_ok, "msg": v5_msg}

    v6_ok, v6_msg = validate_v6_summary_facts(trace)
    report["V6"] = {"ok": v6_ok, "msg": v6_msg}

    v7_ok, v7_msg = validate_v7_summary_safe(trace)
    report["V7"] = {"ok": v7_ok, "msg": v7_msg}

    v8_ok, v8_msg = validate_v8_summary_length(trace)
    report["V8"] = {"ok": v8_ok, "msg": v8_msg}

    all_pass = all([v1_ok, v2_ok, v3_ok, v4_ok, v5_ok, v6_ok, v7_ok, v8_ok])
    return all_pass, report
