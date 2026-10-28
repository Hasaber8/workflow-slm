import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional
from exp.prompts import load_sop_prompt
from exp.data.split import scenario_from_dict
from exp.runtime.clients import FakeClient, GeminiClient, LocalHfClient, VllmClient
from exp.runtime.loop import canonical_json, run
from exp.teacher.run_traces import create_oracle_teacher_responder
from exp.validators import validate_trace
from exp.world.procedure import expected_next
from exp.world.scenarios import compute_oracle_trajectory
from exp.world.services import REGIONS, SERVICES
from exp.world.simulator import Scenario, Simulator, make_call_key

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "data"
SOP_FILE = Path(__file__).resolve().parent.parent / "prompts" / "sop.md"


def build_scenario_from_text(alert_text: str) -> Scenario:
    """Build a complete simulated scenario from arbitrary alert text for testing."""
    text_lower = alert_text.lower()

    # 1. Match service
    matched_svc = "checkout-api"
    for s in SERVICES:
        s_alias = s.replace("-", " ")
        s_stem = s.split("-")[0]
        if s in text_lower or s_alias in text_lower or (len(s_stem) >= 4 and s_stem in text_lower):
            matched_svc = s
            break

    # 2. Match symptom
    matched_sym = "high_error_rate"
    if "latency" in text_lower or "slow" in text_lower or "timeout" in text_lower or "p99" in text_lower:
        matched_sym = "high_latency"
    elif "cpu" in text_lower:
        matched_sym = "cpu_saturation"
    elif "memory" in text_lower or "oom" in text_lower:
        matched_sym = "memory_saturation"
    elif "error" in text_lower or "5xx" in text_lower or "exception" in text_lower:
        matched_sym = "high_error_rate"

    # 3. Match region
    matched_reg = "us-east"
    for r in REGIONS:
        if r in text_lower:
            matched_reg = r
            break

    # 4. Match start time or minutes ago
    m_ago = 12
    match_min = re.search(r"(\d+)\s*(?:m|min|minutes)", text_lower)
    if match_min:
        m_ago = int(match_min.group(1))

    # 5. Populate states
    deps = SERVICES.get(matched_svc, {}).get("deps", [])
    has_timeouts = "timeout" in text_lower or "downstream" in text_lower
    down_dep = deps[0] if (deps and has_timeouts) else None

    metric_map = {
        "high_error_rate": ("error_rate", 0.18, 0.01, 1700.0),
        "high_latency": ("latency_p99", 450.0, 45.0, 900.0),
        "cpu_saturation": ("cpu", 94.0, 40.0, 135.0),
        "memory_saturation": ("memory", 96.0, 50.0, 92.0),
    }
    m_name, curr, base, pct = metric_map.get(matched_sym, ("error_rate", 0.18, 0.01, 1700.0))

    if down_dep:
        root_cause = "upstream_dependency_failure"
        culprit = down_dep
        sig = "upstream_timeout"
        deploy_min = 95  # not correlated
    elif "deploy" in text_lower or "release" in text_lower or not down_dep:
        root_cause = "bad_deploy"
        culprit = "d-7731"
        sig = "unhandled_exception"
        deploy_min = 15  # correlated
    else:
        root_cause = "transient_noise"
        culprit = None
        sig = "none"
        deploy_min = 120

    return Scenario(
        id="cli_interactive",
        service=matched_svc,
        symptom=matched_sym,
        region=matched_reg,
        started_at=None,
        started_minutes_ago=m_ago,
        received_at="2026-10-22T15:30:00Z",
        root_cause=root_cause,
        culprit=culprit,
        log_signature=sig,
        metrics_state={matched_svc: {m_name: {"current": curr, "baseline": base, "change_pct": pct, "status": "critical"}}},
        deploys_state={matched_svc: [{"id": "d-7731", "kind": "code", "version": "v3.8.2", "minutes_before_start": deploy_min}]},
        logs_state={matched_svc: [
            f"15:21:04 WARN request to {down_dep or 'upstream'} exceeded deadline" if down_dep else "15:21:04 ERROR unhandled exception in request handler",
            "15:21:09 ERROR upstream call failed, returning 502" if down_dep else "15:21:09 FATAL NullPointerException at line 177",
            "15:21:15 INFO healthcheck ping 200 OK",
        ]},
        dependencies_state={matched_svc: deps},
        dependency_health_state={
            d: ({"health": "down", "incident_id": "INC-4471"} if d == down_dep else {"health": "healthy", "incident_id": None})
            for d in deps
        },
        expected_extraction={
            "step": "extract_alert",
            "service": matched_svc,
            "symptom": matched_sym,
            "region": matched_reg,
            "started_at": None,
            "started_minutes_ago": m_ago,
        },
    )


def load_stored_scenario(scen_id: str) -> Optional[Scenario]:
    """Search for scenario across all split files in artifacts/data."""
    for split_name in ("train", "dev", "test", "shift", "probe"):
        file_path = DATA_DIR / f"{split_name}.jsonl"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("id") == scen_id:
                            return scenario_from_dict(data)
    return None


def main():
    parser = argparse.ArgumentParser(description="workflow-slm interactive incident triage CLI")
    parser.add_argument("alert", nargs="?", default=None, help="Raw alert text (e.g. '[FIRING] checkout-api 5xx rate elevated...')")
    parser.add_argument("--model", default=None, help="Model to run: 'oracle', 'gemini', 'ep1', 'ep2', 'ep3', or 'base'")
    parser.add_argument("--scenario", default=None, help="Scenario ID to replay from artifacts/data (e.g. 'scen_test_00001')")
    parser.add_argument("--inject-error", default=None, help="Inject error format: <call_key>:<count>, e.g. 'get_metrics:1'")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="Base URL for local vLLM server")
    args = parser.parse_args()

    # Determine default model
    model_choice = args.model
    if not model_choice:
        model_choice = "oracle"

    # 1. Determine scenario and raw input
    scen: Optional[Scenario] = None
    if args.scenario:
        scen = load_stored_scenario(args.scenario)
        if not scen:
            print(f"Error: Scenario '{args.scenario}' not found in artifacts/data/*.jsonl", file=sys.stderr)
            sys.exit(1)
        raw_input = scen.rendered_alert or f"Received: {scen.received_at}\nAlert on {scen.service}"
        print(f"Loaded Scenario: {scen.id} (True Root Cause: {scen.root_cause}, Culprit: {scen.culprit})")
    elif args.alert:
        raw_input = args.alert
        if not raw_input.startswith("Received:"):
            raw_input = f"Received: 2026-10-22T15:30:00Z\n{raw_input}"
        scen = build_scenario_from_text(args.alert)
    else:
        # Default sample alert
        raw_input = (
            "Received: 2026-10-22T15:30:00Z\n"
            "[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago. lots of timeouts calling downstream. -- pagerduty bot"
        )
        print("No input specified, using default sample alert:\n" + raw_input + "\n")
        scen = build_scenario_from_text(raw_input)

    # 2. Handle error injection
    if args.inject_error:
        parts = args.inject_error.split(":")
        call_key = parts[0]
        cnt = int(parts[1]) if len(parts) > 1 else 1
        scen.injected_errors[call_key] = cnt
        # Also key by prefix if only step name given (e.g. "get_metrics")
        for m in ("error_rate", "latency_p99", "cpu", "memory", "request_rate", "db_pool_utilization"):
            scen.injected_errors[f"get_metrics:{scen.service}:{m}"] = cnt
        print(f"Injected error plan: {call_key} will fail {cnt} time(s)")

    # 3. Select client
    sop_prompt = load_sop_prompt(SOP_FILE)
    if model_choice == "oracle":
        client = FakeClient(response_fn=create_oracle_teacher_responder(scen))
    elif model_choice == "gemini":
        client = GeminiClient(model="gemini-2.5-pro")
    elif model_choice in ("ep1", "ep2", "ep3", "epoch1", "epoch2", "epoch3"):
        adapter_name = "epoch" + model_choice[-1]
        adapter_path = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "adapters" / "triage_v1" / adapter_name
        client = LocalHfClient(adapter_path=str(adapter_path))
    else:
        # Local vLLM server: base, ep1, ep2, ep3
        client = VllmClient(base_url=args.vllm_url, model=model_choice)

    # 4. Execute interaction loop
    sim = Simulator(scen)
    print(f"\n--- Running Interactive Triage with Model: [{model_choice}] ---")
    sys_prompt = sop_prompt if model_choice in ("oracle", "gemini") else "incident_triage_v1"
    res = run(raw_input=raw_input, client=client, world=sim, system_prompt=sys_prompt)

    # 5. Print Step-by-Step Trace
    print(f"\nRun Status: {res.status}")
    print("=" * 60)
    for i, step_entry in enumerate(res.trace, 1):
        step_obj = step_entry["step"]
        step_name = step_obj.get("step")
        print(f"[Turn {i}: {step_name}]")
        print(f"  Model Output : {canonical_json(step_obj)}")

        if "result" in step_entry:
            print(f"  Tool Result  : {canonical_json(step_entry['result'])}")
        elif "decision" in step_entry:
            print(f"  Decision     : {canonical_json(step_entry['decision'])}")
        print("-" * 60)

    # 6. Validate trace if ground truth is available
    if scen.root_cause and scen.root_cause != "unknown":
        from exp.world.services import RUNBOOKS
        truth_dict = {
            "expected_extraction": scen.expected_extraction,
            "root_cause": scen.root_cause,
            "runbook_id": RUNBOOKS[scen.root_cause][0],
            "culprit": scen.culprit,
            "log_signature": scen.log_signature,
        }
        all_ok, report = validate_trace(res.trace, truth_dict)
        print(f"\nTrace Validation: {'PASSED (V1-V8)' if all_ok else 'FAILED'}")
        if not all_ok:
            print(f"Validation Details: {report}")


if __name__ == "__main__":
    main()
