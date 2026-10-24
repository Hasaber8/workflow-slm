import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
from exp.config import load_config
from exp.data.split import scenario_from_dict
from exp.runtime.clients import FakeClient, GeminiClient, VllmClient
from exp.runtime.escalate import maybe_escalate
from exp.runtime.loop import run
from exp.teacher.run_traces import create_oracle_teacher_responder
from exp.validators import validate_trace
from exp.world.procedure import expected_next
from exp.world.scenarios import compute_oracle_trajectory
from exp.world.services import RUNBOOKS
from exp.world.simulator import Scenario, Simulator

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "data"
BASELINES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "baselines"
SOP_FILE = Path(__file__).resolve().parent.parent / "prompts" / "sop.md"


def run_baseline_on_split(
    baseline_id: str,
    split_name: str,
    client: Any,
    system_prompt: Optional[str] = None,
    limit: Optional[int] = None,
    fallback_client: Optional[Any] = None
) -> Dict[str, Any]:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    split_file = DATA_DIR / f"{split_name}.jsonl"
    sop_prompt = SOP_FILE.read_text(encoding="utf-8") if system_prompt is None else system_prompt

    scenarios: List[Scenario] = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(scenario_from_dict(json.loads(line)))

    if limit:
        scenarios = scenarios[:limit]

    results: List[Dict[str, Any]] = []
    start_t = time.time()
    n_e2e_ok = 0
    n_system_ok = 0
    total_tokens_in = 0
    total_tokens_out = 0
    total_cached_tokens = 0

    for scen in scenarios:
        sim = Simulator(scen)
        res = run(
            raw_input=scen.rendered_alert or f"Received: {scen.received_at}\nAlert on {scen.service}",
            client=client,
            world=sim,
            system_prompt=sop_prompt
        )

        hist, _ = compute_oracle_trajectory(scen)
        exp_conclude, _ = expected_next(scen.expected_extraction, hist)
        truth_dict = {
            "expected_extraction": scen.expected_extraction,
            "root_cause": exp_conclude.get("root_cause"),
            "runbook_id": exp_conclude.get("runbook_id"),
            "culprit": exp_conclude.get("culprit"),
            "log_signature": exp_conclude.get("log_signature"),
        }

        e2e_ok, report = validate_trace(res.trace, truth_dict)
        if e2e_ok:
            n_e2e_ok += 1

        # Apply escalation fallback if available and run failed
        system_ok = e2e_ok
        if not e2e_ok and fallback_client:
            escalated_res = maybe_escalate(
                result=res,
                raw_input=scen.rendered_alert or "",
                scenario=scen,
                teacher_client=fallback_client,
                sop_prompt=sop_prompt
            )
            esc_ok, _ = validate_trace(escalated_res.trace, truth_dict)
            if esc_ok:
                system_ok = True

        if system_ok:
            n_system_ok += 1

        # Sum tokens from trace usage
        trace_in = sum(entry.get("usage", {}).get("input_tokens", 0) for entry in res.trace)
        trace_out = sum(entry.get("usage", {}).get("output_tokens", 0) for entry in res.trace)
        trace_cached = sum(entry.get("usage", {}).get("cache_read_input_tokens", 0) for entry in res.trace)
        total_tokens_in += trace_in
        total_tokens_out += trace_out
        total_cached_tokens += trace_cached

        results.append({
            "id": scen.id,
            "status": res.status,
            "e2e_ok": e2e_ok,
            "system_ok": system_ok,
            "report": report,
            "tokens_in": trace_in,
            "tokens_out": trace_out,
            "cached_tokens": trace_cached
        })

    duration = time.time() - start_t
    n_scens = len(scenarios)
    summary = {
        "baseline_id": baseline_id,
        "split": split_name,
        "n_scenarios": n_scens,
        "e2e_ok_rate": round(n_e2e_ok / n_scens, 4) if n_scens > 0 else 0.0,
        "system_ok_rate": round(n_system_ok / n_scens, 4) if n_scens > 0 else 0.0,
        "avg_tokens_in": round(total_tokens_in / n_scens, 1) if n_scens > 0 else 0.0,
        "avg_tokens_out": round(total_tokens_out / n_scens, 1) if n_scens > 0 else 0.0,
        "avg_cached_tokens": round(total_cached_tokens / n_scens, 1) if n_scens > 0 else 0.0,
        "duration_sec": round(duration, 2),
    }

    out_file = BASELINES_DIR / f"{baseline_id}_{split_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"Baseline {baseline_id} on {split_name}: e2e_ok={summary['e2e_ok_rate']*100:.1f}%, system_ok={summary['system_ok_rate']*100:.1f}%")
    return summary


if __name__ == "__main__":
    split_name = "test"
    test_scens = []
    split_file = DATA_DIR / f"{split_name}.jsonl"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_scens.append(scenario_from_dict(json.loads(line)))
    client = FakeClient(response_fn=create_oracle_teacher_responder(test_scens[0]) if test_scens else None)
    run_baseline_on_split("Baseline_B", split_name, client, limit=10)
