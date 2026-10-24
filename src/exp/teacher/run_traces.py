from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from exp.config import load_config
from exp.data.split import scenario_from_dict
from exp.runtime.clients import FakeClient, GeminiClient
from exp.runtime.loop import run
from exp.validators import validate_trace
from exp.world.procedure import expected_next
from exp.world.scenarios import compute_oracle_trajectory
from exp.world.services import RUNBOOKS
from exp.world.simulator import Scenario, Simulator

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "data"
TRACES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "traces"
SOP_FILE = Path(__file__).resolve().parent.parent / "prompts" / "sop.md"


def get_oracle_conclude(scen: Scenario) -> Dict[str, Any]:
    hist, _ = compute_oracle_trajectory(scen)
    conclude_step, _ = expected_next(scen.expected_extraction, hist)
    return conclude_step


def create_oracle_teacher_responder(scen: Scenario):
    """Fallback high-speed oracle responder for teacher rollout testing."""
    ex = scen.expected_extraction

    def responder(messages: List[Dict[str, str]]) -> str:
        hist = []
        curr_assistant = None
        conclude_seen = False
        conclude_call = None
        last_decision = None

        for m in messages[2:]:
            if m["role"] == "assistant":
                curr_assistant = json.loads(m["content"])
                if curr_assistant.get("step") == "conclude":
                    conclude_seen = True
                    conclude_call = curr_assistant
            elif m["role"] == "user" and m["content"].startswith("TOOL_RESULT: "):
                if m["content"] == "TOOL_RESULT: continue":
                    continue
                res_dict = json.loads(m["content"][13:])
                if res_dict.get("tool") == "decision":
                    last_decision = res_dict
                elif curr_assistant and curr_assistant.get("step") in ("get_metrics", "get_recent_deploys", "query_logs", "get_dependencies", "check_dependency_health"):
                    if curr_assistant.get("step") == "query_logs":
                        res_dict["_sig"] = scen.log_signature
                    hist.append((curr_assistant, res_dict))

        if conclude_seen and last_decision and conclude_call:
            root_c = conclude_call["root_cause"]
            rb_id = conclude_call["runbook_id"]
            culprit_str = f" Culprit: {conclude_call['culprit']}." if conclude_call.get("culprit") else ""
            sev = last_decision["severity"]
            team = last_decision["team"]
            svc_str = scen.service or "unspecified"
            summary = f"{sev.upper()}: {svc_str} root cause {root_c}. Follow {rb_id}. Team {team}.{culprit_str}"
            return json.dumps({"step": "draft_summary", "text": summary})

        if len(messages) <= 2:
            return json.dumps(ex)
        else:
            next_act, _ = expected_next(ex, hist)
            return json.dumps(next_act)

    return responder


def run_single_trace(
    scen: Scenario,
    sop_prompt: str,
    client: Any,
    use_oracle_fallback: bool = False
) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    sim = Simulator(scen)
    target_client = FakeClient(response_fn=create_oracle_teacher_responder(scen)) if use_oracle_fallback else client

    res = run(
        raw_input=scen.rendered_alert or f"Received: {scen.received_at}\nAlert on {scen.service}",
        client=target_client,
        world=sim,
        system_prompt=sop_prompt,
    )

    exp_conclude = get_oracle_conclude(scen)
    truth_dict = {
        "expected_extraction": scen.expected_extraction,
        "root_cause": exp_conclude.get("root_cause"),
        "runbook_id": exp_conclude.get("runbook_id"),
        "culprit": exp_conclude.get("culprit"),
        "log_signature": exp_conclude.get("log_signature"),
    }

    all_ok, report = validate_trace(res.trace, truth_dict)
    return all_ok, report, res.trace, exp_conclude


def run_teacher_traces(
    use_oracle: bool = False,
    max_workers: int = 4,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    sop_prompt = SOP_FILE.read_text(encoding="utf-8")
    cfg = load_config()

    train_file = DATA_DIR / "train.jsonl"
    if not train_file.exists():
        from exp.data.split import generate_all_splits
        generate_all_splits(profile="smoke")

    scenarios: List[Scenario] = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(scenario_from_dict(json.loads(line)))

    if limit:
        scenarios = scenarios[:limit]

    client = GeminiClient(model=cfg.get("teacher", {}).get("model", "gemini-2.5-pro"))

    pass_traces: List[Dict[str, Any]] = []
    fail_traces: List[Dict[str, Any]] = []

    print(f"Running teacher traces on {len(scenarios)} scenarios (oracle={use_oracle})...")
    start_t = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_trace, scen, sop_prompt, client, use_oracle): scen
            for scen in scenarios
        }

        for fut in as_completed(futures):
            scen = futures[fut]
            try:
                ok, report, trace, exp_conclude = fut.result()
                entry = {
                    "id": scen.id,
                    "service": scen.service,
                    "root_cause": scen.root_cause,
                    "trace": trace,
                    "validation": report,
                }
                if ok:
                    pass_traces.append(entry)
                else:
                    fail_traces.append(entry)
            except Exception as e:
                fail_traces.append({
                    "id": scen.id,
                    "error": str(e),
                })

    duration = time.time() - start_t
    pass_file = TRACES_DIR / "train_pass.jsonl"
    fail_file = TRACES_DIR / "train_fail.jsonl"

    with open(pass_file, "w", encoding="utf-8") as f:
        for t in pass_traces:
            f.write(json.dumps(t) + "\n")

    with open(fail_file, "w", encoding="utf-8") as f:
        for t in fail_traces:
            f.write(json.dumps(t) + "\n")

    total = len(scenarios)
    n_pass = len(pass_traces)
    pass_rate = (n_pass / total) if total > 0 else 0.0

    summary = {
        "total": total,
        "pass_count": n_pass,
        "fail_count": len(fail_traces),
        "pass_rate": round(pass_rate, 4),
        "duration_sec": round(duration, 2),
    }

    with open(TRACES_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Teacher trace generation completed: {n_pass}/{total} passed ({pass_rate*100:.1f}%) in {duration:.1f}s")
    assert pass_rate >= 0.50, f"Teacher pass rate {pass_rate*100:.1f}% below 50% threshold!"

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run teacher traces")
    parser.add_argument("--oracle", action="store_true", help="Use oracle teacher responder")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke profile subset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios")
    parser.add_argument("--workers", type=int, default=None, help="Concurrency workers")
    args = parser.parse_args()

    cfg = load_config()
    profile = "smoke" if args.smoke or cfg.get("profile") == "smoke" else "full"
    
    trace_source = cfg.get("teacher", {}).get("trace_source", "teacher")
    use_oracle = args.oracle or (trace_source == "oracle")

    lim = args.limit
    if lim is None and profile == "smoke":
        lim = cfg.get("smoke", {}).get("n_train", 36)

    workers = args.workers or cfg.get("teacher", {}).get("concurrency", 8)
    run_teacher_traces(use_oracle=use_oracle, max_workers=workers, limit=lim)
