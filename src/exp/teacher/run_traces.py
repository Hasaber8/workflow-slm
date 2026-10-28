from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from exp.config import load_config
from exp.prompts import load_sop_prompt
from exp.data.split import scenario_from_dict
from exp.data.split import scenario_to_dict
from exp.runtime.clients import FakeClient, GeminiClient
from exp.runtime.codex import CodexClient, CodexError
from exp.runtime.loop import run
from exp.validators import validate_trace
from exp.world.procedure import expected_next
from exp.world.scenarios import compute_oracle_trajectory
from exp.world.services import RUNBOOKS
from exp.world.simulator import Scenario, Simulator
from exp.util.cache import DiskCache

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
    max_workers: Optional[int] = None,
    limit: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    cfg = load_config()
    teacher_cfg = cfg.get("teacher", {})
    provider = "oracle" if use_oracle else (provider or teacher_cfg.get("provider", "gemini"))
    if model is None and provider == teacher_cfg.get("provider", "gemini"):
        model = teacher_cfg.get("model")
    max_workers = max_workers if max_workers is not None else teacher_cfg.get("concurrency", 1)
    if max_workers < 1 or (limit is not None and limit < 1):
        raise ValueError("Workers and scenario limit must be positive.")
    client = None
    if provider == "codex":
        client = CodexClient(
            model=model,
            command=teacher_cfg.get("codex_command", "codex"),
            timeout=teacher_cfg.get("timeout_seconds", 300),
        )
        client.validate_setup()
    elif provider == "gemini":
        model = model or "gemini-2.5-pro"
        client = GeminiClient(model=model)
        if not client.api_key:
            raise ValueError(
                "Gemini teacher traces require GEMINI_API_KEY. Export it before running "
                "make traces (API charges apply), or explicitly use "
                "make traces TRACE_ARGS=--oracle for deterministic, non-LLM traces."
            )
    elif provider != "oracle":
        raise ValueError(f"Unsupported teacher provider: {provider}. Choose codex, gemini, or --oracle.")
    else:
        model = None

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    sop_prompt = load_sop_prompt(SOP_FILE)

    train_file = DATA_DIR / "train.jsonl"
    if not train_file.exists():
        from exp.data.split import generate_all_splits
        generate_all_splits(profile="smoke")

    scenarios: List[Scenario] = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(scenario_from_dict(json.loads(line)))

    teacher = {"provider": provider, "model": model}
    if provider == "codex":
        teacher["cli_version"] = client.version
    fingerprints = {
        scen.id: DiskCache.compute_hash(scenario_to_dict(scen), sop_prompt, json.dumps(teacher, sort_keys=True))
        for scen in scenarios
    }
    scenarios_by_id = {scen.id: scen for scen in scenarios}
    scenarios = scenarios[:limit] if limit is not None else scenarios
    if not scenarios:
        raise ValueError("No training scenarios found. Run make data first.")

    pass_file = TRACES_DIR / "train_pass.jsonl"
    fail_file = TRACES_DIR / "train_fail.jsonl"
    saved = {}
    if resume and pass_file.exists():
        with open(pass_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if (entry.get("teacher") != teacher or entry.get("fingerprint") is None or
                        entry["fingerprint"] != fingerprints.get(entry["id"])):
                    raise ValueError(
                        "Existing accepted traces use a different teacher, SOP, or dataset. "
                        "Back them up, then use --fresh to explicitly replace them."
                    )
                scen = scenarios_by_id[entry["id"]]
                conclusion = get_oracle_conclude(scen)
                ok, report = validate_trace(entry.get("trace", []), {
                    "expected_extraction": scen.expected_extraction, **conclusion,
                })
                if not ok:
                    raise ValueError(
                        f"Saved accepted trace {scen.id} fails current validation. "
                        "Preserve it and inspect the failed checks before resuming; "
                        f"do not silently reuse it: {report}"
                    )
                entry["validation"] = report
                saved[entry["id"]] = entry

    pass_traces = [saved[s.id] for s in scenarios if s.id in saved]
    pending = [s for s in scenarios if s.id not in saved]
    resumed_count = len(pass_traces)
    fail_traces: List[Dict[str, Any]] = []

    print(f"Running teacher traces on {len(scenarios)} scenarios "
          f"(provider={provider}, model={model or 'CLI default'}, resumed={resumed_count})...", flush=True)
    start_t = time.time()
    fatal_error = None

    (TRACES_DIR / "sop_rendered.md").write_text(sop_prompt, encoding="utf-8")
    with open(pass_file, "a" if resume else "w", encoding="utf-8") as accepted, \
            open(fail_file, "w", encoding="utf-8") as rejected, \
            ThreadPoolExecutor(max_workers=max_workers) as executor:
        remaining = iter(pending)
        futures = {
            executor.submit(run_single_trace, scen, sop_prompt, client, provider == "oracle"): scen
            for scen in islice(remaining, max_workers)
        }

        while futures:
            fut = next(as_completed(futures))
            scen = futures.pop(fut)
            try:
                ok, report, trace, exp_conclude = fut.result()
                entry = {
                    "id": scen.id,
                    "service": scen.service,
                    "root_cause": scen.root_cause,
                    "trace": trace,
                    "validation": report,
                    "teacher": teacher,
                    "fingerprint": fingerprints[scen.id],
                }
                if ok:
                    pass_traces.append(entry)
                    accepted.write(json.dumps(entry) + "\n")
                    accepted.flush()
                else:
                    fail_traces.append(entry)
                    rejected.write(json.dumps(entry) + "\n")
                    rejected.flush()
            except Exception as e:
                entry = {
                    "id": scen.id,
                    "error": str(e),
                    "teacher": teacher,
                }
                fail_traces.append(entry)
                rejected.write(json.dumps(entry) + "\n")
                rejected.flush()
                if isinstance(e, CodexError):
                    # Stop scheduling, but drain every already-submitted result.
                    # Cancelling/breaking here loses other workers' completed traces.
                    fatal_error = fatal_error or str(e)
            print(f"  {len(pass_traces) + len(fail_traces)}/{len(scenarios)} completed; "
                  f"{len(pass_traces)} passed", flush=True)
            next_scenario = next(remaining, None) if fatal_error is None else None
            if next_scenario is not None:
                future = executor.submit(run_single_trace, next_scenario, sop_prompt, client, provider == "oracle")
                futures[future] = next_scenario

    duration = time.time() - start_t

    total = len(scenarios)
    n_pass = len(pass_traces)
    pass_rate = (n_pass / total) if total > 0 else 0.0

    summary = {
        "total": total,
        "pass_count": n_pass,
        "fail_count": len(fail_traces),
        "pass_rate": round(pass_rate, 4),
        "duration_sec": round(duration, 2),
        "teacher": teacher,
        "resumed_count": resumed_count,
        "incomplete": fatal_error is not None,
        "sop_sha256": hashlib.sha256(sop_prompt.encode("utf-8")).hexdigest(),
    }

    with open(TRACES_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Teacher trace generation completed: {n_pass}/{total} passed ({pass_rate*100:.1f}%) in {duration:.1f}s")
    if fatal_error:
        raise CodexError(f"Teacher batch stopped: {fatal_error} Accepted traces are saved; rerun to resume.")
    if pass_rate < 0.50:
        detail = next((t["error"] for t in fail_traces if "error" in t), "Trace validation failed.")
        raise ValueError(
            f"Teacher pass rate {pass_rate*100:.1f}% below 50% threshold! "
            f"Example failure: {detail} See {fail_file}. Do not continue to SFT or training."
        )

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run teacher traces")
    parser.add_argument("--oracle", action="store_true", help="Use oracle teacher responder")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke profile subset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios")
    parser.add_argument("--workers", type=int, default=None, help="Concurrency workers")
    parser.add_argument("--provider", choices=["codex", "gemini"], default=None, help="Teacher backend")
    parser.add_argument("--model", default=None, help="Model name (otherwise use provider/config default)")
    parser.add_argument("--fresh", action="store_true", help="Replace existing trace files instead of resuming")
    args = parser.parse_args()

    cfg = load_config()
    profile = "smoke" if args.smoke or cfg.get("profile") == "smoke" else "full"
    
    trace_source = cfg.get("teacher", {}).get("trace_source", "teacher")
    use_oracle = args.oracle or (trace_source == "oracle")

    lim = args.limit
    if lim is None and profile == "smoke":
        lim = cfg.get("smoke", {}).get("n_train", 36)

    run_teacher_traces(
        use_oracle=use_oracle, max_workers=args.workers, limit=lim,
        provider=args.provider, model=args.model,
        resume=not args.fresh and cfg.get("teacher", {}).get("resume", True),
    )
