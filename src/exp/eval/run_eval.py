import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
from exp.prompts import load_sop_prompt
from exp.baselines.run_baseline import run_baseline_on_split
from exp.data.split import scenario_from_dict
from exp.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff
from exp.eval.metrics import compute_metrics
from exp.eval.report import generate_report
from exp.eval.teacher_forced import evaluate_teacher_forced
from exp.runtime.clients import FakeClient, GeminiClient, VllmClient
from exp.teacher.run_traces import create_oracle_teacher_responder

EVAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "eval"
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "data"
SOP_FILE = Path(__file__).resolve().parent.parent / "prompts" / "sop.md"


def run_full_evaluation(model_name: str = "oracle", split_name: str = "test", limit: int = 10):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sop_prompt = load_sop_prompt(SOP_FILE)
    split_file = DATA_DIR / f"{split_name}.jsonl"

    scenarios = []
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    scenarios.append(scenario_from_dict(json.loads(line)))

    if limit:
        scenarios = scenarios[:limit]
    if not scenarios:
        raise ValueError(f"No scenarios found in {split_file}; generate data before evaluating.")

    print(f"Running evaluation of {model_name} on {split_name} ({len(scenarios)} scenarios)...")

    # Client selection
    client_factory = None
    if model_name == "oracle":
        client = None
        client_factory = lambda scen: FakeClient(response_fn=create_oracle_teacher_responder(scen))
    elif model_name == "gemini":
        client = GeminiClient(model="gemini-2.5-pro")
    else:
        client = VllmClient(model=model_name)

    system_prompt = sop_prompt if model_name != "Model_D" else "incident_triage_v1"

    # 1. Free-run mode
    summary = run_baseline_on_split(
        baseline_id=model_name,
        split_name=split_name,
        client=client,
        system_prompt=system_prompt,
        limit=limit,
        client_factory=client_factory,
    )

    # 2. Teacher-forced mode
    tf_summary = evaluate_teacher_forced(
        client=client,
        scenarios=scenarios,
        system_prompt=system_prompt,
        client_factory=client_factory,
    )

    eval_out = {
        "model": model_name,
        "split": split_name,
        "free_run": summary,
        "teacher_forced": tf_summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    out_file = EVAL_DIR / f"{model_name}_{split_name}_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2)

    print(f"Evaluation complete. Results saved to {out_file}")
    return eval_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="oracle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    run_full_evaluation(model_name=args.model, split_name=args.split, limit=args.limit)
