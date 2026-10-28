import json
from pathlib import Path
from typing import Any, Dict, List
from exp.config import load_config
from exp.runtime.loop import canonical_json

TRACES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "traces"
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "data"
SFT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "sft"

RARE_ROOT_CAUSES = {"credential_failure", "db_pool_exhausted", "unknown"}


def validate_sft_file(path: Path) -> None:
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            if any(line.strip() for line in f):
                return
    raise ValueError(
        f"Training dataset is missing or empty: {path}. "
        "Generate passing teacher traces, then run make sft before training."
    )


def build_sft_dataset(
    workflow_id: str = "incident_triage_v1",
    oversample_rare: bool = True
) -> List[Dict[str, Any]]:
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    pass_file = TRACES_DIR / "train_pass.jsonl"
    train_scens_file = DATA_DIR / "train.jsonl"

    if not pass_file.is_file():
        raise ValueError(f"Accepted teacher traces not found: {pass_file}. Run make traces successfully first.")

    scen_alerts = {}
    if train_scens_file.exists():
        with open(train_scens_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    scen_alerts[item["id"]] = item.get("rendered_alert")

    rows: List[Dict[str, Any]] = []
    with open(pass_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            scen_id = item["id"]
            root_cause = item.get("root_cause")
            trace = item["trace"]

            raw_input = scen_alerts.get(scen_id)
            if not raw_input:
                # Fallback to header format
                raw_input = f"Received: 2026-10-22T12:00:00Z\nAlert notification for {item.get('service', 'checkout-api')}"

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": workflow_id},
                {"role": "user", "content": raw_input}
            ]

            for entry in trace:
                step_obj = entry["step"]
                messages.append({"role": "assistant", "content": canonical_json(step_obj)})

                # Tool results or decision turn
                if "result" in entry:
                    pub = dict(entry["result"])
                    # Strip hidden annotations if present
                    pub = {k: v for k, v in pub.items() if not k.startswith("_") and k != "truth"}
                    messages.append({"role": "user", "content": f"TOOL_RESULT: {canonical_json(pub)}"})
                elif "decision" in entry:
                    dec = entry["decision"]
                    messages.append({"role": "user", "content": f"TOOL_RESULT: {canonical_json({'tool': 'decision', **dec})}"})
                elif step_obj.get("step") == "extract_alert":
                    messages.append({"role": "user", "content": "TOOL_RESULT: continue"})

            row = {"id": scen_id, "messages": messages}
            rows.append(row)

            # Repeat accepted traces from rare branches once if enabled
            if oversample_rare and root_cause in RARE_ROOT_CAUSES:
                rows.append({"id": f"{scen_id}_oversample", "messages": messages})

    if not rows:
        raise ValueError(
            f"No training examples found in {pass_file}. Run make traces successfully first; "
            "check artifacts/traces/train_fail.jsonl for errors. Existing SFT data was not overwritten."
        )

    out_file = SFT_DIR / "train.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"SFT dataset assembled: {len(rows)} training examples written to {out_file}")
    return rows


if __name__ == "__main__":
    build_sft_dataset()
