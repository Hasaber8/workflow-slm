import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional
from exp.config import load_config
from exp.data.log_bank import build_log_bank, render_scenario_logs
from exp.data.render import render_alert_templated
from exp.world.scenarios import generate_scenario
from exp.world.services import HELDOUT_SERVICES, TRAIN_SERVICES
from exp.world.simulator import Scenario

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "data"


def scenario_to_dict(scen: Scenario) -> Dict[str, Any]:
    return {
        "id": scen.id,
        "service": scen.service,
        "symptom": scen.symptom,
        "region": scen.region,
        "started_at": scen.started_at,
        "started_minutes_ago": scen.started_minutes_ago,
        "received_at": scen.received_at,
        "root_cause": scen.root_cause,
        "culprit": scen.culprit,
        "log_signature": scen.log_signature,
        "metrics_state": scen.metrics_state,
        "deploys_state": scen.deploys_state,
        "logs_state": scen.logs_state,
        "dependencies_state": scen.dependencies_state,
        "dependency_health_state": scen.dependency_health_state,
        "injected_errors": scen.injected_errors,
        "error_type": scen.error_type,
        "rendered_alert": scen.rendered_alert,
        "expected_extraction": scen.expected_extraction,
        "oracle_branch_labels": scen.oracle_branch_labels,
    }


def scenario_from_dict(d: Dict[str, Any]) -> Scenario:
    return Scenario(**d)


def build_split(
    name: str,
    count: int,
    styles: List[str],
    is_eval: bool,
    heldout_fraction: float,
    log_bank: Dict[str, List[str]],
    rng: random.Random,
    weights: Optional[Dict[str, int]] = None,
) -> List[Scenario]:
    scenarios: List[Scenario] = []
    seen_texts = set()

    for i in range(count):
        scen_id = f"scen_{name}_{i:05d}"
        scen = generate_scenario(
            scenario_id=scen_id,
            rng=rng,
            is_eval=is_eval,
            heldout_fraction=heldout_fraction,
            root_cause_weights=weights,
        )

        # Pick style and typo level
        style = rng.choice(styles)
        typo_level = rng.randint(0, 2)

        # Render alert
        rendered_text = render_alert_templated(
            service=scen.service,
            symptom=scen.symptom,
            region=scen.region,
            started_at=scen.started_at,
            started_minutes_ago=scen.started_minutes_ago,
            received_at=scen.received_at,
            style=style,
            typo_level=typo_level,
            rng=rng
        )

        # If duplicate text within split, retry or mutate
        if rendered_text in seen_texts:
            rendered_text += f" [id={i}]"
        seen_texts.add(rendered_text)
        scen.rendered_alert = rendered_text

        # Populate scenario log lines from log bank
        if scen.service:
            deps = scen.dependencies_state.get(scen.service, [])
            dep_name = scen.culprit if scen.culprit in deps else (deps[0] if deps else None)
            lines = render_scenario_logs(
                service=scen.service,
                log_signature=scen.log_signature,
                dependency=dep_name,
                log_bank=log_bank,
                rng=rng
            )
            scen.logs_state[scen.service] = lines

        scenarios.append(scen)

    return scenarios


def generate_all_splits(
    profile: str = "smoke",
    force: bool = False
) -> Dict[str, List[Scenario]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    if profile == "smoke":
        p_cfg = cfg["smoke"]
    else:
        p_cfg = cfg["data"]

    seed = cfg.get("seed", 1234)
    rng = random.Random(seed)
    log_bank = build_log_bank(force=force)

    train_styles = cfg["data"]["train_styles"]
    shift_style = [cfg["data"]["shift_style"]]
    heldout_fraction = cfg["data"]["heldout_service_fraction_eval"]
    weights = cfg["data"].get("root_cause_weights")

    splits_data = {}

    # 1. Train split
    splits_data["train"] = build_split(
        name="train",
        count=p_cfg["n_train"],
        styles=train_styles,
        is_eval=False,
        heldout_fraction=0.0,
        log_bank=log_bank,
        rng=rng,
        weights=weights
    )

    # 2. Dev split
    splits_data["dev"] = build_split(
        name="dev",
        count=p_cfg["n_dev"],
        styles=train_styles,
        is_eval=False,
        heldout_fraction=0.0,
        log_bank=log_bank,
        rng=rng,
        weights=weights
    )

    # 3. Test split
    splits_data["test"] = build_split(
        name="test",
        count=p_cfg["n_test"],
        styles=train_styles,
        is_eval=True,
        heldout_fraction=heldout_fraction,
        log_bank=log_bank,
        rng=rng,
        weights=weights
    )

    # 4. Shift split (uses shift_style only)
    splits_data["shift"] = build_split(
        name="shift",
        count=p_cfg["n_shift"],
        styles=shift_style,
        is_eval=True,
        heldout_fraction=heldout_fraction,
        log_bank=log_bank,
        rng=rng,
        weights=weights
    )

    # 5. Probe split
    splits_data["probe"] = build_split(
        name="probe",
        count=p_cfg["n_probe"],
        styles=train_styles,
        is_eval=True,
        heldout_fraction=heldout_fraction,
        log_bank=log_bank,
        rng=rng,
        weights=weights
    )

    # Leakage and isolation checks
    all_ids = set()
    all_texts = set()

    for s_name, scen_list in splits_data.items():
        for s in scen_list:
            assert s.id not in all_ids, f"Duplicate scenario ID across splits: {s.id}"
            assert s.rendered_alert not in all_texts, f"Duplicate rendered text across splits: {s.id}"
            all_ids.add(s.id)
            all_texts.add(s.rendered_alert)

            # Isolation checks for train & dev
            if s_name in ("train", "dev"):
                if s.service:
                    assert s.service not in HELDOUT_SERVICES, f"Held-out service {s.service} in {s_name}"
                for d in s.dependencies_state.get(s.service or "", []):
                    assert d not in HELDOUT_SERVICES, f"Held-out dependency {d} in {s_name}"
                for st in shift_style:
                    assert f"Handoff note" not in (s.rendered_alert or ""), f"Shift style in {s_name}"

        # Write out to jsonl
        out_file = DATA_DIR / f"{s_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in scen_list:
                f.write(json.dumps(scenario_to_dict(s)) + "\n")

    return splits_data


if __name__ == "__main__":
    import sys
    prof = "smoke" if "--smoke" in sys.argv else "full"
    splits = generate_all_splits(profile=prof, force=True)
    for k, v in splits.items():
        print(f"Generated {k}: {len(v)} scenarios")
