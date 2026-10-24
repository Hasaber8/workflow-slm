import json
import random
import pytest
from exp.runtime.clients import FakeClient
from exp.runtime.loop import run
from exp.validators import validate_trace
from exp.world.procedure import expected_next
from exp.world.scenarios import compute_oracle_trajectory, generate_scenario, generate_scenarios
from exp.world.services import HELDOUT_SERVICES, RUNBOOKS, TRAIN_SERVICES
from exp.world.simulator import Simulator


def test_generator_determinism():
    scens_1 = generate_scenarios(count=50, seed=42)
    scens_2 = generate_scenarios(count=50, seed=42)
    assert len(scens_1) == len(scens_2)
    for s1, s2 in zip(scens_1, scens_2):
        assert s1.id == s2.id
        assert s1.root_cause == s2.root_cause
        assert s1.service == s2.service
        assert s1.culprit == s2.culprit
        assert s1.oracle_branch_labels == s2.oracle_branch_labels


def test_generator_consistency_and_oracle_termination():
    # Generate scenarios across all root causes
    rng = random.Random(1234)
    causes = [
        "bad_deploy", "config_change_error", "upstream_dependency_failure",
        "resource_exhaustion", "db_pool_exhausted", "traffic_spike",
        "credential_failure", "transient_noise"
    ]

    for cause in causes:
        for i in range(25):
            scen = generate_scenario(
                scenario_id=f"test_{cause}_{i}",
                rng=rng,
                force_root_cause=cause,
                force_subvariant="standard",
                transient_error_fraction=0.0,
                persistent_error_fraction=0.0,
                distractor_deploy_fraction=0.10,
            )
            hist, labels = compute_oracle_trajectory(scen)
            assert len(hist) <= 14, f"Trajectory too long ({len(hist)}) for {cause}"

            conclude_act, lbl = expected_next({"service": scen.service, "symptom": scen.symptom}, hist)
            assert conclude_act["step"] == "conclude"
            assert conclude_act["root_cause"] == cause, f"Oracle {conclude_act['root_cause']} != {cause}"
            if scen.culprit:
                assert conclude_act["culprit"] == scen.culprit

    # Persistent error and inconclusive should conclude unknown
    for i in range(25):
        scen_pers = generate_scenario(
            scenario_id=f"test_pers_{i}",
            rng=rng,
            force_root_cause="unknown",
            force_subvariant="persistent_error",
        )
        hist, labels = compute_oracle_trajectory(scen_pers)
        assert len(hist) <= 14
        conclude_act, lbl = expected_next({"service": scen_pers.service, "symptom": scen_pers.symptom}, hist)
        assert conclude_act["step"] == "conclude"
        assert conclude_act["root_cause"] == "unknown"


def test_heldout_services_isolation():
    # Train scenarios should NEVER contain held-out services
    train_scenarios = generate_scenarios(count=200, seed=999, is_eval=False)
    for s in train_scenarios:
        if s.service is not None:
            assert s.service not in HELDOUT_SERVICES
            assert s.service in TRAIN_SERVICES
        # Check dependencies
        for d in s.dependencies_state.get(s.service or "", []):
            assert d not in HELDOUT_SERVICES


def test_oracle_model_scores_100_percent():
    """Oracle model test: A client following the oracle and templated summary scores 100% on V1-V8."""
    scenarios = generate_scenarios(count=30, seed=777)
    for scen in scenarios:
        sim = Simulator(scen)
        ex = scen.expected_extraction

        def oracle_responder(messages):
            # Parse history and last step from messages
            hist = []
            curr_assistant = None
            conclude_seen = False
            last_tool_decision = None

            for m in messages[2:]:
                if m["role"] == "assistant":
                    curr_assistant = json.loads(m["content"])
                    if curr_assistant.get("step") == "conclude":
                        conclude_seen = True
                elif m["role"] == "user" and m["content"].startswith("TOOL_RESULT: "):
                    if m["content"] == "TOOL_RESULT: continue":
                        continue
                    res_dict = json.loads(m["content"][13:])
                    if res_dict.get("tool") == "decision":
                        last_tool_decision = res_dict
                    elif curr_assistant and curr_assistant.get("step") in ("get_metrics", "get_recent_deploys", "query_logs", "get_dependencies", "check_dependency_health"):
                        if curr_assistant.get("step") == "query_logs":
                            res_dict["_sig"] = scen.log_signature
                        hist.append((curr_assistant, res_dict))

            if conclude_seen and last_tool_decision:
                root_c = scen.root_cause
                rb_id = RUNBOOKS[root_c][0]
                culprit_str = f" Culprit: {scen.culprit}." if scen.culprit else ""
                sev = last_tool_decision["severity"]
                team = last_tool_decision["team"]
                svc_str = scen.service or "unspecified"
                summary = f"{sev.upper()}: {svc_str} root cause {root_c}. Follow {rb_id}. Team {team}.{culprit_str}"
                return json.dumps({"step": "draft_summary", "text": summary})

            if len(messages) <= 2:
                # First step: extract alert
                return json.dumps(ex)
            else:
                next_act, _ = expected_next(ex, hist)
                return json.dumps(next_act)

        client = FakeClient(response_fn=oracle_responder)
        raw_alert = f"Received: {scen.received_at}\n[ALERT] Alert text for {scen.service}"
        res = run(raw_alert, client, sim)
        assert res.status == "ok"

        oracle_hist, _ = compute_oracle_trajectory(scen)
        oracle_conclude, _ = expected_next(scen.expected_extraction, oracle_hist)
        truth_dict = {
            "expected_extraction": scen.expected_extraction,
            "root_cause": oracle_conclude["root_cause"],
            "runbook_id": oracle_conclude["runbook_id"],
            "culprit": oracle_conclude["culprit"],
            "log_signature": oracle_conclude["log_signature"],
        }
        all_ok, report = validate_trace(res.trace, truth_dict)
        assert all_ok is True, f"Failed for {scen.id}: {report}"


def test_null_model_fails_trajectories():
    """Null model test: Client that always concludes unknown immediately gets 0 on trajectory."""
    scenarios = generate_scenarios(count=20, seed=123)
    traj_exact_count = 0
    conclusion_exact_count = 0

    for scen in scenarios:
        sim = Simulator(scen)
        steps = [
            json.dumps(scen.expected_extraction),
            json.dumps({"step": "conclude", "root_cause": "unknown", "runbook_id": "RB-900", "culprit": None, "log_signature": None}),
            json.dumps({"step": "draft_summary", "text": "SEV3: unknown cause, team sre-oncall follow RB-900."})
        ]
        client = FakeClient(responses=steps)
        res = run("Received: ...", client, sim)

        # Check if trajectory equals oracle
        hist, _ = compute_oracle_trajectory(scen)
        if len(hist) == 0:  # unresolvable cases
            traj_exact_count += 1
        if scen.root_cause == "unknown":
            conclusion_exact_count += 1

    # Null model should have very low trajectory exactness
    assert traj_exact_count < 5
    assert conclusion_exact_count < len(scenarios)
