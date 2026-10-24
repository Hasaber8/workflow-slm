# Compiling a Workflow into a Small Model: Incident Triage Experiment

Implementation spec, designed so Claude Code can build and run it end to end with minimal back and forth.

This version replaces the earlier expense intake spec. The harness (data generation, validators, training, serving, evaluation) does not depend on the domain. Only the world, schema, procedure, validators, and prompts changed.

## Purpose and success criteria

Test whether a LoRA-tuned small model, running one step at a time under constrained decoding, can follow a branching, tool-driven procedure that lives in its weights, and match a frontier model that carries the full procedure in its prompt, at a fraction of the cost per request.

The workflow is production incident triage. An alert arrives as messy text. The model investigates with tools, where each next action depends on what the previous tool returned, then concludes with a root cause and a runbook, and drafts a summary for the on-call channel.

Hypothesis: the tuned 4B model reaches within 3 points of the frontier-with-caching baseline on end-to-end accuracy, at 5x or lower cost per request, with no SOP in the runtime prompt.

Decision thresholds, fixed before any results exist:

- Free-run end-to-end accuracy (`e2e_ok`) within 3 points of baseline B on the test set, using a paired bootstrap on the same examples. A difference inside the 95 percent interval counts as a tie.
- Teacher-forced next-step accuracy of at least 95 percent on every branch type that has at least 30 test examples.
- Accuracy drop from the test set to the shifted test set under 8 points, or the escalation fallback closes the gap.
- Cost per 1000 requests at least 5x lower than baseline B, or token and energy figures that imply it once local GPU cost is filled in.
- Escalation rate under 10 percent on the test set.

## What this experiment can and cannot show

Read this before interpreting results.

- The tool results are structured and the branching rules are fixed, so the control flow is deterministic once the free text has been interpreted. The fuzzy parts are reading the messy alert, reading raw log lines to recognize the failure signature, and writing the summary. A plain code decision graph with a small model for only those three skills is a strong alternative design.
- This experiment therefore tests a narrower claim: whether the whole loop (tool choice, branching, retries, stopping) can live in the weights well enough to be worth it. Ablation 6 builds the code-orchestrated alternative so the comparison is direct.
- If the code-orchestrated variant matches the tuned model, the lesson is that the weights should hold only the fuzzy skills. That is a useful result, not a failed experiment.

## Kickoff prompt for Claude Code

Paste this into Claude Code from an empty repo directory that contains this file:

```
Read incident-triage-experiment spec (expense-intake-experiment.md) fully before writing any code. Implement and run the whole experiment as specified. Do not ask me questions. Resolve any ambiguity with the defaults in the document and record the choice in docs/DECISIONS.md. Work stage by stage, run the tests for a stage before starting the next, commit after each stage, and append every command and result to docs/RUNLOG.md. When all stages are done, generate docs/REPORT.md.
```

## Instructions for the implementing agent

Follow these rules for the whole run.

1. Read this document completely before writing code.
2. Make decisions without asking. When the document is silent, choose the simplest option that keeps the experiment valid.
3. Document why. For every non-trivial choice, including choices this document already made, add an entry to `docs/DECISIONS.md` using the template below. Explain the reason in your own words, not only the choice. Prefix any departure from this document with `DEVIATION:` and say why it was necessary.
4. Log as you go. Append to `docs/RUNLOG.md` for each command run: timestamp, command, duration, outcome, and key numbers.
5. Work in the stage order below. Do not start a stage until the previous stage's tests pass. If a stage fails twice, log the failure, skip every stage that depends on it, finish the independent stages, and still write the report.
6. Build and prove the simulator and the oracle (stage S1) before any API call or GPU use. Every later stage depends on them being correct.
7. Make every stage idempotent. Each stage writes to `artifacts/<stage>/` plus a `manifest.json` with inputs, config hash, seed, and counts. Skip a stage if its manifest matches, unless `FORCE=1`.
8. Fix all random seeds from `config.yaml`.
9. Cache all teacher API calls on disk, keyed by a hash of the full request. Reruns must cost nothing.
10. Enforce the spend guard. Stop API calls when `max_api_calls` or `max_api_spend_usd` is reached, write a note to the run log, and continue with the data collected so far.
11. Use synthetic data only. Never write real personal or production data. Read secrets from environment variables and never write them to files or logs.
12. Never train and serve at the same time. The GPU has 24 GB. Stop the vLLM server before training and restart it after.
13. Keep the code plain. No frameworks beyond those listed.

Stages:

| Stage | Name                                            |
| ----- | ----------------------------------------------- |
| S0    | Environment and GPU checks                      |
| S1    | World, simulator, oracle, schema, unit tests    |
| S2    | Log bank, scenario generation, alert rendering  |
| S3    | SOP prompt, validators, teacher traces          |
| S4    | Baselines A, B, C                               |
| S5    | SFT build and QLoRA training                    |
| S6    | Serving, evaluation of D, report                |
| S7    | Ablations                                       |

Decision log entry template:

```
## D-012: Title of the decision

- Date: 2026-10-23
- Stage: S2
- Context: what problem or ambiguity forced a choice
- Options considered: A, B, C
- Decision: what was chosen
- Why: the reasoning, including trade-offs
- Consequences: what this affects later
- Revisit if: the condition that would change the decision
```

## Environment

- Machine: RTX 4090 (24 GB), Ryzen 9, 128 GB RAM
- OS: Linux, or WSL2 Ubuntu with a recent NVIDIA driver. vLLM and Unsloth are far smoother on Linux than on native Windows.
- Python 3.11, one virtual environment at `.venv`
- Packages: `vllm`, `unsloth`, `trl`, `peft`, `bitsandbytes`, `transformers`, `datasets`, `pydantic`, `openai` (client for the local vLLM server), `anthropic` (teacher and baselines), `numpy`, `pytest`, `pyyaml`, `tqdm`, `tenacity`
- After installing, write exact versions to `requirements.lock` and record the install decisions in `docs/DECISIONS.md`.
- Environment variables: `ANTHROPIC_API_KEY` (required), `HF_TOKEN` (only if a model download needs it)
- Verify first: `nvidia-smi` shows the GPU, `torch.cuda.is_available()` is true, `bf16` works, and a 5-token vLLM generation succeeds. If this fails, stop and write the failure to the run log.

## Repository layout

```
incident-triage-lab/
  README.md
  Makefile
  pyproject.toml
  requirements.lock
  config/
    config.yaml
    pricing.yaml
  docs/
    DECISIONS.md
    RUNLOG.md
    REPORT.md              (generated last)
  src/exp/
    config.py
    schema.py              (Pydantic step models)
    world/
      services.py          (service catalog, dependency graph, teams, runbooks)
      simulator.py         (Scenario class and tool implementations)
      procedure.py         (expected_next(): the oracle)
      policy.py            (triage_decision(): severity, action, team)
      scenarios.py         (scenario generator)
    prompts/
      sop.md               (shared by teacher and baselines)
      render_alert.md      (prompt that renders alerts)
      log_bank.md          (prompt that builds the log line bank)
    data/
      log_bank.py
      render.py
      split.py
    runtime/
      loop.py
      structure.py         (structural legality checks)
      clients.py           (AnthropicClient, VllmClient with the same interface)
      escalate.py
      cli.py
    validators.py
    teacher/run_traces.py
    baselines/run_baseline.py
    train/build_sft.py
    train/train_lora.py
    serve/serve.sh
    eval/metrics.py
    eval/teacher_forced.py
    eval/bootstrap.py
    eval/run_eval.py
    eval/report.py
    util/cache.py
    util/log.py
  tests/
  artifacts/               (gitignored)
```

Makefile targets: `setup`, `test`, `smoke`, `world`, `data`, `traces`, `baselines`, `sft`, `train`, `serve`, `eval`, `ablate`, `report`, `all`.

## Configuration

Write `config/config.yaml` with these defaults:

```yaml
seed: 1234
profile: full            # full or smoke

teacher:
  model: claude-sonnet-5-5
  concurrency: 8
  max_retries: 5
  max_api_calls: 150000
  max_api_spend_usd: 300     # placeholder guard, adjust to your budget
  prompt_caching: true
  trace_source: teacher      # teacher, or oracle (ablation 3)

baseline:
  model: claude-sonnet-5-5   # baselines A and B use the same model, caching off and on

local:
  base_model: Qwen/Qwen3-4B
  served_name: base
  vllm_port: 8000
  max_model_len: 12288       # baseline C carries the full SOP prompt
  gpu_memory_utilization: 0.85
  enable_thinking: false

runtime:
  max_steps: 16

data:
  n_scenarios: 6000
  n_train: 4000
  n_dev: 400
  n_test: 500
  n_shift: 500
  n_probe: 100
  heldout_services: [shipping-svc, notification-svc]
  heldout_service_fraction_eval: 0.25   # share of test and shift scenarios on held-out services
  train_styles: [pagerduty_style, slack_message, email_notification, oncall_free_text, terse_sms]
  shift_style: handoff_note
  root_cause_weights:
    bad_deploy: 14
    config_change_error: 8
    upstream_dependency_failure: 18
    resource_exhaustion: 10
    db_pool_exhausted: 9
    traffic_spike: 9
    credential_failure: 6
    transient_noise: 12
    unknown: 14
  transient_error_fraction: 0.10     # one tool call fails once, then succeeds on retry
  persistent_error_fraction: 0.03    # one tool call fails twice
  distractor_deploy_fraction: 0.10   # a correlated deploy that is not the cause
  log_bank_lines_per_signature: 40
  log_bank_benign_lines: 60
  min_traces_per_branch: 100

train:
  max_seq_length: 2048
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.0
  learning_rate: 2.0e-4
  epochs: 3
  per_device_batch: 2
  grad_accum: 16
  warmup_ratio: 0.03
  load_in_4bit: true
  gradient_checkpointing: true

eval:
  latency_concurrency: 1
  throughput_concurrency: 16
  bootstrap_samples: 2000
  gpu_hourly_usd: null     # fill in to get amortized local cost
  kwh_price_usd: null      # fill in to get energy cost

smoke:
  n_scenarios: 60
  n_train: 36
  n_dev: 6
  n_test: 10
  n_shift: 6
  n_probe: 2
  train_max_steps: 20
```

Write `config/pricing.yaml` with every price set to `null`:

```yaml
anthropic:
  claude-sonnet-5-5:
    input_per_mtok: null
    output_per_mtok: null
    cache_write_per_mtok: null
    cache_read_per_mtok: null
```

Prices change, so do not guess them. When a price is `null`, the report shows token counts only and prints "pricing not set, fill config/pricing.yaml". The owner fills these in from the provider's pricing page.

## The workflow

### Overview

An alert arrives as text. The system investigates with tools and produces a triage decision and a summary.

| Step                    | Owner                         | What it does                                                                |
| ----------------------- | ----------------------------- | --------------------------------------------------------------------------- |
| `extract_alert`         | model                         | Read the messy alert and output structured fields                            |
| tool steps              | model emits, runtime executes | Model chooses the next tool call. Runtime runs it against the simulator.     |
| `conclude`              | model                         | Name the root cause, runbook, culprit, and the log signature it recognized   |
| decision                | plain code                    | `triage_decision()` computes severity, action, and team after `conclude`     |
| `draft_summary`         | model                         | Write the on-call channel summary using only facts from the trace            |

The model owns four fuzzy jobs: understanding the alert, choosing the next tool from the evidence so far, reading raw log lines to recognize the failure signature, and writing the summary. Severity, paging, and status flags (normal, elevated, critical, correlated) are computed in code and returned in tool results, so the model never does arithmetic.

### Input format

Every input starts with a header line, then the alert text:

```
Received: 2026-10-22T15:30:00Z
<messy alert text>
```

The alert may state its start time as an absolute UTC time or as a relative time ("about 12 min ago"). It may not use other time zones in v1. The model extracts the stated form and never converts between them.

### Services

Write `world/services.py`:

```python
SERVICES = {
  "checkout-api":       {"tier": 1, "team": "team-checkout",   "deps": ["auth-gateway", "payments-svc", "inventory-svc"]},
  "payments-svc":       {"tier": 1, "team": "team-payments",   "deps": ["auth-gateway", "orders-db-proxy"]},
  "auth-gateway":       {"tier": 1, "team": "team-identity",   "deps": ["user-profile"]},
  "web-frontend":       {"tier": 1, "team": "team-web",        "deps": ["checkout-api", "search-api", "cart-svc"]},
  "cart-svc":           {"tier": 2, "team": "team-checkout",   "deps": ["inventory-svc", "orders-db-proxy"]},
  "search-api":         {"tier": 2, "team": "team-discovery",  "deps": ["recommendation-svc"]},
  "inventory-svc":      {"tier": 2, "team": "team-fulfilment", "deps": ["orders-db-proxy"]},
  "user-profile":       {"tier": 2, "team": "team-identity",   "deps": []},
  "orders-db-proxy":    {"tier": 2, "team": "team-data",       "deps": []},
  "recommendation-svc": {"tier": 3, "team": "team-discovery",  "deps": []},
  "notification-svc":   {"tier": 3, "team": "team-comms",      "deps": ["user-profile"]},
  "shipping-svc":       {"tier": 3, "team": "team-fulfilment", "deps": ["inventory-svc", "notification-svc"]},
}
REGIONS = ["us-east", "us-west", "eu-west", "ap-south"]
RUNBOOKS = {
  "transient_noise": ("RB-001", "Close as noise"),
  "bad_deploy": ("RB-101", "Roll back the deploy"),
  "config_change_error": ("RB-102", "Revert the config change"),
  "upstream_dependency_failure": ("RB-201", "Escalate to the dependency owner"),
  "resource_exhaustion": ("RB-301", "Scale up capacity"),
  "db_pool_exhausted": ("RB-302", "Resize the connection pool"),
  "traffic_spike": ("RB-303", "Enable rate limiting"),
  "credential_failure": ("RB-401", "Rotate credentials"),
  "unknown": ("RB-900", "Escalate to a human"),
}
```

The held-out services (`shipping-svc`, `notification-svc`) never appear anywhere in train data, neither as the alert service nor as a dependency. Because no train service depends on them, the train world has 10 services and the held-out services appear only in test and shift scenarios.

### Tools and result formats

Tool results are compact JSON. Statuses and flags come from code. Window arguments are fixed.

| Step                       | Arguments                                   | Result                                                                                              |
| -------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `get_metrics`              | service, metric, `window_minutes` = 30      | `current`, `baseline`, `change_pct`, `status` (`normal` under 25 percent, `elevated` under 100, else `critical`) |
| `get_recent_deploys`       | service, `window_minutes` = 120             | `deploys` (id, kind `code` or `config`, version, `minutes_before_start`), `correlated`, `correlated_deploy_id`, `correlated_kind`. Correlated means a deploy within 45 minutes before the start. |
| `query_logs`               | service, `window_minutes` = 30              | `lines`: 5 raw log lines in free text, in time order                                                 |
| `get_dependencies`         | service                                     | `dependencies`: ordered list of up to 3 service names                                                |
| `check_dependency_health`  | dependency                                  | `health` (`healthy`, `degraded`, `down`), `incident_id` or null                                      |

Metric names: `error_rate`, `latency_p99`, `cpu`, `memory`, `request_rate`, `db_pool_utilization`.

Any tool call can return an error result instead: `{"tool": ..., "error": "timeout"}` with error types `timeout`, `rate_limited`, or `internal_error`.

The simulator keeps hidden annotations beside each public result, for example the true log signature for `query_logs`. The runtime shows the model only the public result. The oracle and validators use the annotations.

### Log signatures and the procedure

Log signatures: `unhandled_exception`, `upstream_timeout`, `connection_refused`, `oom_killed`, `cpu_throttled`, `db_pool_exhausted`, `auth_failure`, `config_parse_error`, `none`.

Symptoms and their primary metric:

```python
SYMPTOM_TO_METRIC = {
  "high_error_rate": "error_rate", "high_latency": "latency_p99", "high_cpu": "cpu",
  "high_memory": "memory", "db_pool_saturation": "db_pool_utilization",
  "traffic_surge": "request_rate", "service_down": "error_rate",
}
```

The canonical procedure, in words:

1. If the service or the symptom is null, conclude `unknown` with no tool calls.
2. Check the primary metric. If its status is `normal`, conclude `transient_noise`.
3. Get recent deploys, then query logs.
4. If a deploy is correlated and the logs match it (kind `code` with `unhandled_exception`, or kind `config` with `config_parse_error`), conclude `bad_deploy` or `config_change_error` with the deploy as culprit.
5. Otherwise branch on the log signature:
   - `upstream_timeout` or `connection_refused`: get the dependencies, then check each in order until one is not healthy, and conclude `upstream_dependency_failure` with that dependency as culprit. If all are healthy, conclude `unknown`.
   - `db_pool_exhausted`: confirm with `db_pool_utilization`. Not normal means `db_pool_exhausted`, else `unknown`.
   - `oom_killed`: confirm with `memory`. Not normal means `resource_exhaustion`, else `unknown`.
   - `cpu_throttled`: confirm with `cpu`. Not normal means `resource_exhaustion`, else `unknown`.
   - `none`: confirm with `request_rate`. Not normal means `traffic_spike`, else `unknown`.
   - `auth_failure`: conclude `credential_failure`.
   - `unhandled_exception` or `config_parse_error` without a matching deploy: conclude `unknown`.
6. A confirming metric already fetched earlier (for example when the primary metric is the confirming metric) is reused, not fetched again.
7. Error handling: if the last call returned an error, repeat the identical call once. If the repeat also errors, conclude `unknown`.

### Oracle

Write `world/procedure.py` exactly like this. Each `return` carries a branch label used by the evaluation. `history` is a list of `(call, result)` pairs. Results carry the hidden `_sig` annotation for logs.

```python
def call(step, **kw):
    return {"step": step, **kw}

def conclude(cause, culprit, sig):
    return {"step": "conclude", "root_cause": cause, "runbook_id": RUNBOOKS[cause][0],
            "culprit": culprit, "log_signature": sig}

def first_ok(ok, step, **kw):
    for c, r in ok:
        if c["step"] == step and all(c.get(k) == v for k, v in kw.items()):
            return r
    return None

def expected_next(ex, history):
    """Returns (action_dict, branch_label)."""
    svc, sym = ex["service"], ex["symptom"]
    if svc is None or sym is None:
        return conclude("unknown", None, None), "unresolvable_alert"
    if history:
        last_call, last_res = history[-1]
        if "error" in last_res:
            n = 0
            for c, r in reversed(history):
                if c == last_call and "error" in r:
                    n += 1
                else:
                    break
            if n >= 2:
                logs = first_ok([(c, r) for c, r in history if "error" not in r], "query_logs")
                return conclude("unknown", None, logs["_sig"] if logs else None), "error_giveup"
            return last_call, "error_retry"
    ok = [(c, r) for c, r in history if "error" not in r]
    primary = SYMPTOM_TO_METRIC[sym]
    m1 = first_ok(ok, "get_metrics", metric=primary)
    if m1 is None:
        return call("get_metrics", service=svc, metric=primary, window_minutes=30), "start_metrics"
    if m1["status"] == "normal":
        return conclude("transient_noise", None, None), "metrics_normal"
    dep = first_ok(ok, "get_recent_deploys")
    if dep is None:
        return call("get_recent_deploys", service=svc, window_minutes=120), "need_deploys"
    logs = first_ok(ok, "query_logs")
    if logs is None:
        return call("query_logs", service=svc, window_minutes=30), "need_logs"
    sig = logs["_sig"]
    if dep["correlated"]:
        if dep["correlated_kind"] == "config" and sig == "config_parse_error":
            return conclude("config_change_error", dep["correlated_deploy_id"], sig), "deploy_config_match"
        if dep["correlated_kind"] == "code" and sig == "unhandled_exception":
            return conclude("bad_deploy", dep["correlated_deploy_id"], sig), "deploy_code_match"
    if sig in ("upstream_timeout", "connection_refused"):
        deps = first_ok(ok, "get_dependencies")
        if deps is None:
            return call("get_dependencies", service=svc), "logs_upstream_need_deps"
        for d in deps["dependencies"]:
            h = first_ok(ok, "check_dependency_health", dependency=d)
            if h is None:
                return call("check_dependency_health", dependency=d), "dep_check_next"
            if h["health"] != "healthy":
                return conclude("upstream_dependency_failure", d, sig), "dep_unhealthy"
        return conclude("unknown", None, sig), "dep_all_healthy"
    confirm = {"db_pool_exhausted": ("db_pool_utilization", "db_pool_exhausted"),
               "oom_killed": ("memory", "resource_exhaustion"),
               "cpu_throttled": ("cpu", "resource_exhaustion"),
               "none": ("request_rate", "traffic_spike")}
    if sig in confirm:
        metric, cause = confirm[sig]
        m = first_ok(ok, "get_metrics", metric=metric)
        if m is None:
            return call("get_metrics", service=svc, metric=metric, window_minutes=30), "confirm_metric_" + sig
        if m["status"] != "normal":
            return conclude(cause, None, sig), "confirmed_" + sig
        return conclude("unknown", None, sig), "confirm_failed_" + sig
    if sig == "auth_failure":
        return conclude("credential_failure", None, sig), "logs_auth"
    return conclude("unknown", None, sig), "logs_no_match"
```

### Triage decision (plain code)

Write `world/policy.py` exactly like this. The runtime calls it after every `conclude` step and returns the result to the model as a tool message.

```python
SEV_TABLE = {(1, "critical"): "sev1", (1, "elevated"): "sev2",
             (2, "critical"): "sev2", (2, "elevated"): "sev3",
             (3, "critical"): "sev3", (3, "elevated"): "sev4"}
ACTIONS = {"sev1": "page_oncall_immediately", "sev2": "page_oncall",
           "sev3": "create_ticket", "sev4": "log_only"}

def triage_decision(service, primary_status, root_cause, culprit):
    title = RUNBOOKS[root_cause][1]
    if service is None:
        return {"severity": "sev3", "action": "create_ticket", "team": "sre-oncall", "runbook_title": title}
    tier = SERVICES[service]["tier"]
    status = primary_status if primary_status in ("elevated", "critical") else "elevated"
    sev = "sev4" if root_cause == "transient_noise" else SEV_TABLE[(tier, status)]
    if root_cause == "unknown":
        team = "sre-oncall"
    elif root_cause == "upstream_dependency_failure" and culprit in SERVICES:
        team = SERVICES[culprit]["team"]
    else:
        team = SERVICES[service]["team"]
    return {"severity": sev, "action": ACTIONS[sev], "team": team, "runbook_title": title}
```

`primary_status` is the status of the first successful primary-metric call, or `None` when that call never succeeded.

### Step schema

Write `src/exp/schema.py` exactly like this, and export `STEP_JSON_SCHEMA = TypeAdapter(Step).json_schema()`:

```python
from typing import Annotated, Literal, Union, Optional
from pydantic import BaseModel, Field, TypeAdapter

Service = Literal["checkout-api","payments-svc","auth-gateway","web-frontend","cart-svc","search-api",
                  "inventory-svc","user-profile","orders-db-proxy","recommendation-svc",
                  "notification-svc","shipping-svc"]
Region = Literal["us-east","us-west","eu-west","ap-south"]
Symptom = Literal["high_error_rate","high_latency","high_cpu","high_memory",
                  "db_pool_saturation","traffic_surge","service_down"]
Metric = Literal["error_rate","latency_p99","cpu","memory","request_rate","db_pool_utilization"]
RootCause = Literal["bad_deploy","config_change_error","upstream_dependency_failure","resource_exhaustion",
                    "db_pool_exhausted","traffic_spike","credential_failure","transient_noise","unknown"]
Runbook = Literal["RB-001","RB-101","RB-102","RB-201","RB-301","RB-302","RB-303","RB-401","RB-900"]
LogSig = Literal["unhandled_exception","upstream_timeout","connection_refused","oom_killed",
                 "cpu_throttled","db_pool_exhausted","auth_failure","config_parse_error","none"]

class ExtractAlert(BaseModel):
    step: Literal["extract_alert"]
    service: Optional[Service]
    symptom: Optional[Symptom]
    region: Optional[Region]
    started_at: Optional[str]            # absolute UTC, YYYY-MM-DDTHH:MM:SSZ, only if the alert states one
    started_minutes_ago: Optional[int]   # only if the alert states a relative time

class GetMetrics(BaseModel):
    step: Literal["get_metrics"]
    service: Service
    metric: Metric
    window_minutes: Literal[30]

class GetRecentDeploys(BaseModel):
    step: Literal["get_recent_deploys"]
    service: Service
    window_minutes: Literal[120]

class QueryLogs(BaseModel):
    step: Literal["query_logs"]
    service: Service
    window_minutes: Literal[30]

class GetDependencies(BaseModel):
    step: Literal["get_dependencies"]
    service: Service

class CheckDependencyHealth(BaseModel):
    step: Literal["check_dependency_health"]
    dependency: Service

class Conclude(BaseModel):
    step: Literal["conclude"]
    root_cause: RootCause
    runbook_id: Runbook
    culprit: Optional[str]               # deploy id or dependency name, else null
    log_signature: Optional[LogSig]      # the signature recognized in the logs, null if logs were never read

class DraftSummary(BaseModel):
    step: Literal["draft_summary"]
    text: str = Field(max_length=700)

Step = Annotated[Union[ExtractAlert, GetMetrics, GetRecentDeploys, QueryLogs, GetDependencies,
                       CheckDependencyHealth, Conclude, DraftSummary], Field(discriminator="step")]
```

If the vLLM structured-output backend rejects this schema, flatten it to a single object schema with an `enum` on `step` and optional fields, and record the change as a deviation.

### Runtime loop

Write `src/exp/runtime/loop.py` as plain Python. It must work with any client that implements `complete(messages, schema) -> (text, usage)`, so the same loop runs the teacher, the baselines, and the tuned model.

```python
def run(raw_input, client, world, workflow_id="incident_triage_v1", system_prompt=None, max_steps=16):
    sys_msg = system_prompt if system_prompt else workflow_id
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": raw_input}]
    state, extracted, trace, primary_status = "START", None, [], None
    for _ in range(max_steps):
        text, usage = client.complete(messages, STEP_JSON_SCHEMA)
        step = try_parse(text)
        if step is None:
            return Result(status="invalid_output", trace=trace)
        problem = structural_problem(state, step, extracted, trace)
        if problem:
            return Result(status="illegal:" + problem, trace=trace)
        messages.append({"role": "assistant", "content": canonical_json(step)})
        entry = {"step": step.model_dump(), "usage": usage}
        if step.step == "extract_alert":
            extracted = step
        elif step.step in TOOL_STEPS:
            public, truth = world.call(step)          # may return an error result
            entry.update(result=public, truth=truth)
            if primary_status is None and is_primary_metric_ok(step, public, extracted):
                primary_status = public["status"]
            messages.append({"role": "user", "content": "TOOL_RESULT: " + canonical_json(public)})
        elif step.step == "conclude":
            decision = triage_decision(extracted.service, primary_status, step.root_cause, step.culprit)
            entry.update(decision=decision)
            messages.append({"role": "user", "content": "TOOL_RESULT: " + canonical_json({"tool": "decision", **decision})})
        trace.append(entry)
        if step.step == "draft_summary":
            return Result(status="ok", trace=trace, extracted=extracted)
        state = step.step
    return Result(status="max_steps", trace=trace)
```

`structural_problem()` is the structural legality check that the runtime can enforce without the oracle:

- The first step must be `extract_alert`, and it may appear only once.
- `conclude` must follow at least one prior step, and `draft_summary` must immediately follow `conclude`. Nothing follows `draft_summary`.
- Tool steps must be a known type, their `service` must equal the extracted service, and `check_dependency_health.dependency` must be a service named in a prior `get_dependencies` result.
- A call that already succeeded may not be repeated. A call that just errored may be repeated, at most once.
- If the extracted service or symptom is null, the only legal next step is `conclude`.

`canonical_json` uses compact separators and the schema field order. Tool results go back as a `user` message prefixed with `TOOL_RESULT:`, because that works with every chat template without special tool-role handling.

Offline validators check the stricter requirement that every step equals the oracle's step. The runtime never calls the oracle.

`escalate.py`: when a run ends with any non-`ok` status, or the summary validator fails, rerun the whole input through the teacher client with the full SOP prompt and a fresh simulator episode. Record `escalated=true` and the reason.

## Data pipeline

### Stage D0: log bank

Call the teacher once per log signature to build a bank of realistic raw log lines, using `prompts/log_bank.md`. Request `log_bank_lines_per_signature` distinct lines per signature, in varied formats (plain text, key=value, JSON fragments, framework prefixes), with `{service}`, `{dependency}`, and `{n}` placeholders where relevant. Request `log_bank_benign_lines` benign lines (health checks, debug, routine warnings) as distractors. The signature `none` uses only benign lines.

Check every bank line in code: it has no words like "root cause", "the problem is", or runbook IDs, and the lines are unique. Save `artifacts/data/log_bank.json`.

A scenario's logs are 5 lines: 3 or 4 signature lines plus 1 or 2 benign lines, shuffled, with timestamps filled in by code. For signature `none`, all 5 lines are benign.

### Stage D1: scenario generation

Generate `n_scenarios` scenarios with the seed. Each scenario is a hidden world state plus its ground truth.

- Draw the root cause from `root_cause_weights`. The `unknown` weight splits into: inconclusive variants (7 points), unresolvable alerts (4 points), and persistent tool errors on an otherwise normal scenario (3 points).
- Draw the service, region, and symptom (consistent with the root cause), and the true start time. Received time is 2 to 40 minutes after the start. Held-out services follow `heldout_service_fraction_eval` in test and shift sets and never appear in train.
- Build consistent hidden state for each root cause:

| Root cause                    | Primary metric | Deploys                                    | Logs signature                         | Other state                                              |
| ----------------------------- | -------------- | ------------------------------------------ | -------------------------------------- | -------------------------------------------------------- |
| `bad_deploy`                  | elevated or critical | code deploy 3 to 40 min before start | `unhandled_exception`                  | Dependencies healthy                                     |
| `config_change_error`         | elevated or critical | config deploy 3 to 40 min before start | `config_parse_error`                  | Dependencies healthy                                     |
| `upstream_dependency_failure` | elevated or critical | none correlated                       | `upstream_timeout` or `connection_refused` | One dependency (position 1 to 3 in the list) degraded or down, with an incident ID, the rest healthy |
| `resource_exhaustion`         | elevated or critical | none correlated                       | `oom_killed` or `cpu_throttled`        | Matching metric (memory or cpu) elevated                 |
| `db_pool_exhausted`           | elevated or critical | none correlated                       | `db_pool_exhausted`                    | `db_pool_utilization` elevated                           |
| `traffic_spike`               | elevated or critical | none correlated                       | `none`                                 | `request_rate` elevated                                  |
| `credential_failure`          | elevated or critical | none correlated                       | `auth_failure`                         |                                                          |
| `transient_noise`             | normal         | any                                        | any                                    |                                                          |
| `unknown`, inconclusive       | elevated or critical | none correlated                       | one of: `upstream_timeout` with all dependencies healthy, `unhandled_exception` with no correlated deploy, `oom_killed` with normal memory, `none` with normal request rate | |

- Metrics not set by the table return normal values. Calls on any service return values consistent with the scenario state, so a wrong tool call still gets a plausible answer.
- Distractor deploys: for `distractor_deploy_fraction` of non-deploy causes, add a correlated deploy whose kind does not match the logs. The oracle ignores it.
- Error injection: choose a call from the error-free oracle trajectory. For `transient_error_fraction` of scenarios it fails once. For `persistent_error_fraction` it fails twice. Store the plan as `{call_key: failures}` so behavior does not depend on trajectory.
- Compute the oracle trajectory against the simulator, with its branch labels, and store it as ground truth with the expected extraction.

Assert for every non-inconclusive, non-error scenario that the oracle's conclusion equals the hidden root cause. This is the consistency test for the generator.

### Stage D2: render alerts

Call the teacher with `render_alert.md` to turn each scenario's facts into a messy alert. Write the prompt so that it:

- States the exact facts that must be recoverable: the service (rendered with an alias such as "the checkout service" or "co-api"), the symptom in natural words, the region (with aliases such as "virginia" or "us-east-1", or omitted), and the start time in the absolute or relative form from the truth.
- For unresolvable cases, names a service outside the catalog (for example "billing-worker") or gives no usable symptom, and states that ground truth is null for those fields.
- Requires a style from the list: `pagerduty_style`, `slack_message`, `email_notification`, `oncall_free_text`, `terse_sms`, and `handoff_note` (shift set only).
- Requires a typo level from 0 to 3 and optional noise (an unrelated stack trace excerpt, a signature block, dashboard links, a reply-all chain).
- Forbids mentioning the cause, the logs, deploys, dependencies, runbooks, or severity.

Prepend the `Received:` header in code. Deduplicate by a hash of normalized text and drop duplicates. Verify recoverability: the teacher extraction (stage S3) must agree with ground truth on most examples, and the data review in the acceptance checks looks at failures.

### Stage D3: splits

Split scenarios into train, dev, test, shift, and probe. Train and dev use only `train_styles` and non-held-out services. Test uses `train_styles` and holds `heldout_service_fraction_eval` of its scenarios on held-out services. Shift uses only `shift_style`. Probe holds 100 messages that contain two alerts for different services, or an alert plus an unrelated request. It has no accuracy target and measures how the system fails.

Assert no scenario ID or rendered text appears in two splits.

## Teacher traces and validators

### SOP prompt

Write `prompts/sop.md` once. Baselines A, B, and C and the teacher all use it. It contains:

- The workflow description and the full procedure from this document, as numbered rules
- The service list, symptom list, and the extraction rules below
- The tool schemas, argument values, and result formats, including that errors are retried once
- The log signature definitions, each with descriptions of how it typically appears
- The runbook table
- The output contract: one JSON object per turn matching the schema, no prose, no code fences
- The tool message format, including the decision message after `conclude`
- Summary rules: use only facts from the trace, state severity, root cause, culprit, runbook ID, action, and team, stay under 700 characters, be direct
- Four few-shot examples as full trajectories: one with a correlated deploy, one with a dependency check chain, one with an error retry, one noise case

Extraction rules:

- `service`: the catalog service the alert refers to, resolved from aliases. If the service is not in the catalog or is unclear, use null.
- `symptom`: one of the enum values. Map "5xx", "errors", and "failing" to `high_error_rate`, "slow" and "p99" to `high_latency`, and so on. If no symptom is stated, use null.
- `region`: one of the four regions, resolved from aliases, or null.
- `started_at`: only when the alert states an absolute UTC time. Otherwise null.
- `started_minutes_ago`: only when the alert states a relative time. Convert "an hour ago" to 60 and "half an hour" to 30. Otherwise null.

Target roughly 4000 to 7000 tokens. A realistic SOP size matters, because prompt caching has a minimum prompt length and the cost comparison depends on the real prompt size. Record the measured token count.

### Validators

Write each validator as a pure function in `validators.py`. Every accepted trace must pass all of them.

| ID | Name          | Check                                                                                                                       |
| -- | ------------- | --------------------------------------------------------------------------------------------------------------------------- |
| V1 | schema        | Every step parses against `Step`                                                                                            |
| V2 | trajectory    | Every non-conclude step after the extraction equals `expected_next(...)` for the actual history, including retries and arguments |
| V3 | extraction    | Fields equal ground truth. `started_at` and `started_minutes_ago` match the form that was rendered.                          |
| V4 | conclusion    | The `conclude` step equals the oracle's: `root_cause`, `runbook_id`, `culprit`, `log_signature`. Report each field separately. |
| V5 | decision      | The decision in the trace equals `triage_decision()` on the trace's facts                                                    |
| V6 | summary facts | Summary contains the severity, runbook ID, team, a root cause keyword, and the culprit when not null                          |
| V7 | summary safe  | Every ID, service name, and number in the summary appears in the extraction, tool results, or decision                        |
| V8 | summary length| Within schema limits and not empty                                                                                          |

Summary keyword map by root cause: `bad_deploy` needs "deploy", `config_change_error` needs "config", `upstream_dependency_failure` needs "dependency" or the culprit name, `resource_exhaustion` needs "resource", "memory", "cpu", or "capacity", `db_pool_exhausted` needs "pool", `traffic_spike` needs "traffic", `credential_failure` needs "credential" or "auth", `transient_noise` needs "noise", "transient", or "recovered", `unknown` needs "unknown", "inconclusive", or "escalat".

### Stage T1: run the teacher

Run the teacher through the real runtime loop against a fresh simulator episode for every train scenario. Use prompt caching on the system block. Keep traces that pass V1-V8. Save passing traces to `artifacts/traces/train_pass.jsonl` and failing traces with their failure reasons to `train_fail.jsonl`.

Gates:

- Stop and write a note if the overall pass rate is below 50 percent.
- Report the pass rate per style, per root cause, per branch label, and per validator. A low pass rate concentrated in one branch signals an SOP or oracle bug, so add a decision entry about the cause.
- Coverage: every branch label that appears in the oracle must have at least `min_traces_per_branch` accepted train traces. If a branch falls short, generate and run extra scenarios targeted at that branch (rare branches are persistent errors and the inconclusive variants), up to the spend guard.

When assembling SFT data, repeat accepted traces from rare branches once.

## Baselines

Run baselines on the test and shift sets only. All use the same runtime loop and the same `sop.md` as the system prompt.

| ID | Model                   | Prompt           | Notes                                                       |
| -- | ----------------------- | ---------------- | ----------------------------------------------------------- |
| A  | baseline.model (API)    | full SOP         | No caching                                                  |
| B  | baseline.model (API)    | full SOP         | Prompt caching on the system block                          |
| C  | Qwen3-4B untuned (vLLM) | full SOP         | Constrained decoding with the step schema                   |
| D  | Qwen3-4B + LoRA (vLLM)  | workflow ID only | The treatment. Constrained decoding.                        |

For API baselines, parse the model's JSON text. Count the first failure as `invalid_output` even if a retry succeeds. Allow one retry. Record usage fields: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`.

For B, check that `cache_read_input_tokens` is greater than zero on most calls after the first. If it is zero, the prompt is probably under the minimum cacheable length. Record this in the report, because baseline B would then be mislabeled.

## Training

Stage S5a builds the SFT data from passing traces. Write one JSONL row per trace: `{"id", "messages": [...]}`. Messages are the system turn (the workflow ID `incident_triage_v1` only), the user turn (raw alert with header), then alternating assistant steps and `TOOL_RESULT:` user turns, exactly as the runtime builds them. Strip hidden annotations (`_sig`, `truth`) from all messages. Use `canonical_json` for assistant turns.

Stage S5b trains with Unsloth QLoRA using the `train` config.

- Render with the tokenizer's chat template. For inference, the server must apply the same template with `enable_thinking=false`. Write a test that renders one training example and one runtime prompt for the same input and asserts the shared prefix is identical.
- Compute loss on assistant tokens only. With Unsloth, use `train_on_responses_only` with `instruction_part="<|im_start|>user\n"` and `response_part="<|im_start|>assistant\n"`.
- Disable packing. Assert every example fits in `max_seq_length`. If any do not, raise and record the count.
- Save the adapter after each epoch to `artifacts/adapters/triage_v1/epoch{1,2,3}`.
- Log training loss and tokens per second. Estimate time remaining in the run log.

## Serving

`serve/serve.sh`:

```bash
vllm serve Qwen/Qwen3-4B \
  --served-model-name base \
  --enable-lora --max-lora-rank 16 \
  --lora-modules ep1=artifacts/adapters/triage_v1/epoch1 ep2=artifacts/adapters/triage_v1/epoch2 ep3=artifacts/adapters/triage_v1/epoch3 \
  --max-model-len 12288 --gpu-memory-utilization 0.85 --port 8000
```

`VllmClient` calls the OpenAI-compatible endpoint with `response_format={"type": "json_schema", "json_schema": {"name": "step", "schema": STEP_JSON_SCHEMA}}`, temperature 0, and `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. Select the model by name (`base`, `ep1`, `ep2`, `ep3`). If the installed vLLM version uses a different structured-output parameter, adapt the client and record the change. Write a startup check that sends one request per model name and asserts schema-valid output.

Pick the best epoch by end-to-end accuracy on the dev set, then freeze that choice before touching the test set.

## Evaluation

`eval/run_eval.py` runs a client on a split in two modes, and `eval/report.py` aggregates.

Free-run mode runs the real loop against a fresh simulator episode per scenario. Each model step compounds on its own earlier outputs. This measures the system as deployed.

Teacher-forced mode (`eval/teacher_forced.py`) replays the oracle trajectory and, at each model-owned step, builds the messages from the oracle's history up to that point, asks the client for the next step, and compares it with the oracle's step. Errors do not compound, so this isolates decision quality by branch label. Run it for baselines B and C and for D, not for A.

Metrics per baseline and split:

- `e2e_ok`: V1-V8 pass and no escalation (raw model accuracy in free run)
- `system_ok`: V1-V8 pass after escalation applied (accuracy of the model plus the fallback)
- Per-step: `extract_exact`, `extract_field_acc` per field, `tool_selection_acc`, `conclusion_field_acc` (`root_cause`, `runbook_id`, `culprit`, `log_signature`), `summary_ok` (V6-V8)
- Trajectory: `traj_exact` (all tool steps equal the oracle), `over_call_rate` (extra calls), `under_call_rate` (concluded early), `redundant_call_rate` (repeated successful call), `steps_vs_oracle` (mean difference in step count)
- Error recovery: `retry_acc` on `error_retry` decision points, `giveup_acc` on `error_giveup`
- Teacher-forced: next-step accuracy by branch label and by step type
- Reliability: `invalid_output_rate`, `illegal_transition_rate`, `escalation_rate`
- Cost and speed: tokens in and out per request (split into cached and uncached for API runs), latency p50 and p95 at concurrency 1 (full request, all steps, tool time included), throughput in requests per second at concurrency 16, GPU energy per 1000 requests in Wh for local runs (from `nvidia-smi` power samples every 500 ms), and cost per 1000 requests when prices are filled in. API cost comes from `pricing.yaml`. Local cost comes from `gpu_hourly_usd` with the measured throughput, and from `kwh_price_usd` with the energy measure. Print "not set" for missing inputs.

Statistics:

- 95 percent confidence intervals by bootstrap for each accuracy
- Paired bootstrap for the difference between D and B on the same examples, with `bootstrap_samples` resamples
- Breakdowns by root cause, by branch label, by style, by held-out versus seen service, and by number of tool calls in the oracle trajectory (short versus long investigations)

`docs/REPORT.md` must contain: a results table for A, B, C, D on test and shift, the threshold check with pass or fail per criterion, a branch accuracy table from teacher-forced mode, the five most common failure types for D with three example traces each, an error-recovery section, the escalation analysis, the cost table, the held-out service result, an answer to "did the hypothesis hold", and the list of decisions that most affected the result with links to their `DECISIONS.md` entries.

## Testing strategy

Test at four levels. Run levels 1 and 2 before any API or GPU spend.

### Level 1: unit tests (`make test`, under 1 minute, no network, no GPU)

- Policy: table-driven cases for `triage_decision()` covering every tier and status pair, transient noise, unknown with and without a service, and the upstream team override.
- Simulator: tools return the documented result shapes, statuses cross the 25 and 100 percent thresholds correctly, the correlation window boundary (44, 45, and 46 minutes), and the hidden annotations never appear in public results.
- Error injection: a transient plan fails once then succeeds, a persistent plan fails twice, and plans are keyed by call so results do not depend on trajectory.
- Oracle: at least 40 table-driven history cases, one per branch label, including retry, giveup, reuse of an already fetched metric, and a dependency chain with the unhealthy dependency at position 1, 2, and 3.
- Oracle properties: over 5000 random scenarios, every trajectory terminates in at most 14 steps, never repeats a successful call, and only calls valid tools with valid arguments.
- Schema: valid and invalid JSON for every step type. Unknown `step` values, wrong window values, and extra fields fail.
- Structural legality: every allowed and disallowed state and step pair, including tool service mismatch, a dependency not in the last dependency list, and a repeated successful call.
- Validators V1-V8: one passing and at least two failing fixtures each.
- Runtime with a scripted fake client: happy path, clarification-free unresolvable path, error retry, persistent error, invalid JSON, illegal step, `max_steps`, and escalation triggering.
- Golden trace: the sample trace at the end of this document must replay through the runtime and validators with a fake client and pass.

### Level 2: oracle and null tests (no network, no GPU)

- Generator consistency: over 5000 scenarios per root cause, the oracle conclusion equals the hidden root cause for every non-inconclusive, non-error scenario, and equals `unknown` for the inconclusive and persistent-error ones.
- Oracle model test: a fake client that emits the oracle's steps and a templated summary for every dev scenario must score 100 percent on all metrics, in both free-run and teacher-forced modes. This proves the validators and runtime do not reject correct behavior.
- Null model test: a fake client that always concludes `unknown` immediately must score near the share of `unknown` scenarios on `conclusion` metrics and near 0 on trajectory metrics. This proves the metrics can fail.
- Leakage test: no scenario ID or rendered text is shared across splits, held-out services never appear in train (as alert service, dependency, or in any rendered text), and the shift style never appears in train or dev.
- Determinism test: running generation twice with the same seed gives identical scenarios and oracle trajectories.
- Template test: the training render and the runtime prompt share an identical prefix (see Training).
- Trajectory length report: print the distribution of oracle step counts. Assert the maximum and record the share of trajectories with 6 or more tool calls.

### Level 3: smoke test (`make smoke`, 15-30 minutes)

Run the full pipeline with the `smoke` profile: 60 scenarios, the real teacher API, a 20-step training run, serving, evaluation on 10 test scenarios in both modes, and the report. This catches integration bugs before the long run. It must pass before `make all` runs the full profile. Assert these outputs exist: a manifest per stage, a non-empty adapter, a metrics JSON, and a report with no unfilled placeholders.

### Level 4: acceptance checks on the real run

- Teacher gate: overall validator pass rate of at least 50 percent, and branch coverage met.
- Data quality review: print 30 random rendered alerts with their ground truth and read them. The agent records in the run log whether each is recoverable. Stop if more than 3 are not. Also print 20 rendered log sets and confirm the signature is identifiable by reading.
- Training sanity: loss decreases, and the adapter beats baseline C on the dev set. If it does not, check the chat template alignment test first.
- Constrained decoding sanity: `invalid_output_rate` for C and D is 0 on every split. If it is not, the structured-output setup is broken.
- Baseline B cache check: cache reads are greater than zero.
- Human spot check: the owner reads 20 failed traces from D and 20 passing summaries for tone and accuracy.

### How the owner can test it independently

After the run, a few manual checks the owner can do in 20 minutes:

1. Start the server with `make serve`, then run `python -m exp.runtime.cli --model ep3 "<alert text>"` with a hand-written alert. Read the full trace with the tool results.
2. Run `python -m exp.runtime.cli --scenario <id>` to replay a stored scenario and compare against its oracle trajectory.
3. Inject an error with `--inject-error get_metrics:1` and confirm the model retries, then with `get_metrics:2` and confirm it concludes `unknown`.
4. Change a threshold in code (for example the `elevated` cutoff from 25 to 40 percent, or one row of `SEV_TABLE`), rerun the same alerts, and confirm statuses and severity change with no retraining.
5. Change the dependency graph for one service in `services.py`, rerun an upstream scenario, and confirm the model follows the new dependency order.
6. Send an alert for a service name the model never saw, an alert with two services, and one in another language, and see whether the system escalates or fails. This shows where the tuned model's edge is.

## Ablations

Run these after the main result exists. Each is an optional stage controlled by `make ablate`.

1. Data size curve: train on 500, 1000, 2000, and all accepted traces. Plot dev and test `e2e_ok` and teacher-forced branch accuracy against size.
2. Model size: repeat the main training with Qwen3 1.7B and Qwen3 8B (QLoRA with batch 1 and accumulation 32 for 8B).
3. Trace source: train on oracle-synthesized traces (oracle steps, with teacher-written summaries that pass V6-V8) versus teacher-rollout traces. This measures how much the rejection sampling filter loses against perfect trajectories, and whether the teacher's mistakes are being copied.
4. Rare-branch oversampling on versus off.
5. Weights versus code changes: change one runbook ID in the mapping, and change one threshold in code. Measure how often the tuned model still emits the old runbook ID (expected: often, since it lives in the weights) and confirm the code change takes effect immediately. This quantifies the retraining cost the design carries.
6. Code-orchestrated variant: the oracle's decision graph runs in code, and the model is called only for the three fuzzy skills: `extract_alert`, a `classify_logs` call that returns the signature, and `draft_summary`. Build single-turn SFT examples for each skill from the same traces, train one adapter, and evaluate on the same test set with the same metrics. This is the direct test of whether the control flow needs to live in the weights.
7. Optional: escalate on low confidence. Capture token log probabilities on the extract and conclude steps and measure whether a minimum log probability predicts errors.

Each ablation writes its own result file and gets a short decision entry.

## Time estimate

Rough estimates, not measurements. API throughput and your rate limits are the largest source of variance.

| Stage                                                     | Claude Code build time | Machine run time | Notes                                                                   |
| --------------------------------------------------------- | ---------------------- | ---------------- | ----------------------------------------------------------------------- |
| S0. Environment and GPU checks                            | 20-40 min              | 10 min           | WSL2 and CUDA problems can add an hour                                  |
| S1. Services, simulator, oracle, policy, schema, tests    | 3-5 h                  | 2 min            | The largest build item. Levels 1 and 2 tests run here.                  |
| S2. Log bank, scenario generation, alert rendering        | 1.5-2.5 h              | 45-90 min        | About 6700 render calls at concurrency 8                                |
| S3. SOP, validators, teacher traces                       | 2-3 h                  | 2-4 h            | About 28000 teacher calls (4000 traces, 6-8 steps each), with caching   |
| S4. Baselines A, B, C                                     | 1-1.5 h                | 2-3 h            | 1000 scenarios each, about 7 calls per scenario for the API baselines   |
| S5. SFT build and QLoRA training (4B, 3 epochs)           | 1-1.5 h                | 1.5-3 h          | About 3000 traces, up to 2048 tokens each                               |
| S6. Serving, evaluation of D in both modes, report        | 1.5-2 h                | 1-1.5 h          | Includes latency, throughput, and energy runs                           |
| Smoke test (before the full run)                          | included               | 15-30 min        |                                                                         |
| Core total                                                | 10-16 h                | 8-14 h           | Overlaps, since Claude Code works while runs go                         |
| S7. Ablations (curve, 1.7B, 8B, trace source, code variant) | 2-3 h                | 10-16 h          | The 8B run is 4-5 h alone                                               |
| Review, decisions pass, owner testing                     | 2-3 h owner time       |                  |                                                                         |

Planning figures:

- Core experiment: about 2 to 3 calendar days with Claude Code running mostly unattended.
- With ablations: about 4 to 5 calendar days.
- Your active time: about 3 to 4 hours, mostly for the data quality review, the threshold check, and the manual tests.

## Definition of done

- `make test`, `make smoke`, and `make all` complete without errors.
- `docs/REPORT.md` exists, contains the A, B, C, D table with confidence intervals, the branch accuracy table, and states pass or fail for each threshold.
- `docs/DECISIONS.md` has an entry for every non-trivial choice, each with a stated reason. Every deviation is marked.
- `docs/RUNLOG.md` has every command with timing and outcome.
- All stage manifests exist, and rerunning `make all` finishes with every stage skipped.
- The repo has a `README.md` with the setup, the commands, and how to run the manual checks.

## Known pitfalls

- Qwen3 thinking mode: disable it at training and inference. A mismatch produces empty or stray think blocks.
- Chat template mismatch between training and serving is the most common cause of a tuned model that underperforms. Use the alignment test.
- Whitespace under constrained decoding can differ from the compact JSON used in training. Parse the output and do not compare strings.
- Union schemas can be unsupported by some structured-output backends. See the schema fallback.
- Prompt caching only applies above a minimum prompt length. Check the cache read counts.
- Do not run the vLLM server and training together. Free the GPU between them.
- A 4090 supports bf16. Use it for training compute and serving.
- Training in 4-bit and serving the base in bf16 is a standard setup but introduces a small mismatch. Record it, and test merging the adapter only if dev results look off.
- Free-text leakage: rendered alerts and log bank lines can reveal the answer through phrasing such as "this looks like a bad deploy". The prompts forbid it, and the data review should look for it. Leaked answers inflate every accuracy number.
- Hidden annotations (`_sig`, `truth`) must never reach any message the model sees, in training data or at runtime. Add a test that scans all messages for them.
- Teacher-forced and free-run numbers answer different questions. Report both, and do not average them.
- Spend: caching of teacher calls on disk is mandatory, so a crash does not repeat paid calls.

## Sample trace

Input as received by the tuned model, after the workflow ID system turn:

```
Received: 2026-10-22T15:30:00Z
[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago. lots of timeouts calling downstream. -- pagerduty bot
```

Step 1, model output:

```json
{"step":"extract_alert","service":"checkout-api","symptom":"high_error_rate","region":"us-east","started_at":null,"started_minutes_ago":12}
```

Step 2, model output:

```json
{"step":"get_metrics","service":"checkout-api","metric":"error_rate","window_minutes":30}
```

Tool message:

```
TOOL_RESULT: {"tool":"get_metrics","service":"checkout-api","metric":"error_rate","window_minutes":30,"current":0.18,"baseline":0.01,"change_pct":1700,"status":"critical"}
```

Step 3, model output:

```json
{"step":"get_recent_deploys","service":"checkout-api","window_minutes":120}
```

Tool message (a deploy 95 minutes before the start is outside the 45 minute correlation window, so code marks it not correlated):

```
TOOL_RESULT: {"tool":"get_recent_deploys","service":"checkout-api","deploys":[{"id":"d-7731","kind":"code","version":"v3.8.2","minutes_before_start":95}],"correlated":false,"correlated_deploy_id":null,"correlated_kind":null}
```

Step 4, model output:

```json
{"step":"query_logs","service":"checkout-api","window_minutes":30}
```

Tool message:

```
TOOL_RESULT: {"tool":"query_logs","service":"checkout-api","window_minutes":30,"lines":["15:21:04 WARN payments client: request to payments-svc exceeded 3000ms deadline (attempt 2/3)","15:21:09 ERROR checkout.handler: upstream call failed, returning 502 to client","15:21:11 WARN payments client: request to payments-svc exceeded 3000ms deadline (attempt 3/3)","15:21:15 INFO healthcheck ok","15:21:20 ERROR checkout.handler: upstream call failed, returning 502 to client"]}
```

The model reads the lines as the `upstream_timeout` signature, so the procedure moves to the dependency path.

Step 5, model output:

```json
{"step":"get_dependencies","service":"checkout-api"}
```

Tool message:

```
TOOL_RESULT: {"tool":"get_dependencies","service":"checkout-api","dependencies":["auth-gateway","payments-svc","inventory-svc"]}
```

Step 6, model output:

```json
{"step":"check_dependency_health","dependency":"auth-gateway"}
```

Tool message (an injected transient error):

```
TOOL_RESULT: {"tool":"check_dependency_health","dependency":"auth-gateway","error":"timeout"}
```

Step 7, model output (the identical call, retried once):

```json
{"step":"check_dependency_health","dependency":"auth-gateway"}
```

Tool message:

```
TOOL_RESULT: {"tool":"check_dependency_health","dependency":"auth-gateway","health":"healthy","incident_id":null}
```

Step 8, model output:

```json
{"step":"check_dependency_health","dependency":"payments-svc"}
```

Tool message:

```
TOOL_RESULT: {"tool":"check_dependency_health","dependency":"payments-svc","health":"down","incident_id":"INC-4471"}
```

Step 9, model output:

```json
{"step":"conclude","root_cause":"upstream_dependency_failure","runbook_id":"RB-201","culprit":"payments-svc","log_signature":"upstream_timeout"}
```

Runtime decision message, computed in code (checkout-api is tier 1 and the primary status is critical, so severity is sev1, and the team is the culprit's owner):

```
TOOL_RESULT: {"tool":"decision","severity":"sev1","action":"page_oncall_immediately","team":"team-payments","runbook_title":"Escalate to the dependency owner"}
```

Step 10, model output:

```json
{"step":"draft_summary","text":"SEV1: checkout-api error rate is critical in us-east (started about 12 min ago). Cause: upstream dependency failure, payments-svc is down (INC-4471). Follow RB-201 (escalate to the dependency owner). Paging team-payments immediately."}
```

Validation of this trace:

- V1: every step parses.
- V2: each tool step equals the oracle's step for the history so far, including the identical retry after the injected error and the dependency check order.
- V3: all extracted fields equal ground truth, with `started_minutes_ago` matching the rendered relative form.
- V4: the conclusion equals the oracle's, including the culprit `payments-svc` and the signature `upstream_timeout`.
- V5: the decision equals `triage_decision()` on the trace's facts.
- V6: the summary contains the severity, runbook ID, team, a root cause keyword, and the culprit.
- V7: every ID, service name, and number in the summary appears in the extraction, tool results, or decision.
- V8: the summary is within the length limit.

Branch labels visited by the oracle on this trace: `start_metrics`, `need_deploys`, `need_logs`, `logs_upstream_need_deps`, `dep_check_next`, `error_retry`, `dep_check_next`, `dep_unhealthy`.
