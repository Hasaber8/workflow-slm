# workflow-slm: Compiling a Workflow into a Small Model

This project explores whether a small language model (SLM), running step-by-step under constrained decoding, can follow a branching, tool-driven procedure that lives directly in its weights, matching a large frontier model that carries the entire procedure in its prompt, at a fraction of the cost and latency.

Here's a detailed write-up on the same: [Compiling a Workflow into a 3B Model](https://arpitbhayani.me/blogs/compiling-an-sop-into-a-3b-model/).

TL;DR: Compiling a 3,500-token Standard Operating Procedure (Workflow) into a QLoRA-tuned 3B model hit 93.0% end-to-end accuracy (vs. 92.0% on Gemini 2.5 Pro) while cutting input tokens by 91.3% (5,200 → 450) and cost by 8.4x ($1.85 → $0.22 per 1k incidents).

```
End-to-End Success Rate (500 Test Scenarios — Higher is Better)
─────────────────────────────────────────────────────────────────────────────
Fine-Tuned 3B SLM (This Run)  [████████████████████████████░░]  93.0% (465/500)
Gemini 2.5 Pro (With Cache)   [███████████████████████████░░░]  92.0% (460/500)
Gemini 2.5 Pro (Raw API)      [███████████████████████████░░░]  90.0% (450/500)
Untuned Qwen 2.5 3B Base      [██████░░░░░░░░░░░░░░░░░░░░░░░░]  20.0% (100/500)

Average Input Tokens per Incident (Lower is Better)
─────────────────────────────────────────────────────────────────────────────
Prompt-Driven Setups (SOP)    [██████████████████████████████]  5,200 tokens
Fine-Tuned 3B SLM (Compiled)  [███░░░░░░░░░░░░░░░░░░░░░░░░░░░]    450 tokens (-91.3%)

Cost per 1,000 Incidents (Lower is Better)
─────────────────────────────────────────────────────────────────────────────
Gemini 2.5 Pro (Raw API)      [██████████████████████████████]  $3.50+
Gemini 2.5 Pro (With Cache)   [████████████████░░░░░░░░░░░░░░]  $1.85
Fine-Tuned 3B SLM (This Run)  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  $0.22  (8.4x cheaper)
```

---

## Key Experimental Results

| Model / Baseline | Runtime Prompt | End-to-End Success (`e2e_ok`) | Avg Input Tokens | Cost / 1k Incidents | Cost Ratio vs Baseline B |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline A** (Gemini 2.5 Pro, No Cache) | Full SOP (~3,500 tokens) | 90.0% | 5,200 | $3.50+ | 0.5x |
| **Baseline B** (Gemini 2.5 Pro, Cached) | Full SOP (~3,500 tokens) | 92.0% | 5,200 | $1.85 | 1.0x (ref) |
| **Baseline C** (Untuned Qwen 3B Base) | Full SOP (~3,500 tokens) | 20.0% | 5,200 | N/A (failing) | N/A |
| **Model D (Our Fine-Tuned SLM)** | **None (`"incident_triage_v1"`)** | **93.0%** | **450** | **$0.22** | **8.4x cheaper** |

* **91.3% Input Token Reduction:** Slashed input tokens from 5,200 down to 450 across the multi-turn triage loop.
* **8.4x Cost Reduction:** Runs at $0.22 per 1,000 incidents vs $1.85 with frontier prompt caching.
* **Zero Prompt Overhead:** Complete procedural logic and runbook mappings compiled directly into `Qwen2.5-3B-Instruct` QLoRA weights.

---

## What Workflow is Automated?

The workflow automates **production incident triage**. 

When an ambiguous, noisy alert fires (from PagerDuty, Slack, SMS, or an email thread), an on-call engineer normally executes a Standard Operating Procedure (SOP): checking metrics, checking recent deploys, reading raw log lines, inspecting upstream dependencies, choosing a remediation runbook, paging the right team, and drafting a status channel update.

Instead of pasting a [6,000+ token SOP](src/exp/prompts/sop.md) into a frontier model on every single turn, this system compiles the procedure directly into a compact model:

```text
[Messy Alert]
      │
      ▼
 1. extract_alert      -> Normalize service, symptom, and region
 2. get_metrics        -> Verify metrics (close false alarms early)
 3. get_recent_deploys -> Correlate recent code/config releases
 4. query_logs         -> Inspect crash stack traces in logs
 5. conclude & decide  -> Determine root cause, runbook, and severity
 6. draft_summary      -> Broadcast verified status update
      │
      ▼
[Resolved Incident]
```

---

## Quickstart

### Install Dependencies

Requires `make` and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Setup creates a Python 3.11 environment (or reuses `venv/`) and installs the
project plus the PyTorch, Transformers, PEFT, Accelerate, and bitsandbytes
dependencies used for local inference and QLoRA training:

```bash
make setup
source venv/bin/activate
```

The GPU packages require several GB of downloads. Local 4-bit inference and
training require a compatible NVIDIA GPU and driver; setup does not download
model weights or trained adapters.

For only the core/API dependencies and CPU-only tests, use `make setup EXTRAS=`.
For the optional vLLM server (`make serve`), use `make setup EXTRAS=local,serve`.
These commands also work when `venv/` already exists.

### Sign In to the Codex Teacher

The default teacher is **GPT-6.1 Terra** (`gpt-6.1-terra`) through the official
[Codex CLI](https://developers.openai.com/codex/cli/) (tested with 0.165.0).
Install the native CLI inside your Linux/WSL environment, then sign in with your
ChatGPT subscription during setup or separately:

```bash
make setup LOGIN=1
# Or, if dependencies are already installed:
make login
```

Login prints a device code and browser URL. Complete the sign-in yourself;
do not paste credentials or the code into chat. If device login is unavailable
for your account, use `make login LOGIN_ARGS=` for the normal browser flow.
`codex login status` checks the result. This uses subscription allowances and
requires model access on your plan; it is not unlimited or an OpenAI API credit.
Python setup does not install the Codex CLI itself.

### 1. Run the Full Test Suite
Runs the unit, setup, simulator, and oracle property tests (no network or GPU needed):
```bash
make test
```

Generator tests write to temporary directories, not your real scenario splits
or log bank. The separate `make data` and `make smoke` commands intentionally
write artifacts; do not run them alongside a teacher trace run.

### 2. Run the End-to-End Pipeline
Runs an offline simulator pipeline check using oracle traces and placeholder
adapter metadata. It does not train model weights or call hosted APIs:
```bash
make smoke
```

---

## Execution Modes & Giving New Inputs

The CLI supports three execution engines:
1. **Fine-Tuned SLM (`--model ep3`)**: The local Qwen 3B model executing the workflow compiled into its weights on your GPU (no API key, zero prompt overhead).
2. **Frontier Model (`--model gemini`)**: Live Google Gemini 2.5 Pro with the [3,500-token SOP prompt (`src/exp/prompts/sop.md`)](src/exp/prompts/sop.md) in its context.
3. **Reference Oracle (`--model oracle`, default)**: High-speed deterministic Python SOP engine for instant verification.

### A. Run with the Fine-Tuned Local Model (`--model ep3`)
Pass any alert string to run entirely on local GPU weights:

```bash
./venv/bin/python -m exp.runtime.cli --model ep3 "[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago"
```

What you will see:
1. **Turn 1 (`extract_alert`)**: Extracted `checkout-api`, `high_error_rate`, `us-east`, `started_minutes_ago: 12`.
2. **Turn 2 (`get_metrics`)**: Queried `error_rate` for `checkout-api` -> status `critical`.
3. **Turn 3 (`get_recent_deploys`)**: Checked releases -> deploy was 15m ago (correlated code deploy `d-7731`).
4. **Turn 4 (`query_logs`)**: Read raw log lines -> recognized `unhandled_exception`.
5. **Turn 5 (`conclude`)**: Identified root cause `bad_deploy`, culprit `d-7731`, runbook `RB-101`.
6. **Decision**: Deterministic code computed `sev1`, action `page_oncall_immediately`, assigned to `team-checkout`.
7. **Turn 6 (`draft_summary`)**: Generated the on-call channel summary citing only verified facts.

### B. Test with the Live Gemini API (`--model gemini`)
To run the live frontier model against any alert (carrying [`src/exp/prompts/sop.md`](src/exp/prompts/sop.md) in its prompt):
```bash
./venv/bin/python -m exp.runtime.cli --model gemini "payments service is super slow in us-west, started 10m ago"
```

### C. Replay Stored Scenarios
Replay any scenario generated in `artifacts/data/` using either the oracle or fine-tuned model:
```bash
# Using the fine-tuned model
./venv/bin/python -m exp.runtime.cli --model ep3 --scenario scen_test_00000

# Using the oracle
./venv/bin/python -m exp.runtime.cli --scenario scen_test_00000
```

### D. Test Fault Tolerance (Error Retries & Human Escalation)
See how the system behaves when diagnostic tools fail:

```bash
# Transient tool failure: Tool fails once, system retries and continues
./venv/bin/python -m exp.runtime.cli --scenario scen_test_00000 --inject-error get_metrics:1

# Persistent tool failure: Tool fails twice, system gives up and safely escalates to a human (RB-900)
./venv/bin/python -m exp.runtime.cli --scenario scen_test_00000 --inject-error get_metrics:2
```

---

## Model Fine-Tuning (QLoRA)

To compile the incident triage SOP directly into model weights:

First generate scenarios with `make data`, then choose a trace source:

- **Codex teacher (default):** sign in using `make login`, then run
  `make traces TRACE_ARGS="--limit 5"` to verify GPT-6.1 Terra access and output
  before the full batch. `make traces` targets all 4,000 training scenarios.
  It uses ten scenario workers by default; override with `--workers 1` for serial
  runs or another positive worker count. Each scenario still executes its turns
  sequentially. The teacher is remote, so local CPU/GPU utilization does not
  measure available inference capacity. A full run may exceed subscription allowances.
- **Gemini teacher (previous default):** export `GEMINI_API_KEY`, then run
  `make traces TRACE_ARGS="--provider gemini --model gemini-2.5-pro --limit 5"`.
  This uses the separately billed Gemini API. A `.env` file is not loaded automatically.
- **Deterministic oracle (no API):** run `make traces TRACE_ARGS=--oracle`.
  These are simulator/oracle demonstrations, not frontier-model teacher traces,
  so this is a different training-data source from the Gemini experiment.

Codex runs in a temporary directory with a read-only sandbox, tool features
disabled, and JSON-schema output. It requires ChatGPT authentication and does
not fall back to an API key. Set `teacher.model` in `config/config.yaml` or pass
`--model` to choose another model available to your subscription. User Codex
configuration/rules are not loaded for teacher completions.

Accepted traces are checkpointed after each scenario. Rerun the same command to
resume; increasing `--limit` reuses matching completed scenarios. Authentication,
quota, timeout, and CLI failures stop the batch instead of failing thousands of
scenarios: no new scenarios are scheduled after a Codex failure, but all already
submitted workers are drained and their successful traces checkpointed before
the batch exits. Run only one trace process per output directory.
Saved accepted traces are revalidated with the current validators on resume.
Existing accepted traces from a different teacher, dataset, SOP, or
Codex version are not mixed in: back them up first, then explicitly pass `--fresh`
to replace the trace files. Switching between Codex, Gemini, and oracle also
requires this explicit fresh run when accepted traces already exist.

Trace metadata records the teacher model/source. New Codex-trained results are
not the published Gemini-teacher experiment. Baseline API clients remain Gemini.

The SOP is a template: `exp.prompts.load_sop_prompt()` renders its service/region
reference from `src/exp/world/aliases.py`, the same catalog used by alert generation.
All runtime, teacher, and evaluation prompt loaders use this rendered SOP, and
resume fingerprints include the expanded alias reference. Alias guidance never
includes scenario labels or expected answers, and model extractions are not
silently repaired. Missing/ambiguous fields remain `null`. Summaries may only use
numbers from normalized extraction, tool results, or policy decisions—not
unverified numbers copied from the original alert.
Each trace run saves the exact expanded prompt as `artifacts/traces/sop_rendered.md`
and its SHA-256 in `summary.json` for comparison and reproducibility.

See [the Codex teacher pilot](docs/CODEX_PILOT.md) for the measured 100-scenario
result and known limitations. This is a teacher-data acceptance result, not a
fine-tuned model benchmark.

Only continue after traces succeed:

```bash
make sft && make train
```

Missing credentials and missing/empty training data cause an early error.

1. **Dataset Assembly**:
   4,000 teacher traces are converted into 5,116 multi-turn SFT conversation turns with rare-branch oversampling (`artifacts/sft/train.jsonl`). Gradients are computed strictly on assistant turns and the `<|im_end|>` stop token (inputs and tool results are masked to `-100`).

2. **Launch Training**:
   Trains `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4 with $r=16, \alpha=32$ on CUDA:
   ```bash
   ./venv/bin/python -m exp.train.train_lora --epochs 3 --batch-size 4 --grad-accum 8
   ```

3. **Checkpoints**:
   Adapters are saved at each epoch boundary:
   - `artifacts/adapters/triage_v1/epoch1/`
   - `artifacts/adapters/triage_v1/epoch2/`
   - `artifacts/adapters/triage_v1/epoch3/`

---

## How Cost Savings Are Measured

Cost efficiency is measured by tracking token usage across all turns of the incident triage interaction loop (`extract_alert` to `draft_summary`) and applying standard pricing models:

### 1. Token Reduction Across Multi-Turn Triage

Across 5 to 6 diagnostic turns, the runtime sums all input, output, and cached tokens:

| Metric per Incident | Baseline B (Gemini 2.5 Pro + SOP Prompt) | Model D (Our Fine-Tuned 3B SLM) | What Changed |
| :--- | :---: | :---: | :--- |
| **Input Tokens** | **5,200** | **450** | **91.3% reduction** (eliminated the 3,500-token SOP on every turn) |
| **Output Tokens** | **220** | **180** | Concise, validated JSON outputs |
| **Cached Tokens** | 4,800 | 0 | Not needed because the prompt is zero-overhead |
| **Total Tokens / Incident** | **5,420** | **630** | **88.4% fewer total tokens processed** |

### 2. Dollar Cost Calculation (Per 1,000 Incidents)

* **Baseline B (Frontier Model with Prompt Caching):**
  Even with prompt caching enabled (a 75% discount on the system prompt), processing 1,000 incidents requires 4.8M cached tokens plus 0.62M uncached and output tokens, totaling **$1.85 per 1,000 incidents**. (Without prompt caching, Baseline A exceeds $3.50+).

* **Model D (Our Fine-Tuned 3B SLM):**
  Because the procedure lives directly in model weights, 1,000 incidents consume only 630,000 total tokens (0.63M tokens). At standard small-model inference / GPU serving rates, the cost is **$0.22 per 1,000 incidents**.

* **Result:** **8.4x cheaper**, comfortably beating the pre-experiment success criterion of at least 5x lower cost.

### 3. Local Hardware Amortization

For fully air-gapped on-premise execution, local inference cost is evaluated by:
* Sampling GPU power draw every 500 ms via `nvidia-smi` to measure energy consumption in Watt-hours per 1,000 requests.
* Dividing host GPU hourly cost by the measured throughput (concurrency 16).

---

## Documentation & Deep Dives

- [Technical Approach (`docs/APPROACH.md`)](docs/APPROACH.md): Full end-to-end technical approach, state machine architecture, loss masking mechanics, and QLoRA pipeline.
- [Experiment Narrative (`EXPERIMENT.md`)](EXPERIMENT.md): The on-call story, turn-by-turn trace walkthrough, and sample training conversation.
- [Evaluation Report (`docs/REPORT.md`)](docs/REPORT.md): Benchmark results across Baselines A, B, C, and Model D, branch accuracies, and threshold checks.
- [Architectural Decisions (`docs/DECISIONS.md`)](docs/DECISIONS.md): Key design decisions, hardware adaptations (12GB VRAM), and schema adjustments.

---

## Makefile Reference

- `make setup`: Install core and local-model dependencies into `venv/`.
- `make setup LOGIN=1`: Install dependencies, then sign in to Codex.
- `make login`: Sign in to Codex using your ChatGPT account (device-code flow).
- `make test`: Run all unit, oracle, and structural tests.
- `make smoke`: Run the complete end-to-end smoke test pipeline.
- `make data`: Generate scenario splits and log banks.
- `make traces`: Run teacher traces against the scenarios.
- `make sft`: Assemble the training dataset from accepted traces.
- `make train`: Train QLoRA adapters with CUDA on local GPU.
- `make baselines`: Run baseline evaluations.
- `make eval`: Run free-run and teacher-forced evaluations.
- `make ablate`: Run all 4 ablation experiments.
- `make report`: Update `docs/REPORT.md`.

Teacher-forced evaluation reports extraction, tool calls, conclusions, and
summary validation separately in `phases`, as well as workflow branch accuracy.
Each step is scored against an oracle prefix independently of earlier model
mistakes. Summary text is checked for required facts, grounding, and length,
not exact string equality. Oracle evaluation creates a separate responder for
each scenario; compiled-model evaluation uses the workflow ID in both modes.
