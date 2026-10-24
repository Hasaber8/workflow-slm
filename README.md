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

### 1. Run the Full Test Suite
Runs all 65 unit, simulator, and oracle property tests (~0.15s, no network or GPU needed):
```bash
make test
```

### 2. Run the End-to-End Pipeline
Executes data generation, teacher traces, SFT assembly, adapter training, baseline evaluation, and report generation:
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
