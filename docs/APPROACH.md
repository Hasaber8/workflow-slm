# Technical Approach: Compiling an Incident Triage SOP into Model Weights

This document describes the exact technical approach and engineering pipeline used in this experiment. It is written as a factual reference for anyone analyzing or writing about the project.

---

## 1. Core Concept and Motivation

A Standard Operating Procedure (SOP) in software operations is essentially a state machine. When an ambiguous alert fires, an engineer follows a branching sequence:
1. Normalize and extract entities from the alert.
2. Query monitoring metrics to verify whether the issue is real or transient noise.
3. Check deployment history within a correlation window (45 minutes).
4. Query recent log lines to identify specific failure signatures (exceptions, OOM, timeouts).
5. Conclude the root cause, culprit version/commit, and remediation runbook ID.
6. Calculate severity and paging action via deterministic business rules.
7. Broadcast a verified summary to team channels.

### The Problem with Prompting
Traditionally, executing this workflow with large language models requires pasting a comprehensive 3,500+ token SOP manual into the system prompt on every turn. In a multi-turn diagnostic loop (5 to 6 turns), this means sending 5,000+ tokens back and forth per incident. For off-the-shelf small language models (3B-4B parameters), this long procedural context overwhelms the model, resulting in frequent hallucinations, step skipping, and schema syntax errors (20.0% end-to-end success in our baseline).

### The Hypothesis
Instead of carrying the procedure in the context window, can we compile the entire branching decision tree, tool schema constraints, and runbook knowledge directly into the weights of a 3-billion-parameter model using QLoRA fine-tuning?

---

## 2. End-to-End Pipeline Overview

```text
[Synthetic Scenario Generator]
             │
             ▼
    [World Simulator] ◄───► [Oracle / Teacher Generator]
                                      │
                                      ▼ (Validated via V1-V8)
                             [Gold Traces (JSONL)]
                                      │
                                      ▼
                             [SFT Dataset Builder]
                          (Assistant Loss Masking)
                                      │
                                      ▼
                          [QLoRA 4-bit Training]
                        (Qwen2.5-3B-Instruct)
                                      │
                                      ▼
                              [Adapter Weights]
                                (ep1, ep2, ep3)
                                      │
                                      ▼
                        [Interactive Runtime Loop]
                  (LocalHfClient / Gemini / VllmClient)
                                      │
                                      ▼
                        [Invariant Evaluation Suite]
```

---

## 3. Detailed Stage-by-Stage Implementation

### Stage 1: World Simulator and Procedural Rules
* **Module:** `src/exp/world/simulator.py`, `src/exp/world/procedure.py`
* **What it does:** Implements a deterministic operational environment. It manages:
  * 12 microservices with distinct baseline metric profiles and region assignments.
  * 9 root causes (`bad_deploy`, `config_change_error`, `upstream_dependency_failure`, `resource_exhaustion`, `db_pool_exhausted`, `traffic_spike`, `credential_failure`, `transient_noise`, and `unknown`).
  * Simulated diagnostic tools:
    * `get_metrics`: Returns current vs baseline telemetry (`error_rate`, `latency_p99`, `cpu`, `memory`, `request_rate`, `db_pool_utilization`).
    * `get_recent_deploys`: Returns code and config releases within 120 minutes, marking deployments within 45 minutes of incident onset as correlated.
    * `query_logs`: Returns simulated log lines with realistic stack traces and error messages matching the injected root cause.
* **Error Injection:** Supports transient tool errors (retrying on failure) and persistent tool errors (triggering escalation to human on-call via runbook `RB-900`).

### Stage 2: Synthetic Data and Style Distribution Shift
* **Module:** `src/exp/data/generator.py`, `src/exp/data/split.py`
* **Dataset Splits:** Generates 6,000 scenarios:
  * `train`: 4,000 scenarios
  * `dev`: 400 scenarios
  * `test`: 500 scenarios
  * `shift`: 500 scenarios
  * `probe`: 100 scenarios
* **Style Generalization:** To prevent overfitting to a single prompt format, raw alerts are rendered across 5 distinct styles:
  * `pagerduty_style`
  * `slack_message`
  * `email_notification`
  * `oncall_free_text`
  * `terse_sms`
* **Held-Out Generalization (`shift` split):** Evaluated against a completely unseen input style (`handoff_note`) and two held-out services (`shipping-svc`, `notification-svc`) comprising 25% of the evaluation split.

### Stage 3: Teacher Trace Generation and Validation
* **Module:** `src/exp/teacher/run_traces.py`
* **Trace Engine:** Can run against either:
  1. The reference Python oracle (`FakeClient`), which computes deterministic next steps via state-machine logic.
  2. The frontier model (`GeminiClient` using `gemini-2.5-pro` with full SOP prompt).
* **Validation Filter:** Traces are only accepted into the training set if they pass all 8 formal invariants (`V1` to `V8`). Failed traces are discarded.

### Stage 4: SFT Dataset Construction with Loss Masking
* **Module:** `src/exp/train/build_sft.py`, `src/exp/train/train_lora.py`
* **Output:** 5,116 multi-turn conversation trajectories stored in `artifacts/sft/train.jsonl`.
* **ChatML Format:** Formatted with standard tokens: `<|im_start|>system`, `<|im_start|>user`, and `<|im_start|>assistant`.
* **Loss Masking Strategy:**
  * During causal language model training, standard fine-tuning often computes loss across all tokens, causing the model to memorize user inputs and tool responses.
  * In our custom `SFTDataset` collator, tokens corresponding to `<|im_start|>system`, the user prompt, and the `TOOL_RESULT` strings are set to label `-100`.
  * Gradients are computed strictly on assistant turns and the `<|im_end|>` token:
    $$\mathcal{L} = -\sum_{t \in \text{assistant tokens}} \log P(x_t \mid x_{<t})$$
  * This forces the model to focus purely on predicting the correct next diagnostic action given the evolving evidence.

### Stage 5: Parameter-Efficient Fine-Tuning (QLoRA)
* **Module:** `src/exp/train/train_lora.py`
* **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
* **Quantization:** 4-bit NormalFloat4 (NF4) via `bitsandbytes`, with double quantization enabled and `bfloat16` compute dtype.
* **LoRA Hyperparameters:**
  * Rank ($r$): 16
  * Alpha ($\alpha$): 32
  * Dropout: 0.0
  * Target modules: All linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  * Trainable parameters: ~115 MB adapter file.
* **Training Setup:**
  * Learning rate: $2.0 \times 10^{-4}$ with linear warmup and cosine decay.
  * Effective batch size: 32 (per-device batch size 2, gradient accumulation 16).
  * Checkpoints: Adapters saved per epoch (`epoch1`, `epoch2`, `epoch3`).
  * Hardware constraint: Optimized for 12 GB VRAM cards (RTX 3060) by setting `max_seq_length: 2048` and `max_model_len: 4096`.

### Stage 6: Interactive Runtime Loop
* **Module:** `src/exp/runtime/loop.py`, `src/exp/runtime/cli.py`
* **Execution Flow:**
  1. Initializes the `Simulator` with the scenario state.
  2. Ingests raw alert string with system prompt set to `"incident_triage_v1"` (zero SOP instructions).
  3. Iterates up to 16 turns. On each turn:
     * Dispatches context to the client (`LocalHfClient` for local GPU weights, `GeminiClient` for API, or `FakeClient` for oracle).
     * Extracts and validates the generated JSON action.
     * Executes the corresponding tool in the `Simulator`.
     * If the step is `conclude`, runs deterministic code to compute severity and assignment.
     * Appends `TOOL_RESULT` to the conversation history and continues.
  4. Stops when `draft_summary` is produced or maximum steps are reached.

### Stage 7: Invariant Validation Suite
* **Module:** `src/exp/validators.py`
* Every execution trace is verified against 8 strict invariants:
  * **V1:** Transition order conforms to SOP (e.g., metric check must precede deploy lookup).
  * **V2:** Response conforms to strict JSON schema.
  * **V3:** Service, symptom, and region match the original alert text.
  * **V4:** Concluded root cause matches simulation ground truth.
  * **V5:** Runbook correctly remediates the root cause (`RB-101` for `bad_deploy`, `RB-102` for config error, etc.).
  * **V6:** Identified culprit exists in the simulated deployment history (no hallucinated commit IDs).
  * **V7:** Log signature matches actual lines returned by the log tool.
  * **V8:** Channel summary references only facts proven during previous turns.

---

## 4. Key Architectural Decisions and Deviations

During development, several practical adaptations were made (documented in `docs/DECISIONS.md`):

1. **Local Model Selection (`Qwen/Qwen2.5-3B-Instruct`):**
   * Selected for strong structured instruction-following and lightweight footprint.
   * Total adapter weights take 115 MB on disk, allowing fast loading and low VRAM footprint.

2. **Unified Schema Flattening (`UnifiedStep`):**
   * Discriminated unions (`oneOf`) and integer literals (`Literal[30]`) caused structured output parser failures in the Google GenAI backend.
   * Following the fallback path in `doc.md` (line 544), schemas were flattened into a single object (`UnifiedStep`) with optional fields, stripping null keys before strict Pydantic validation.

3. **VRAM Optimization (12 GB Budget):**
   * Configured `max_model_len: 4096` and `gpu_memory_utilization: 0.80`. Since multi-turn triage traces rarely exceed 1,500 to 2,000 tokens, 4,096 tokens provided plenty of headroom without risking CUDA Out-of-Memory (OOM).

---

## 5. Summary of Experimental Results

| Metric | Baseline B (Gemini 2.5 Pro + Full SOP) | Baseline C (Untuned Qwen 3B + Full SOP) | Model D (Our Fine-Tuned SLM, No Prompt) |
| :--- | :---: | :---: | :---: |
| **System Prompt** | ~3,500-token manual | ~3,500-token manual | `"incident_triage_v1"` (1 token) |
| **Average Input Tokens / Turn** | 5,200 | 5,200 | 450 (91.3% reduction) |
| **End-to-End Success Rate (`e2e_ok`)** | 92.0% | 20.0% | 93.0% |
| **System Integrity (`system_ok`)** | 92.0% | 50.0% | 97.0% |
| **Cost per 1,000 Incidents** | $1.85 (prompt cached) | N/A (unreliable) | $0.22 (8.4x cheaper) |
| **Execution Environment** | External API | Local / External | Fully local GPU (NF4 QLoRA) |

### Factual Conclusion
The experiment verified that a small 3B language model can reliably execute a multi-step, branching Standard Operating Procedure without any procedural text in its prompt. The knowledge of when to query metrics, how to correlate deploys, and how to pick runbooks was successfully embedded into the model weights, achieving higher reliability than prompting an untuned model and matching frontier-level accuracy at an 8.4x lower cost.
