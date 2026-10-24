# Incident Triage with Workflow SLM: The Experiment Story

When you are on call at 3:00 AM and your phone buzzes with an alert, your heart sinks. The notification is almost always messy, fragmented, and stressful:

> `"[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago"`

In that moment of cognitive overload, an on-call engineer has to do several things at once: fight the panic, verify whether the alert is even real or just transient noise, trace recent git deploys, sift through hundreds of cryptic log lines, identify the exact runbook to execute, notify the right engineering team, and draft an executive status update.

This experiment investigates a simple, human-centric question:

> **Can we teach a small, lightweight model to know this entire on-call playbook by heart, so it handles the investigation instantly on local hardware without needing to re-read a giant procedural manual on every single turn?**

---

## 1. What Was the Input and What Was the Output?

To understand how the workflow functions, let's step through the journey from the raw emergency ping to the calm resolution.

### The Input: An Ambiguous Alert Ping
The input is just what an engineer receives in Slack or PagerDuty:
```text
"[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago"
```
Notice what is *missing* here:
* It doesn't use the canonical service name (`checkout-api`).
* It uses human slang (`"5xx rate way above normal"`) instead of the standardized metric name (`error_rate`).
* It doesn't tell you whether anyone deployed code, what broke, or who owns the service.

### The Output: A Verified, Calming Resolution
Rather than a rambling chatbot essay, the system's output is structured, actionable, and verified:
* **Root Cause Found:** `bad_deploy`
* **Specific Culprit:** `d-7731` (version `v3.8.2` rolled out 15 min prior)
* **Runbook to Follow:** `RB-101` (*"Roll back the deploy"*)
* **Escalation & Ownership:** Severity `sev1`, action `page_oncall_immediately`, assigned to `team-checkout`.
* **Team Broadcast:** Ready-to-send Slack summary containing zero hallucinations and citing only confirmed facts:
  > *"SEV1: checkout-api root cause bad_deploy. Follow RB-101. Team team-checkout. Culprit: d-7731."*

---

## 2. What Changed? "With Prompt" vs. "Without Prompt"

Normally, if you want an AI to follow a Standard Operating Procedure (SOP), you paste the entire rulebook into the prompt window. Here is how that compares to our fine-tuned model:

```
┌────────────────────────────────────────────────────────────────────────┐
│ WITH PROMPT (Frontier Model e.g. Gemini 2.5 Pro)                       │
│                                                                        │
│  [3,500-token SOP Document] ──► Re-read on every single message turn   │
│  + [Raw Alert String]                                                  │
│  + [Accumulated Diagnostic Output]                                     │
│                                                                        │
│  The Reality: Slow, costly, and resource-heavy (~5,200 tokens / turn). │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ WITHOUT PROMPT / OUR MODEL (Fine-Tuned Qwen-2.5-3B)                    │
│                                                                        │
│  System: "incident_triage_v1" (Zero SOP manual attached!)              │
│  + [Raw Alert String]                                                  │
│  + [Accumulated Diagnostic Output]                                     │
│                                                                        │
│  The Reality: The procedure lives inside the model weights.            │
│               Blazing fast, ~450 tokens / turn, runs entirely locally. │
└────────────────────────────────────────────────────────────────────────┘
```

### The Three Approaches Compared

1. **Frontier Model With Prompt (`Baseline_B`, Gemini 2.5 Pro):**
   * **The Setup:** On every turn, we feed it the comprehensive [3,500-token SOP manual (`src/exp/prompts/sop.md`)](src/exp/prompts/sop.md).
   * **The Experience:** It works well (**92.0% e2e accuracy**), but you pay the latency and financial penalty of moving ~5,200 tokens back and forth across network APIs for every minor step.

2. **Untuned Small Model With Prompt (`Baseline_C`, Off-the-shelf Qwen 3B):**
   * **The Setup:** We give the same [3,500-token SOP (`src/exp/prompts/sop.md`)](src/exp/prompts/sop.md) to a raw, untuned 3-billion-parameter model.
   * **The Experience:** It struggles under cognitive load. It suffers from context fatigue, skipping steps, hallucinating non-existent deploy IDs, and stumbling with an unacceptable **20.0% e2e accuracy**.

3. **Our Fine-Tuned Model (`Model_D`, Qwen 2.5-3B + QLoRA):**
   * **The Setup:** We strip away the 3,500-token manual completely. The system prompt is just a 1-token identifier: `"incident_triage_v1"`.
   * **The Experience:** Because the SOP was compiled directly into its weights during training, it instinctively knows what question to ask next. It reaches **93.0% e2e accuracy**, slashes input tokens by **91%**, and runs **8.4x cheaper** on consumer GPUs without any external API calls.

---

## 3. How Was the Model Trained? (What the Training Data Looked Like)

To compile the entire incident response procedure into model weights, we created a Supervised Fine-Tuning (SFT) dataset containing **5,116 multi-turn conversations** derived from validated incident triage traces.

* The full training dataset lives at [`artifacts/sft/train.jsonl`](artifacts/sft/train.jsonl) (~13 MB).
* A clean standalone sample scenario is available at [`docs/sample_training_conversation.json`](docs/sample_training_conversation.json).

### The Training Conversation Format

Each training example is structured in standard ChatML multi-turn format (`system`, `user`, `assistant`).

> **How the model learns to think:**
> During QLoRA training on `Qwen2.5-3B-Instruct`, we compute loss **strictly on the `assistant` turns and the `<|im_end|>` stop token**. All `system` prompts, initial `user` alerts, and incoming `TOOL_RESULT` payloads are **masked with label `-100`**.
>
> This guarantees the model does not memorize hardcoded tool answers. Instead, it learns the *reasoning transition*: **"Given this evolving history of diagnostic results, what is the exact next step I must take?"**

Here is a representative sample record ([`docs/sample_training_conversation.json`](docs/sample_training_conversation.json)):

```json
{
  "id": "scen_train_01765",
  "messages": [
    {
      "role": "system",
      "content": "incident_triage_v1"
    },
    {
      "role": "user",
      "content": "Received: 2026-10-23T03:28:00Z\nALERT: user profile svc latency spike in virginia starting at 2026-10-23T03:03:00Z"
    },
    {
      "role": "assistant",
      "content": "{\"step\":\"extract_alert\",\"service\":\"user-profile\",\"symptom\":\"high_latency\",\"region\":\"us-east\",\"started_at\":\"2026-10-23T03:03:00Z\",\"started_minutes_ago\":null}"
    },
    {
      "role": "user",
      "content": "TOOL_RESULT: continue"
    },
    {
      "role": "assistant",
      "content": "{\"step\":\"get_metrics\",\"service\":\"user-profile\",\"metric\":\"latency_p99\",\"window_minutes\":30}"
    },
    {
      "role": "user",
      "content": "TOOL_RESULT: {\"tool\":\"get_metrics\",\"service\":\"user-profile\",\"metric\":\"latency_p99\",\"window_minutes\":30,\"current\":244.88,\"baseline\":46.37,\"change_pct\":428.1,\"status\":\"critical\"}"
    },
    {
      "role": "assistant",
      "content": "{\"step\":\"get_recent_deploys\",\"service\":\"user-profile\",\"window_minutes\":120}"
    },
    {
      "role": "user",
      "content": "TOOL_RESULT: {\"tool\":\"get_recent_deploys\",\"service\":\"user-profile\",\"window_minutes\":120,\"deploys\":[{\"id\":\"d-5151\",\"kind\":\"code\",\"version\":\"v2.6.5\",\"minutes_before_start\":22}],\"correlated\":true,\"correlated_deploy_id\":\"d-5151\",\"correlated_kind\":\"code\"}"
    },
    {
      "role": "assistant",
      "content": "{\"step\":\"query_logs\",\"service\":\"user-profile\",\"window_minutes\":30}"
    },
    {
      "role": "user",
      "content": "TOOL_RESULT: {\"tool\":\"query_logs\",\"service\":\"user-profile\",\"window_minutes\":30,\"lines\":[\"15:21:10 ERROR [user-profile] ZeroDivisionError: integer division or modulo by zero at handler_214.py [seq=29]\",\"level=error ts=2026-10-22T12:00:00Z service=user-profile msg=\\\"panic recovered: runtime error: invalid memory address\\\" [seq=27]\",\"15:21:16 FATAL [user-profile] uncaught exception: TypeError: Cannot read properties of undefined (reading 'id') [seq=36]\",\"15:21:19 INFO [user-profile] Auto-renewal of telemetry session completed [seq=46]\",\"{\\\"level\\\":\\\"error\\\",\\\"service\\\":\\\"user-profile\\\",\\\"error\\\":\\\"IndexOutOfBoundsException: index 480 out of range\\\"}\"]}"
    },
    {
      "role": "assistant",
      "content": "{\"step\":\"conclude\",\"root_cause\":\"bad_deploy\",\"runbook_id\":\"RB-101\",\"culprit\":\"d-5151\",\"log_signature\":\"unhandled_exception\"}"
    },
    {
      "role": "user",
      "content": "TOOL_RESULT: {\"tool\":\"decision\",\"severity\":\"sev2\",\"action\":\"page_oncall\",\"team\":\"team-identity\",\"runbook_title\":\"Roll back the deploy\"}"
    },
    {
      "role": "assistant",
      "content": "{\"step\":\"draft_summary\",\"text\":\"SEV2: user-profile root cause bad_deploy. Follow RB-101. Team team-identity. Culprit: d-5151.\"}"
    }
  ]
}
```

Notice how every turn pairs an observation with the next disciplined action. After seeing thousands of these diverse variations (including network partitions, database connection pool exhaustion, memory leaks, and retryable tool errors), the 3B model develops an intuitive, reflexive understanding of incident diagnosis.

---

## 4. The Turn-by-Turn Investigation Walkthrough

Let's walk through what happens behind the scenes during the run. Watch how the model methodically eliminates uncertainty at each step, just like an experienced staff engineer.

```
============================================================
[Turn 1: extract_alert]
  Model Output : {"step":"extract_alert","service":"checkout-api","symptom":"high_error_rate","region":"us-east","started_at":null,"started_minutes_ago":12}
------------------------------------------------------------
[Turn 2: get_metrics]
  Model Output : {"step":"get_metrics","service":"checkout-api","metric":"error_rate","window_minutes":30}
  Tool Result  : {"tool":"get_metrics","service":"checkout-api","metric":"error_rate","window_minutes":30,"current":0.18,"baseline":0.01,"change_pct":1700.0,"status":"critical"}
------------------------------------------------------------
[Turn 3: get_recent_deploys]
  Model Output : {"step":"get_recent_deploys","service":"checkout-api","window_minutes":120}
  Tool Result  : {"tool":"get_recent_deploys","service":"checkout-api","window_minutes":120,"deploys":[{"id":"d-7731","kind":"code","version":"v3.8.2","minutes_before_start":15}],"correlated":true,"correlated_deploy_id":"d-7731","correlated_kind":"code"}
------------------------------------------------------------
[Turn 4: query_logs]
  Model Output : {"step":"query_logs","service":"checkout-api","window_minutes":30}
  Tool Result  : {"tool":"query_logs","service":"checkout-api","window_minutes":30,"lines":["15:21:04 ERROR unhandled exception in request handler","15:21:09 FATAL NullPointerException at line 177","15:21:15 INFO healthcheck ping 200 OK"]}
------------------------------------------------------------
[Turn 5: conclude]
  Model Output : {"step":"conclude","root_cause":"bad_deploy","runbook_id":"RB-101","culprit":"d-7731","log_signature":"unhandled_exception"}
  Decision     : {"severity":"sev1","action":"page_oncall_immediately","team":"team-checkout","runbook_title":"Roll back the deploy"}
------------------------------------------------------------
[Turn 6: draft_summary]
  Model Output : {"step":"draft_summary","text":"SEV1: checkout-api root cause bad_deploy. Follow RB-101. Team team-checkout. Culprit: d-7731."}
------------------------------------------------------------
Trace Validation: PASSED (V1-V8)
```

### The Human Narrative Behind Each Turn

#### **Turn 1: `extract_alert`: Cutting Through the Noise**
* **The Engineer's Mindset:** *"Take a breath. What service is this, what is the symptom, and when did this start?"*
* **What Happens:** The model normalizes conversational chatter into clean system entities. It maps `"checkout api"` to `checkout-api`, understands that `"5xx rate way above normal"` means `high_error_rate`, and records that this started 12 minutes ago in `us-east`.

#### **Turn 2: `get_metrics`: Checking Reality First**
* **The Engineer's Mindset:** *"Before waking anyone up, let's verify: is this a real outage, or did a monitor flinch?"*
* **What Happens:** The model queries the metrics tool. The monitoring environment answers: the error rate is at 18% (baseline is 1%, a +1700% spike) and marks the status as **`critical`**. If it had returned normal, the model would have safely closed the alert as transient noise.

#### **Turn 3: `get_recent_deploys`: Investigating Recent Changes**
* **The Engineer's Mindset:** *"The service is burning. Did someone ship code or change a config right before this started?"*
* **What Happens:** The model checks releases over the last 120 minutes. The system flags deployment `d-7731` (`v3.8.2`), which occurred 15 minutes before the incident. Because it landed inside the 45-minute correlation window, it flags `correlated: true`.

#### **Turn 4: `query_logs`: Examining the Crime Scene**
* **The Engineer's Mindset:** *"We have a suspect deploy. Let's look at the logs to confirm what is actually crashing."*
* **What Happens:** The model queries the service logs. It finds `NullPointerException at line 177` and `unhandled exception in request handler`, directly linking the new code release to the crashing requests.

#### **Turn 5: `conclude` & Decision: Reaching a Confident Diagnosis**
* **The Engineer's Mindset:** *"We know what happened, we know how to fix it, and we know who owns it."*
* **What Happens:** The model concludes that the root cause is `bad_deploy`, the culprit is `d-7731`, and the appropriate fix is runbook `RB-101` (*"Roll back the deploy"*). Deterministic business logic immediately computes the triage severity (`sev1`), assigns `team-checkout`, and triggers `page_oncall_immediately`.

#### **Turn 6: `draft_summary`: Relieving the Team's Anxiety**
* **The Engineer's Mindset:** *"Communicate clearly to the company so everyone knows help is on the way."*
* **What Happens:** The model drafts a concise broadcast citing only verified facts. No hallucinations, no guessing.

---

## 5. Safety Guardrails: The Eight Invariants (V1-V8)

In a production outage, an AI hallucinating an imaginary deploy ID or rolling back the wrong database can turn a minor glitch into a catastrophe. Every single run must pass **8 strict automated invariants**:

* **V1 (Valid Sequence):** The model cannot jump straight to conclusions; it must verify metrics and logs first.
* **V2 (Schema Integrity):** Every response must parse as valid, deterministic JSON.
* **V3 (Entity Consistency):** The service and region must faithfully match the original alert.
* **V4 (Ground-Truth Accuracy):** The diagnosed root cause must match actual environment telemetry.
* **V5 (Runbook Alignment):** Prescribed runbook must accurately remediate the root cause (`RB-101` for bad deploys).
* **V6 (Culprit Verification):** The identified culprit commit/deploy must actually exist in the deploy history.
* **V7 (Log Evidence):** The cited log signature must be present in the queried logs.
* **V8 (Zero-Hallucination Summary):** The final Slack message can only reference facts verified during the trace.

---

## 6. Benchmark Results: Why This Matters

Here are the concrete results from our evaluation suite on the held-out test split:

| Model Setup | Prompt Used | Success Rate (`e2e_ok`) | Input Tokens / Turn | Cost / 1k Incidents | On-Call Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Gemini 2.5 Pro (`Baseline_A`)** | Full SOP (no cache) | 90.0% | 5,200 | $3.50+ | High accuracy, but high latency & cost |
| **Gemini 2.5 Pro (`Baseline_B`)** | Full SOP (cached) | 92.0% | 5,200 | $1.85 | Good performance, still dependent on cloud API |
| **Raw Qwen 3B Base (`Baseline_C`)** | Full SOP in prompt | 20.0% | 5,200 | Slow / Failing | Context overload; confuses small untuned models |
| **Our Fine-Tuned SLM (`Model_D`)** | **Zero Prompt (`"incident_triage_v1"`)** | **93.0%** | **450** | **$0.22** | **Best accuracy, lowest latency, 8.4x cheaper** |

### The Core Takeaway
When you teach the procedure to the weights, you free the model from the burden of re-reading its own job description. It runs cooler, faster, and more reliably, right when you need it most.

---

## 7. Try It Yourself

You can test any alert string right now in your terminal:

```bash
# 1. Run with our fine-tuned local model (zero prompt overhead)
./venv/bin/python -m exp.runtime.cli --model ep3 "[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago"

# 2. Run with the reference oracle (instant deterministic verification)
./venv/bin/python -m exp.runtime.cli --model oracle "[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago"

# 3. Run with Gemini Pro carrying the full prompt (src/exp/prompts/sop.md)
./venv/bin/python -m exp.runtime.cli --model gemini "[FIRING] checkout api 5xx rate way above normal in us-east, started about 12 min ago"
```
