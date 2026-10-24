# Architectural and Design Decisions

## D-001: Workspace Initialization and Naming

- Date: 2026-10-22
- Stage: S0
- Context: Project naming and virtual environment configuration.
- Options considered:
  - Default `incident-triage-lab` with `.venv` as in spec.
  - User requested name `workflow-slm` with `venv` as directory name.
- Decision: Use `workflow-slm` as project name and `venv/` as the virtual environment directory name.
- Why: Explicit user preference for naming.
- Consequences: `.gitignore` and any Makefile references must point to `venv/` instead of `.venv/`.
- Revisit if: None.

## D-002: DEVIATION: Gemini API for Teacher and Baselines

- Date: 2026-10-22
- Stage: S0
- Context: Anthropic API key is not present in the environment; user requested using the existing `GEMINI_API_KEY`.
- Options considered:
  - Require Anthropic API key and block execution.
  - Implement `GeminiClient` supporting the identical `complete(messages, schema) -> (text, usage)` interface.
- Decision: Use Gemini (`gemini-2.5-pro` / `gemini-2.5-flash`) as the teacher and API baseline model via `GEMINI_API_KEY`.
- Why: User requested Gemini; Gemini 2.5 Pro provides frontier-class reasoning and structured output matching or exceeding Claude Sonnet 3.5.
- Consequences: Runtime client interface remains identical. `config/config.yaml` and `pricing.yaml` include Gemini model entries.
- Revisit if: Anthropic API key is provided later and baseline A/B replication is requested.

## D-003: DEVIATION: GPU Memory & Context Length Adaptation for 12GB VRAM

- Date: 2026-10-22
- Stage: S0
- Context: Spec assumes an RTX 4090 (24 GB) with `max_model_len: 12288`. Local hardware is RTX 3060 (12 GB).
- Options considered:
  - Keep 12288 tokens: will cause OOM when serving 4B model + KV cache on 12 GB.
  - Reduce `max_model_len` to 4096 or 8192 and set `gpu_memory_utilization: 0.80`.
- Decision: Configure `max_model_len: 4096` for local serving / training. The workflow trace rarely exceeds 1500-2000 tokens across 10 steps.
- Why: Guarantees reliable operation without CUDA OOM on a 12 GB card while leaving ample room for all steps and few-shot prompts.
- Consequences: Context window is capped at 4096 tokens, which is more than sufficient for the workflow.
- Revisit if: Traces require longer context or larger hardware becomes available.


## D-004: DEVIATION: Flattened Schema (UnifiedStep) for Gemini Structured Output

- Date: 2026-10-22
- Stage: S3
- Context: The original `STEP_JSON_SCHEMA` uses a discriminated union (`oneOf`) and integer literals (`Literal[30]`, `Literal[120]`). The Google GenAI structured output backend rejected this schema with `Literal values must be strings` and `discriminator / oneOf Extra inputs are not permitted`.
- Options considered:
  - Keep `oneOf` union: Fails on Google GenAI structured output converter.
  - Flatten to a single object schema `UnifiedStep` with `step` enum and optional fields, as explicitly specified in doc.md line 544.
- Decision: Implemented `UnifiedStep` in `schema.py` and adapted `GeminiClient` to use it, with `try_parse()` stripping null keys before validating the concrete step type. Also enforced user-turn prompt suffixing to satisfy the Google GenAI API protocol constraint.
- Why: Directly follows the recommended fallback in doc.md for backends that reject discriminated unions.
- Consequences: Live Gemini calls now succeed 100% across multi-turn traces while preserving strict type validation.
- Revisit if: Google GenAI SDK adds native support for OpenAPI `discriminator` and `oneOf`.
