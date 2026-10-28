# Codex Teacher Pilot

Date: 2026-10-28. Teacher: `gpt-6.1-terra` through Codex CLI `0.165.0`,
authenticated with ChatGPT. These are teacher-trace acceptance measurements,
not evaluation results for a fine-tuned small model.

## Extraction pilot

The original first ten training scenarios produced eight accepted traces.
After rendering the generator's public service/region aliases into the SOP
and clarifying typo recovery and summary grounding, a fresh run of the same
ten scenarios produced ten accepted traces. The scenario data and validators
were unchanged for this comparison. Both previous extraction failures were
resolved, and no previously accepted scenario regressed. This small pilot
does not establish general reliability or isolate effects from model variation.

The prompt contains public vocabulary and procedure rules, never a scenario's
expected extraction or hidden root-cause labels. Model outputs are not silently
repaired to match the ground truth.

## 100-scenario run

The serial extension to the first 100 training scenarios reused the ten accepted
pilot traces. It accepted 96 scenarios and rejected four. The extension took
2,748.36 seconds; this excludes the earlier ten-scenario run.

All four rejections were exact-extraction mismatches on the region:

| Scenario | Region text in alert | Expected canonical region |
| :--- | :--- | :--- |
| `scen_train_00011` | `ap-suoth-1` | `ap-south` |
| `scen_train_00024` | `ue-west` | `eu-west` |
| `scen_train_00035` | `dublni` | `eu-west` |
| `scen_train_00080` | `us-eats-1` | `us-east` |

The teacher returned `null` for these regions. The rejected traces remain
rejected; no extraction tolerance or validator exemption was added. Further
region-typo guidance would be a new prompt version and needs a separate comparison.

Accepted conclusion counts (which can differ from latent scenario causes when
tools fail or evidence is inconclusive):

| Conclusion | Accepted traces |
| :--- | ---: |
| `unknown` | 15 |
| `resource_exhaustion` | 9 |
| `db_pool_exhausted` | 9 |
| `credential_failure` | 4 |
| `config_change_error` | 10 |
| `bad_deploy` | 10 |
| `upstream_dependency_failure` | 13 |
| `traffic_spike` | 11 |
| `transient_noise` | 15 |

This is a prefix sample, not a stratified sample. All nine conclusion categories
are represented, but fine-grained branch coverage and rare-branch retention
still need checking before full generation.

## Audit and hardening

- An in-memory audit of 87 accepted conversations with the actual Qwen2.5-3B
  tokenizer found 246–918 tokens per conversation, no 2,048-token truncation,
  no zero-supervision examples, and exact assistant-only loss masking. This
  snapshot does not replace checking lengths on the full eventual SFT dataset.
- The trajectory validator previously accepted a fabricated lucky diagnosis
  that omitted required tools. It now requires the complete tool procedure
  before conclusion, followed by exactly one terminal summary.
- Local revalidation after that fix retained all 96 accepted traces; none were
  newly rejected. The four region-extraction failures remained rejected.
- A fatal Codex error now stops scheduling new scenarios but drains and saves
  other workers' already-submitted results. Resume rechecks saved accepted
  traces with the current validators.
- Oracle evaluation now uses scenario-specific responders. Teacher-forced
  evaluation scores conclusions and grounded summaries, includes the runtime's
  post-extraction continuation turn, and uses the same prompt mode as free-run.

Ten-worker concurrency has offline regression coverage, not a measured live
speedup yet. After reviewing the pilot, a bounded throughput test is:

```bash
make traces TRACE_ARGS="--limit 200 --workers 10"
```

Do not run another writer against the same trace directory. Neither the full
4,000-scenario generation nor model training was started as part of this pilot.
Raw traces, backups, model weights, and login credentials are not committed.
