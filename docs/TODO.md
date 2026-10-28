# Future Tasks

- [ ] **Compare oracle-generated and teacher-generated fine-tuning data.**
  - Export deterministic oracle demonstrations as JSON training conversations,
    without hosted-model calls. Keep hidden annotations and expected answers out
    of student inputs, just as with teacher-generated data.
  - Compare against GPT-6.1 Terra demonstrations using matched training scenario
    IDs, the same base model, training budget, and controlled random seeds.
  - Evaluate both students on the same held-out scenarios: end-to-end success,
    extraction accuracy, workflow/branch adherence, and grounded summaries.
  - Report generation cost/time and differences in branch coverage or filtering.
    Account for the oracle's privileged extraction and log-signature information.
  - Treat this as a later experiment; do not replace or interrupt the current
    teacher-data generation run.
