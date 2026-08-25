# Evaluation Integrity

This document defines the implementation used for corrected experiment runs.
Legacy artifacts produced before these corrections must not be combined with new
results.

## Inference Context

Future-horizon targets remain in output records for scoring but are excluded from
all agent prompts. Corrected rows and metrics use
`inference_context_schema: target_free_v1` and
`target_isolation_verified: true`.

## Prediction Endpoints

The Reasoner predicts three support probabilities:

- `vasopressor_signal`
- `resp_support_signal`
- `renal_support_signal`

`any_deterioration` is a composite endpoint, not an independent Reasoner output:

```text
p(any_deterioration) = max(p(vasopressor), p(respiratory support), p(renal support))
```

Its binary ground truth is the logical OR of the same three future intervention
labels.

## Persistent State

The Steward maintains five bounded integer states in `[0, 5]`:

- `hemodynamic_state`
- `respiratory_state`
- `renal_state`
- `metabolic_state`
- `systemic_inflammation_state`

## Counterfactual Evaluation

`counterfactual_runner.py` applies four standardized recovery perturbations to
the current step:

- low MAP to 75 mmHg
- rising lactate to stable, with the latest value capped at 2.0 mmol/L
- high/rising creatinine to stable, with the latest value capped at 1.2 mg/dL
- low SpO2 to 95% and/or high respiratory rate to 20 breaths/min

For each eligible perturbation, the runner reruns Router and Reasoner with the
configured experiment backend and model. Auditor and Steward are then rerun to
measure protocol-activation and individual-state changes. Results are accepted
by collectors only when
`evaluation_method: standardized_recovery_model_rerun_v2` is present.

Adjacent observed trajectory steps are not treated as counterfactuals. Baseline
counterfactual results are omitted unless a matching baseline-specific inference
adapter is available.

## Confidence Intervals

Macro AUROC, AUPRC, and F1 confidence intervals use 1,000 patient-cluster
bootstrap replicates by default. Patients are sampled with replacement and every
sampled patient's rows are concatenated with multiplicity. The replicate count
is configurable with `--bootstrap-replicates`.
