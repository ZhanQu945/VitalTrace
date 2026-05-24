# Reproducibility Guide

This document describes the full pipeline from raw data preprocessing to core experiments.

## 1. Preprocessing Pipeline

The preprocessing is staged and checkpointed.

## Stage 1: Cohort + Intervention Relevance

Purpose:
- build encounter-level cohort,
- keep one encounter per patient,
- retain encounters with relevant intervention evidence.

Entry:
- `python -m src.preprocess_pipeline.signal_pipeline_stage1 --help`

Outputs:
- `stage1_cohort.parquet`
- `stage1_events_interventions.parquet`
- `stage1_summary.json`

## Stage 2: Signal Extraction + Consistency

Purpose:
- extract protocol-relevant physiological signals,
- merge with intervention events,
- remove inconsistent encounters (abnormal signal patterns without matching intervention evidence under configured policy).

Entry:
- `python -m src.preprocess_pipeline.signal_pipeline_stage2 --help`

Outputs:
- `stage2_cohort.parquet`
- `stage2_events.parquet`
- `stage2_summary.json`

## Stage 3: Transition Construction + Horizon Labels

Purpose:
- construct longitudinal transition steps from signal state transitions,
- label outcomes at prediction horizons (e.g., h6/h12),
- rank encounters and keep top-K,
- select one shared contiguous window per encounter (e.g., 20 steps).

Entry:
- `python -m src.preprocess_pipeline.signal_pipeline_stage3 --help`

Key settings used in paper runs:
- `--horizons 6,12`
- `--simplified-top-k 1000`
- `--meaningful-steps-per-encounter 20`

Outputs:
- `stage3_steps.jsonl`
- `stage3_labeled_h6.jsonl`
- `stage3_labeled_h12.jsonl`
- `stage3_summary.json`

## Stage 4: Sample Export (optional)

Purpose:
- export smaller ranked subsets for rapid iteration and diagnostics.

Entry:
- `python -m src.preprocess_pipeline.signal_pipeline_stage4 --help`

---

## 2. Experiments

## Staged Vital Trace / Free-form / Ablations

Runner:
- `python -m src.latent_pipeline.run_staged_from_config --config <yaml>`

Config fields:
- `io`: input JSONL, output dir, protocol JSON
- `runtime`: agent backend, max rules, runner mode, retry/fail policy, ablation switch
- `model`: model id + decoding/input limits

## Single-LLM and Learning Baselines

Runner:
- `python -m src.latent_pipeline.run_baselines --help`

---

## 3. Evaluation Outputs

Main outputs in each experiment directory:
- `metrics_overall.json`
- `per_label_metrics.csv`
- `calibrated_metrics.json`
- `temporal_metrics.json`
- `risk_state_metrics.json`
- `protocol_consistency_metrics.json`
- `counterfactual_metrics.json`
- `efficiency_metrics.json`
- `predictions_with_gt.csv`

---

## 4. Counterfactual Evaluation

Utility:
- `python -m src.latent_pipeline.counterfactual_runner --out-dir <dir> --protocol-json <path>`

Reports:
- directional consistency,
- protocol activation change under perturbation,
- aggregate counterfactual summaries.

---

## 5. Aggregation

Use:
- `python scripts/experiments/collect_results.py --jobs-tsv <jobs.tsv> --out-csv <summary.csv>`

The collector writes one row per experiment run with key predictive, calibration, protocol, counterfactual, and efficiency metrics.
