# Vital Trace: Protocol-Constrained Patient-State Reasoning for Longitudinal Clinical Trajectories

This repository contains the implementation for the paper:

**Vital Trace: Protocol-Constrained Patient-State Reasoning for Longitudinal Clinical Trajectories**

It includes:
- signal-centric longitudinal preprocessing (MIMIC-IV and eICU style schemas),
- staged multi-agent inference (Router/Reasoner/Auditor/Steward),
- single-LLM and learning baselines,
- evaluation utilities (overall, calibration, temporal, protocol, counterfactual, efficiency),
- the manually curated global protocol used by Vital Trace.

## Repository Layout

- `src/preprocess_pipeline/`
  - Stage 1-4 preprocessing for cohorting, signal filtering, transition construction, labeling, and sample selection.
- `src/latent_pipeline/`
  - Multi-agent staged inference, baselines, prompts, metrics, counterfactual evaluation, and runners.
- `data/global_protocol_manual.json`
  - Global protocol (manual curation).
- `scripts/preprocess/`
  - Reproducible preprocessing entry scripts (local + Slurm templates).
- `scripts/experiments/`
  - Reproducible experiment launch scripts (local + Slurm templates).
- `docs/REPRODUCIBILITY.md`
  - Detailed end-to-end runbook for paper experiments.

## Environment

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Expectations

Preprocessing expects MIMIC-IV/eICU style raw tables and mappings. Paths are passed explicitly via CLI or environment variables; no machine-specific paths are required.

## Quick Start

1. Configure paths:

```bash
export PROJECT_ROOT=$(pwd)
export RUN_ROOT=$PROJECT_ROOT/runs
mkdir -p "$RUN_ROOT"
```

2. Run preprocessing:

```bash
bash scripts/preprocess/run_stage1_dual.sh
bash scripts/preprocess/run_stage2_dual.sh
bash scripts/preprocess/run_stage3_full1000_dual.sh
```

3. Run staged experiments:

```bash
bash scripts/experiments/run_mimic_experiments.sh
```

## Notes

- No patient data is included.
- All scripts are path-configurable.
- For exact paper protocol, use `docs/REPRODUCIBILITY.md`.
