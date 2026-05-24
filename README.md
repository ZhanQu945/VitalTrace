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

## Data Access and Expectations

This repository does not include patient data.

To run the full pipeline, you need authorized access to:
- **MIMIC-IV** (PhysioNet credentialed access)
- **eICU Collaborative Research Database** (PhysioNet credentialed access)

Access pages:
- MIMIC access FAQ: https://mimic.mit.edu/docs/faq/how-to-get-access.html
- eICU access guide: https://eicu.mit.edu/gettingstarted/access/
- PhysioNet project pages:
  - https://physionet.org/content/mimiciv/
  - https://physionet.org/content/eicu-crd/

After access is approved, download/decompress the source tables into local folders (for example `./data/mimic` and `./data/eicu`) and pass those paths to preprocessing scripts through environment variables.

Expected raw data are standard MIMIC/eICU relational tables used by:
- `data/preprocess_longitudinal.py`
- `src/preprocess_pipeline/signal_pipeline_stage1.py`
- `src/preprocess_pipeline/signal_pipeline_stage2.py`

The code assumes MIMIC/eICU-style column schemas (encounter IDs, timestamps, labs, vitals, medications, procedures, and mapping tables).

Minimum expected source files include:
- MIMIC-style tables such as admissions, patients, diagnoses/procedures ICD tables, prescriptions, labs, and item dictionaries.
- eICU-style tables for patient/unit stay metadata, labs, vitals, medications, and procedures/treatments.

## Quick Start

1. Configure paths:

```bash
export PROJECT_ROOT=$(pwd)
export RUN_ROOT=$PROJECT_ROOT/runs
export MIMIC_ROOT=$PROJECT_ROOT/data/mimic
export EICU_ROOT=$PROJECT_ROOT/data/eicu
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
