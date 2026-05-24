#!/bin/bash
#SBATCH --job-name=vitaltrace-exp
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=180G
#SBATCH --output=out/%x-%j.out
set -euo pipefail

# Required:
#   PROJECT_ROOT
#   CONFIG_PATH

PROJECT_ROOT=${PROJECT_ROOT:?set PROJECT_ROOT}
CONFIG_PATH=${CONFIG_PATH:?set CONFIG_PATH}

cd "$PROJECT_ROOT"
python -m src.latent_pipeline.run_staged_from_config --config "$CONFIG_PATH"
