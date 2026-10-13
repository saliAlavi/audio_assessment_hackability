#!/bin/bash
# Registers the Jupyter kernel "hackpatch" used by notebooks/: the project venv plus the
# scripts/env.sh variables, so notebooks work from any Jupyter (e.g. OnDemand) without sourcing it first.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
args=()
for v in SCRATCH_DIR PROJECT_NAME HF_HOME TORCH_HOME XDG_CACHE_HOME WANDB_CACHE_DIR WANDB_DATA_DIR WANDB_ENTITY \
         TOKENIZERS_PARALLELISM HACKPATCH_SLURM_ACCOUNT HACKPATCH_SLURM_PARTITION HACKPATCH_WANDB_MODE \
         HACKPATCH_ARTIFACTS HACKPATCH_DATA_ROOT; do
  if [ -n "${!v:-}" ]; then args+=(--env "$v" "${!v}"); fi
done
python -m ipykernel install --user --name hackpatch --display-name "hackpatch (scripts/env.sh)" \
  --env PYTHONPATH "$PROJECT_ROOT/src" "${args[@]}"
