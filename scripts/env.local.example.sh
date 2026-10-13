# Machine-specific settings, sourced by scripts/env.sh. Copy to scripts/env.local.sh (not tracked) and edit;
# every variable is optional and an already-exported value wins.
: "${SCRATCH_DIR:=/path/with/space}"                  # checkpoints (19 GB), caches, run outputs
# : "${HACKPATCH_VENV:=$SCRATCH_DIR/envs/hackpatch}"  # a virtualenv to activate
# export HACKPATCH_DATA_ROOT="${HACKPATCH_DATA_ROOT:-/data/speech}"            # corpora, see docs/eval_corpora.md
# export HACKPATCH_SLURM_ACCOUNT="${HACKPATCH_SLURM_ACCOUNT:-my_account}"     # sbatch --account
# export HACKPATCH_SLURM_PARTITION="${HACKPATCH_SLURM_PARTITION:-gpu}"       # sbatch --partition
# export WANDB_ENTITY="${WANDB_ENTITY:-my_entity}"
# export HACKPATCH_WANDB_MODE="${HACKPATCH_WANDB_MODE:-online}"              # default: offline
