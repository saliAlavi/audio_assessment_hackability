# Sourced by every job and interactive shell for this project.
# Machine-specific settings (SCRATCH_DIR, HACKPATCH_VENV, HACKPATCH_DATA_ROOT, HACKPATCH_SLURM_ACCOUNT/PARTITION,
# WANDB_ENTITY, HACKPATCH_WANDB_MODE) come from the environment or from scripts/env.local.sh, which is not tracked:
# copy scripts/env.local.example.sh. See README.md, "Setup".
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$PROJECT_ROOT/scripts/env.local.sh" ]; then source "$PROJECT_ROOT/scripts/env.local.sh"; fi
export SCRATCH_DIR="${SCRATCH_DIR:-$HOME/hackpatch_scratch}"   # all caches and run outputs (home quotas are small)
if [ -n "${HACKPATCH_VENV:-}" ]; then source "$HACKPATCH_VENV/bin/activate"; fi
export PROJECT_NAME=audio_assessment_hackability
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH_DIR/cache/audio_assessment_hackability_hf"
export TORCH_HOME="$SCRATCH_DIR/cache/audio_assessment_hackability_torch"
export XDG_CACHE_HOME="$SCRATCH_DIR/cache/audio_assessment_hackability_xdg"
export WANDB_CACHE_DIR="$SCRATCH_DIR/cache/audio_assessment_hackability_wandb"
export WANDB_DATA_DIR="$WANDB_CACHE_DIR"
[ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export TOKENIZERS_PARALLELISM=false
