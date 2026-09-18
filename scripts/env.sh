# Sourced by every job and interactive shell for this project.
# All caches on scratch (home quota is tiny); nips conda env + overlay venv for extras.
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SCRATCH_DIR=/fs/scratch/PAS2301/alialavi
export PROJECT_NAME=audio_assessment_hackability
source "$SCRATCH_DIR/envs/hackpatch/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH_DIR/cache/audio_assessment_hackability_hf"
export TORCH_HOME="$SCRATCH_DIR/cache/audio_assessment_hackability_torch"
export XDG_CACHE_HOME="$SCRATCH_DIR/cache/audio_assessment_hackability_xdg"
export WANDB_CACHE_DIR="$SCRATCH_DIR/cache/audio_assessment_hackability_wandb"
export WANDB_DATA_DIR="$WANDB_CACHE_DIR"
export WANDB_ENTITY=alialavi
[ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export TOKENIZERS_PARALLELISM=false
