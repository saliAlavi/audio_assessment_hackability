#!/usr/bin/env python3
"""Slurm submitter enforcing the cluster, storage and W&B conventions.

Fixes RUN_NAME (``{config_name}__{timestamp}``) and the W&B run id at
submission, so the run URL is printed before the job starts. Logs go to the
scratch run dir. Seed arrays give one run (name, dir, W&B id) per seed.

Examples:
    python scripts/submit.py --config-name train_r0 --seeds 0,1,2 --time 02:00:00
    python scripts/submit.py --config-name hack_loop --seeds 0 --time 10:00:00 \
        --after 123456 -- init_ckpt=/path/best.pt
    python scripts/submit.py --config-name train_r0 --dry-run -- train.steps=50
"""
from __future__ import annotations

import argparse
import datetime
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hackpatch.utils import paths  # noqa: E402

ACCOUNT = "PAS2966"
ENTITY = "alialavi"

TEMPLATE = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --time={walltime}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{gpu_line}{dep_line}
set -euo pipefail
# Run from the frozen code snapshot so later repo edits cannot change this job.
cd {snapshot}
source scripts/env.sh
export RUN_NAME="{run_name}"
export WANDB_RUN_ID="{wandb_id}"
export WANDB_RESUME=allow
echo "[wandb] https://wandb.ai/{entity}/{project}/runs/{wandb_id}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python main.py --config-name {config_name} seed={seed} run_name="$RUN_NAME" {overrides}
"""


def _git_state() -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    return sha or "nogit", dirty


def _pending_deps(ids: str | None) -> str | None:
    """Drops dependencies on jobs that already left the queue (Slurm rejects those).

    Raises:
        RuntimeError: If a finished dependency did not complete successfully.
    """
    if not ids:
        return None
    live = []
    for j in ids.split(":"):
        if subprocess.run(["squeue", "-h", "-j", j], capture_output=True, text=True).stdout.strip():
            live.append(j)
            continue
        state = subprocess.run(["sacct", "-n", "-X", "-j", j, "--format=State"], capture_output=True,
                               text=True).stdout.split()
        if state and state[0] != "COMPLETED":
            raise RuntimeError(f"dependency job {j} finished with state {state[0]}")
    return ":".join(live) or None


def _pending_deps_any(ids: str | None) -> str | None:
    """Like :func:`_pending_deps` but ignores the finished jobs' exit state (lane ordering only)."""
    if not ids:
        return None
    live = [j for j in ids.split(":")
            if subprocess.run(["squeue", "-h", "-j", j], capture_output=True, text=True).stdout.strip()]
    return ":".join(live) or None


def submit_one(config_name: str, seed: int, tag: str = "", time: str = "02:00:00", partition: str = "nextgen",
               gpus: int = 1, cpus: int = 12, mem: str = "64G", after: str | None = None,
               overrides: list[str] | None = None, dry_run: bool = False,
               lane_after: str | None = None) -> tuple[str, Path]:
    """Snapshots the code, renders the sbatch script and submits one run.

    Args:
        config_name: Config file name in ``configs/``.
        seed: Seed passed as ``seed=``.
        tag: Variant label appended to the run name.
        time: Walltime.
        partition: Slurm partition.
        gpus: GPUs per node.
        cpus: CPUs per task.
        mem: Memory.
        after: ``afterok`` dependency job ids (colon-separated) or None (data dependencies).
        lane_after: ``afterany`` job id used only to serialize GPU usage (runs even if that job failed).
        overrides: Hydra overrides.
        dry_run: Print instead of submitting.

    Returns:
        (job id, run directory).
    """
    import wandb.util

    after, lane_after = _pending_deps(after), _pending_deps_any(lane_after)
    cfg_label = f"{config_name}{('_' + tag) if tag else ''}_s{seed}"
    run_name = paths.make_run_name(cfg_label)
    rp = paths.RunPaths.create(cfg_label, run_name=run_name)
    wandb_id = wandb.util.generate_id()
    script = TEMPLATE.format(
        account=ACCOUNT, partition=partition, job_name=cfg_label, log_dir=rp.logs,
        walltime=time, cpus=cpus, mem=mem,
        gpu_line=f"#SBATCH --gpus-per-node={gpus}\n" if gpus else "",
        dep_line=("#SBATCH --dependency=" + ",".join(
            ([f"afterok:{after}"] if after else []) + ([f"afterany:{lane_after}"] if lane_after else [])) + "\n")
        if (after or lane_after) else "",
        snapshot=rp.root / "code_snapshot", run_name=run_name, wandb_id=wandb_id, entity=ENTITY,
        project=paths.project_name(), config_name=config_name, seed=seed,
        overrides=" ".join(shlex.quote(o) for o in (overrides or [])))
    # Code snapshot: exact provenance even from an uncommitted tree.
    snap = rp.root / "code_snapshot"
    for item in ("src", "configs", "scripts", "main.py"):
        src, dst = ROOT / item, snap / item
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    path = rp.root / "submit.sbatch"
    path.write_text(script)
    if dry_run:
        print(script)
        return "DRYRUN", rp.root
    out = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    job_id = out.stdout.strip().split()[-1]
    print(f"job {job_id} | {run_name}\n  run dir : {rp.root}\n"
          f"  wandb   : https://wandb.ai/{ENTITY}/{paths.project_name()}/runs/{wandb_id}", flush=True)
    with open(paths.SCRATCH_DIR / "projects" / paths.project_name() / "_submissions.log", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()}\t{job_id}\t{run_name}\t{rp.root}\t{after or ''}\t"
                f"{' '.join(overrides or [])}\n")
    return job_id, rp.root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--seeds", default="0", help="Comma list; one job per seed.")
    ap.add_argument("--tag", default="", help="Suffix for RUN_NAME to distinguish variants.")
    ap.add_argument("--time", default="02:00:00")
    ap.add_argument("--partition", default="nextgen")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=12)
    ap.add_argument("--mem", default="64G")
    ap.add_argument("--after", default=None, help="afterok dependency job id(s), colon-separated.")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*", help="Hydra overrides after --")
    args = ap.parse_args()

    sha, dirty = _git_state()
    if dirty and not args.allow_dirty:
        sys.exit("error: git tree has uncommitted changes; commit first or pass --allow-dirty")
    for seed in [int(s) for s in args.seeds.split(",")]:
        submit_one(args.config_name, seed, args.tag, args.time, args.partition, args.gpus, args.cpus, args.mem,
                   args.after, args.overrides, args.dry_run)


if __name__ == "__main__":
    main()
