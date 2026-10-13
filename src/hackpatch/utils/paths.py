#!/usr/bin/env python3
"""Canonical path resolver for the project's cluster storage layout.

This module is the single source of truth for every path the project touches.
Copy it into the project (``src/utils/paths.py``) and import it everywhere;
no other file may contain a scratch/home path literal.

Layout (see the storage-caching skill):
    RUN_DIR      = {SCRATCH_DIR}/projects/{PROJECT_NAME}/{RUN_NAME}/
    CACHE_DIR    = {SCRATCH_DIR}/cache/{PROJECT_NAME}_{DATASET_NAME}/
    RAW_DATA_DIR = {SCRATCH_DIR}/datasets/{PROJECT_NAME}_{DATASET_NAME}/

Typical usage:
    from src.utils import paths

    run = paths.RunPaths.create(config_name="main_table1")
    torch.save(state, run.checkpoints / "latest.pt")
    print(run.wandb_url_hint())
"""
from __future__ import annotations

import dataclasses
import datetime
import os
import subprocess
from pathlib import Path

_ORIGINAL_SCRATCH = Path("/fs/scratch/PAS2301/alialavi")
#: Root of all large storage: ``$SCRATCH_DIR``, else the original cluster's scratch, else ``~/hackpatch_scratch``.
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR")
                   or (_ORIGINAL_SCRATCH if _ORIGINAL_SCRATCH.is_dir() else Path.home() / "hackpatch_scratch"))


def project_name() -> str:
    """Returns the project name, derived from the git repository root dirname.

    Deriving (rather than hardcoding) the name keeps the layout correct when
    the repo is cloned under a different path and guarantees the W&B project
    name (wandb-diagnostics skill) and the storage tree always agree.

    Returns:
        The basename of the git root, or of the CWD when not in a git repo.
    """
    if os.environ.get("PROJECT_NAME"):
        return os.environ["PROJECT_NAME"]
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        name = Path(root).name
    except Exception:
        name = Path.cwd().name
    # Path("/").name == "" — never let an empty name collapse the layout.
    return name or "unnamed_project"


def make_run_name(config_name: str, when: datetime.datetime | None = None) -> str:
    """Builds the canonical run name ``{config_name}__{YYYY-MM-DD_HH-MM-SS}``.

    Args:
        config_name: Name of the experiment/config that produced this run
            (``cfg.config_name`` in the single-config pattern).
        when: Timestamp to embed; defaults to now. Pass explicitly when a
            submitter pre-allocates the name before the job starts, so the
            sbatch script and the training process agree on one RUN_NAME.

    Returns:
        The run name string, filesystem- and W&B-safe.
    """
    when = when or datetime.datetime.now()
    return f"{config_name}__{when.strftime('%Y-%m-%d_%H-%M-%S')}"


def cache_dir(dataset_name: str, create: bool = True) -> Path:
    """Returns the cache directory for a dataset, creating it if needed.

    Args:
        dataset_name: Short dataset identifier (e.g. ``"cifar10"``).
        create: Whether to ``mkdir -p`` the directory.

    Returns:
        ``{SCRATCH_DIR}/cache/{PROJECT_NAME}_{DATASET_NAME}``.
    """
    p = SCRATCH_DIR / "cache" / f"{project_name()}_{dataset_name}"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def raw_data_dir(dataset_name: str, create: bool = True) -> Path:
    """Returns the raw (untouched) data directory for a dataset.

    Args:
        dataset_name: Short dataset identifier.
        create: Whether to ``mkdir -p`` the directory.

    Returns:
        ``{SCRATCH_DIR}/datasets/{PROJECT_NAME}_{DATASET_NAME}``.
    """
    p = SCRATCH_DIR / "datasets" / f"{project_name()}_{dataset_name}"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


@dataclasses.dataclass(frozen=True)
class RunPaths:
    """All per-run output locations, created together and passed around as one.

    Attributes:
        run_name: The canonical ``{config_name}__{timestamp}`` identifier;
            identical to the W&B run name.
        root: ``{SCRATCH_DIR}/projects/{PROJECT_NAME}/{RUN_NAME}``.
        checkpoints: Model/optimizer/RNG state files.
        results: Metrics, predictions, eval outputs.
        wandb: Local W&B files (``WANDB_DIR`` is pointed here).
        logs: Slurm stdout/stderr and text logs.
        artifacts: Figures, tables, miscellaneous run products.
    """

    run_name: str
    root: Path
    checkpoints: Path
    results: Path
    wandb: Path
    logs: Path
    artifacts: Path

    @classmethod
    def create(cls, config_name: str, run_name: str | None = None) -> "RunPaths":
        """Creates the full run directory tree and exports ``WANDB_DIR``.

        Args:
            config_name: The config name used to build RUN_NAME when
                ``run_name`` is not given.
            run_name: A pre-allocated run name (used when the job submitter
                already fixed it — see the slurm-cluster skill); otherwise
                generated from ``config_name`` and the current time.

        Returns:
            A ``RunPaths`` with every subdirectory existing on disk.
        """
        name = run_name or os.environ.get("RUN_NAME") or make_run_name(config_name)
        root = SCRATCH_DIR / "projects" / project_name() / name
        sub = {k: root / k for k in
               ("checkpoints", "results", "wandb", "logs", "artifacts")}
        for p in sub.values():
            p.mkdir(parents=True, exist_ok=True)
        # W&B must write to scratch, never to $HOME (storage-caching skill).
        os.environ.setdefault("WANDB_DIR", str(sub["wandb"]))
        return cls(run_name=name, root=root, **sub)

    def wandb_url_hint(self, entity: str | None = None) -> str:
        """Returns the W&B run URL if the run id is known, else the project URL.

        The slurm-cluster submitter pre-allocates ``WANDB_RUN_ID`` so this is
        printable at submission time, before the job starts.

        Args:
            entity: W&B entity (team/user); read from ``WANDB_ENTITY`` if unset.

        Returns:
            A best-effort ``https://wandb.ai/...`` URL string.
        """
        entity = entity or os.environ.get("WANDB_ENTITY", "<entity>")
        proj = project_name()
        run_id = os.environ.get("WANDB_RUN_ID")
        base = f"https://wandb.ai/{entity}/{proj}"
        return f"{base}/runs/{run_id}" if run_id else base


if __name__ == "__main__":
    # Self-test: exercise every function against a temp scratch root.
    SCRATCH_DIR = Path("/tmp/scratch_selftest")  # noqa: F811 — test override

    rp = RunPaths.create(config_name="unit_test_cfg")
    assert rp.checkpoints.is_dir() and rp.wandb.is_dir()
    assert rp.run_name.startswith("unit_test_cfg__")
    assert os.environ["WANDB_DIR"] == str(rp.wandb)
    assert cache_dir("toy").name.endswith("_toy")
    assert raw_data_dir("toy").parent.name == "datasets"
    print("paths.py self-test OK")
    print("  run dir :", rp.root)
    print("  cache   :", cache_dir("toy"))
    print("  raw     :", raw_data_dir("toy"))
    print("  wandb   :", rp.wandb_url_hint())
