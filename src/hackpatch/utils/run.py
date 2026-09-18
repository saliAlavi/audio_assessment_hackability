"""Run initialization: seeding, run directory, provenance manifest, W&B."""
from __future__ import annotations

import os
import random
import re

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hackpatch.utils import paths
from hackpatch.utils.run_manifest import write_manifest


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and torch (CPU + CUDA).

    Args:
        seed: The declared seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)[:120]


def init_run(cfg: DictConfig):
    """Creates the scratch run dir, writes the manifest and starts W&B.

    The W&B run name equals the on-disk ``RUN_NAME`` (``{config_name}__{time}``)
    and the id is either pre-allocated by ``scripts/submit.py`` or derived
    from the name, so the URL is deterministic.

    Args:
        cfg: The resolved Hydra config. Needs ``config_name``, ``seed`` and
            ``wandb`` sections; ``run_name`` may be null.

    Returns:
        (RunPaths, wandb.Run) tuple.
    """
    import wandb

    seed_everything(int(cfg.seed))
    run_name = cfg.get("run_name") or os.environ.get("RUN_NAME") \
        or paths.make_run_name(cfg.config_name)
    rp = paths.RunPaths.create(cfg.config_name, run_name=run_name)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (rp.root / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    manifest = write_manifest(rp.root, resolved)
    run = wandb.init(
        project=paths.project_name(),
        entity=cfg.wandb.get("entity"),
        name=run_name,
        id=os.environ.get("WANDB_RUN_ID") or _sanitize(run_name),
        dir=str(rp.wandb),
        group=cfg.wandb.get("group") or cfg.config_name,
        job_type=cfg.get("phase", "main"),
        tags=[cfg.task, f"seed:{cfg.seed}", cfg.config_name],
        config=resolved,
        mode=cfg.wandb.get("mode", "online"),
        resume="allow",
    )
    run.summary["config_hash"] = manifest["config_hash"]
    run.summary["git_sha"] = manifest["git_sha"]
    print(f"[wandb] {run.url}", flush=True)
    print(f"[run_dir] {rp.root}", flush=True)
    return rp, run
