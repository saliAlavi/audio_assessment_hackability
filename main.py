"""Single entry point: ``python main.py --config-name <cfg> [overrides]``.

The config's ``task`` field selects what runs; every task starts a logged run
(scratch run dir + manifest + W&B) through ``hackpatch.utils.run.init_run``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hackpatch import tasks  # noqa: E402
from hackpatch.utils.run import init_run  # noqa: E402


@hydra.main(version_base=None, config_path="configs", config_name="train_r0")
def main(cfg: DictConfig) -> None:
    rp, run = init_run(cfg)
    try:
        getattr(tasks, cfg.task)(cfg, rp, run)
        (rp.root / "DONE").touch()
    finally:
        run.finish()


if __name__ == "__main__":
    main()
