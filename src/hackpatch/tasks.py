"""Task functions dispatched by ``main.py`` via ``cfg.task``.

Each task has the signature ``task(cfg, run_paths, wandb_run)`` and writes its
outputs under ``run_paths.root``.
"""
from __future__ import annotations

import json

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from hackpatch.utils.paths import RunPaths


def train_predictor(cfg: DictConfig, rp: RunPaths, run) -> None:
    """Trains R0 or a one-shot baseline predictor (noise / PGD / fixed pair buffer)."""
    from hackpatch.patch.trainer import train_predictor as _train

    pairs = pd.concat([pd.read_csv(p) for p in cfg.train.pairs_csv]) if cfg.train.get("pairs_csv") else None
    val_pairs = pd.concat([pd.read_csv(p) for p in cfg.train.val_pairs_csv]) \
        if cfg.train.get("val_pairs_csv") else None
    ckpt = _train(cfg, rp.checkpoints, run, init_ckpt=cfg.get("init_ckpt"), pairs=pairs, val_pairs=val_pairs)
    (rp.results / "selected_checkpoint.txt").write_text(str(ckpt))
    print(f"[selected] {ckpt}", flush=True)


def eval_corr(cfg: DictConfig, rp: RunPaths, run) -> None:
    """E2: scores every eval manifest with each predictor and writes correlation reports."""
    import wandb

    from hackpatch.data.manifests import manifest_path, read_manifest
    from hackpatch.eval.correlation import correlation_report
    from hackpatch.models.ssl_mos import load_predictor
    from hackpatch.patch.trainer import predict_manifest

    rows = []
    for name, ckpt in cfg.predictors.items():
        model = load_predictor(ckpt, cfg.model.pretrained)
        for corpus, split in (s.split("/") for s in cfg.eval_sets):
            if not manifest_path(corpus, split).exists():
                print(f"[skip] missing manifest {corpus}/{split}", flush=True)
                continue
            df = read_manifest(corpus, split)
            df["pred"] = predict_manifest(model, df, cfg.eval.batch_size, cfg.eval.max_seconds)
            df.to_csv(rp.results / f"pred_{name}_{corpus}_{split}.csv", index=False)
            rep = correlation_report(df, n_boot=cfg.eval.n_boot, seed=cfg.seed)
            rows.append({"predictor": name, "corpus": corpus, "split": split, **rep})
            print(name, corpus, split, {k: round(v, 3) for k, v in rep.items() if not k.endswith(("_lo", "_hi"))},
                  flush=True)
        del model
    out = pd.DataFrame(rows)
    out.to_csv(rp.results / "correlation.csv", index=False)
    run.log({"correlation": wandb.Table(dataframe=out)})


def hack_loop(cfg: DictConfig, rp: RunPaths, run) -> None:
    """Attack → detect → patch for ``cfg.loop.rounds`` rounds (Hack-and-Patch)."""
    from hackpatch.patch.loop import run_loop

    run_loop(cfg, rp, run)


def attack_eval(cfg: DictConfig, rp: RunPaths, run) -> None:
    """E1: trains a fresh attacker against each target predictor and measures hackability."""
    from hackpatch.attack.evaluate import run_attack_eval

    run_attack_eval(cfg, rp, run)


def metric_bench(cfg: DictConfig, rp: RunPaths, run) -> None:
    """Hackability benchmark of one predictor under k freshly trained attackers."""
    from hackpatch.attack.benchmark import run_metric_benchmark

    run_metric_benchmark(cfg, rp, run)


def audio_examples(cfg: DictConfig, rp: RunPaths, run) -> None:
    """Listening examples for the project page: released attackers on redistributable audio, scored."""
    from hackpatch.attack.examples import run_audio_examples

    run_audio_examples(cfg, rp, run)


def random_pairs(cfg: DictConfig, rp: RunPaths, run) -> None:
    """Builds random-latent-perturbation pairs (the no-RL baseline's pair buffer)."""
    from hackpatch.attack.random_pairs import build_random_pairs

    build_random_pairs(cfg, rp, run)


def se_pretrain(cfg: DictConfig, rp: RunPaths, run) -> None:
    """E3: supervised pre-training of the small enhancement model."""
    from hackpatch.se.train import pretrain

    pretrain(cfg, rp, run)


def se_grpo(cfg: DictConfig, rp: RunPaths, run) -> None:
    """E3: GRPO post-training of the enhancement model with a predictor reward."""
    from hackpatch.se.train import grpo

    grpo(cfg, rp, run)


def se_eval(cfg: DictConfig, rp: RunPaths, run) -> None:
    """E3: external-metric evaluation of enhancement checkpoints."""
    from hackpatch.se.train import evaluate

    evaluate(cfg, rp, run)


def dump_config(cfg: DictConfig, rp: RunPaths, run) -> None:
    """Debug task: prints the resolved config."""
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
