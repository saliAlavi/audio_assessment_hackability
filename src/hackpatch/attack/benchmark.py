"""Hackability benchmark: k freshly trained attackers against one fixed predictor.

Implements the measurement protocol the paper argues for: the ceiling rule is
referenced to the codec round trip, and hackability is summarised by the
worst case over ``k`` independently trained attackers rather than by a single
attacker's mean.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from hackpatch.attack.grpo_attacker import Attacker
from hackpatch.data.manifests import read_manifest
from hackpatch.models.metric_predictors import build_predictor


def run_metric_benchmark(cfg: DictConfig, rp, run) -> None:
    """Trains ``cfg.bench.n_attackers`` attackers against ``cfg.target_ckpt`` and scores them.

    Writes one row per (attacker seed, candidate) to ``results/attack_a{k}.csv``
    and a summary with mean and worst-case attack success rate under both
    ceiling references to ``results/bench_summary.json``.
    """
    from hackpatch.metrics.panel import Panel
    from hackpatch.patch.loop import attack_and_detect

    predictor = build_predictor(cfg.target_ckpt, cfg.model.pretrained)
    train_df = read_manifest(cfg.data.corpus, "train")
    eval_df = read_manifest(*cfg.bench.eval_set.split("/"))
    if cfg.bench.get("max_utts"):
        eval_df = eval_df.sample(min(cfg.bench.max_utts, len(eval_df)), random_state=0)

    # Phase 1: train every attacker before the panel occupies GPU memory.
    paths = []
    for k in range(cfg.bench.n_attackers):
        out = Path(rp.checkpoints) / f"attacker_a{k}.pt"
        if not out.exists():
            print(f"[bench] {cfg.target_name}: training attacker {k}", flush=True)
            torch.manual_seed(cfg.seed * 101 + k)
            att = Attacker(cfg.attack)
            stats = att.train(predictor, train_df, run, log_prefix=f"a{k}/", save_path=out)
            print(f"[bench] attacker {k} train gain {stats.get('gain', float('nan')):.3f}", flush=True)
            del att
            torch.cuda.empty_cache()
        paths.append(out)

    # Phase 2: apply each attacker to the eval set with the shared detector.
    panel = Panel(**cfg.panel)
    rows, summary = [], {}
    for k, path in enumerate(paths):
        att = Attacker(cfg.attack)
        att.load_state_dict(torch.load(path, map_location="cuda", weights_only=False))
        res = attack_and_detect(att, predictor, eval_df, panel, cfg, None, 0, panel_all=True)
        res["attacker"] = k
        res.to_csv(Path(rp.results) / f"attack_a{k}.csv", index=False)
        rows.append(res)
        del att
        torch.cuda.empty_cache()
    all_res = pd.concat(rows, ignore_index=True)
    all_res.to_csv(Path(rp.results) / "attacks_all.csv", index=False)

    per_attacker = []
    for k, res in enumerate(rows):
        m = res[res.cand == 0]
        panel_fire = m["panel"]
        per_attacker.append({
            "attacker": k,
            "asr_ref_recon": float((((m.score_attacked - m.score_recon) > cfg.detect.tau) & panel_fire).mean()),
            "asr_ref_input": float((((m.score_attacked - m.score_input) > cfg.detect.tau) & panel_fire).mean()),
            "gain_ref_recon": float((m.score_attacked - m.score_recon).mean()),
            "recon_gain": float((m.score_recon - m.score_input).mean()),
        })
    pa = pd.DataFrame(per_attacker)
    pa.to_csv(Path(rp.results) / "per_attacker.csv", index=False)
    summary = {"target": cfg.target_name, "n_attackers": len(rows), "n_utts": int(len(eval_df)),
               "tau": float(cfg.detect.tau), "eps": float(cfg.attack.eps),
               **{f"{c}_{stat}": float(getattr(pa[c], stat)()) for c in pa.columns if c != "attacker"
                  for stat in ("mean", "std", "max", "min")}}
    (Path(rp.results) / "bench_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    run.log({f"bench/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
