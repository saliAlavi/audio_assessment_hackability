"""E1: hackability of a predictor under several attackers at a fixed budget.

For one target predictor, each configured attacker is run on the attack-eval
set (BVCC test audio; labels unused) and scored with the same detector:

* ``grpo``     – fresh GRPO latent attacker trained against the target (EnCodec or DAC).
* ``transfer`` – a stored attacker (e.g. trained against R0) applied without retraining.
* ``pgd``      – white-box L-inf waveform PGD on the target.
* ``random``   – random latent directions at the budget (no optimization; floor).
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from hackpatch.attack.grpo_attacker import Attacker
from hackpatch.data.datasets import ManifestDataset, collate
from hackpatch.data.manifests import read_manifest
from hackpatch.detect.detector import flag_hacks, panel_scores
from hackpatch.models.ssl_mos import load_predictor
from hackpatch.patch.loop import attack_and_detect, summarize
from hackpatch.patch.trainer import pgd_attack


def pgd_and_detect(predictor, df: pd.DataFrame, panel, cfg: DictConfig, acfg: DictConfig) -> tuple[pd.DataFrame, dict]:
    """Waveform PGD attack + detection. The "reconstruction" is the input itself."""
    ds = ManifestDataset(df.reset_index(drop=True), max_seconds=cfg.attack.apply.max_seconds, cache_in_ram=False)
    rows, audio = [], {}
    for b in DataLoader(ds, batch_size=acfg.batch_size, collate_fn=collate, num_workers=8):
        wav, lengths = b["wav"].cuda(), b["lengths"].cuda()
        adv = pgd_attack(predictor, wav, lengths, acfg.eps, acfg.steps, acfg.step_size)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s_in = predictor(wav, lengths).float()
            s_adv = predictor(adv, lengths).float()
        for i, idx in enumerate(b["idx"].tolist()):
            uid, n = ds.df.at[idx, "utt_id"], int(lengths[i])
            audio[(uid, 0)] = (wav[i, :n].cpu(), wav[i, :n].cpu(), adv[i, :n].cpu())
            rows.append({"utt_id": uid, "cand": 0, "input_path": ds.df.at[idx, "path"],
                         "score_input": s_in[i].item(), "score_recon": s_in[i].item(),
                         "score_attacked": s_adv[i].item(), "rms": float("nan")})
    res = pd.DataFrame(rows)
    res = res.merge(panel_scores(panel, audio, list(zip(res.utt_id, res.cand)), cfg.detect.panel_batch),
                    on=["utt_id", "cand"])
    return flag_hacks(res, cfg.detect), audio


def run_attack_eval(cfg: DictConfig, rp, run) -> None:
    """Runs every attacker in ``cfg.attackers`` against ``cfg.target_ckpt``.

    Writes ``results/attack_{name}.csv`` (per utterance) and ``results/e1_summary.json``.
    """
    import wandb

    from hackpatch.metrics.external import ExternalMetrics
    from hackpatch.metrics.panel import Panel

    predictor = load_predictor(cfg.target_ckpt, cfg.model.pretrained)
    train_df = read_manifest(cfg.data.corpus, "train")
    eval_df = read_manifest(*cfg.e1.eval_set.split("/"))
    if cfg.e1.get("max_utts"):
        eval_df = eval_df.sample(min(cfg.e1.max_utts, len(eval_df)), random_state=0)
    specs = list(cfg.attackers)
    if cfg.get("transfer_attacker_ckpt"):
        specs.append(OmegaConf.create({"name": "transfer_r0_round1", "type": "transfer",
                                       "attacker_ckpt": cfg.transfer_attacker_ckpt}))

    # Phase 1: train every learned attacker before the evaluation models occupy GPU memory.
    trained: dict[str, Path] = {}
    for spec in specs:
        if spec.type != "grpo":
            continue
        path = Path(rp.checkpoints) / f"attacker_{spec.name}.pt"
        if not path.exists():
            print(f"[E1] target={cfg.target_name} training attacker={spec.name}", flush=True)
            torch.manual_seed(cfg.seed * 7919 + zlib.crc32(spec.name.encode()) % 1000)
            attacker = Attacker(OmegaConf.merge(cfg.attack, spec.get("attack_overrides") or {}))
            attacker.train(predictor, train_df, run, log_prefix=f"{spec.name}/", save_path=path)
            del attacker
            torch.cuda.empty_cache()
        trained[spec.name] = path

    # Phase 2: apply every attacker and score with the shared detector and judges.
    panel = Panel(**cfg.panel)
    judge = ExternalMetrics(measures=tuple(cfg.e1.judges)) if cfg.e1.get("judges") else None
    free, total = torch.cuda.mem_get_info()
    print(f"[E1] GPU memory after loading panel/judges: {(total - free) / 1e9:.1f} GB used of {total / 1e9:.1f}", flush=True)
    summary = {}
    for spec in specs:
        name = spec.name
        print(f"[E1] target={cfg.target_name} attacker={name}", flush=True)
        torch.manual_seed(cfg.seed * 7919 + zlib.crc32(name.encode()) % 1000)
        if spec.type in ("grpo", "transfer", "random"):
            acfg = OmegaConf.merge(cfg.attack, spec.get("attack_overrides") or {})
            attacker = Attacker(acfg)
            if spec.type in ("grpo", "transfer"):
                ckpt = trained[name] if spec.type == "grpo" else spec.attacker_ckpt
                attacker.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False))
            else:
                attacker.fit_latent_stats(train_df)
            sub = OmegaConf.merge(cfg, {"attack": acfg})
            res = attack_and_detect(attacker, predictor, eval_df, panel, sub, None, 0, panel_all=True,
                                    random_policy=spec.type == "random", judge=judge)
            del attacker
        elif spec.type == "pgd":
            res, audio = pgd_and_detect(predictor, eval_df, panel, cfg, spec.pgd)
            if judge is not None:
                keys = list(zip(res.utt_id, res.cand))
                j_rec = judge([audio[k][1] for k in keys])
                j_att = judge([audio[k][2] for k in keys])
                for c in j_rec.columns:
                    res[f"judge_{c}_recon"], res[f"judge_{c}_attacked"] = j_rec[c].values, j_att[c].values
            del audio
        else:
            raise ValueError(spec.type)
        res.to_csv(Path(rp.results) / f"attack_{name}.csv", index=False)
        stats = summarize(res, "")
        for c in [c for c in res.columns if c.startswith("judge_") and c.endswith("_attacked")]:
            base = c[len("judge_"):-len("_attacked")]
            stats[f"judge_{base}_delta"] = float((res[c] - res[f"judge_{base}_recon"]).mean())
        summary[name] = stats
        print(json.dumps({name: stats}, indent=1), flush=True)
        run.log({f"e1/{name}/{k}": v for k, v in stats.items()})
        torch.cuda.empty_cache()
    (Path(rp.results) / "e1_summary.json").write_text(json.dumps(
        {"target": cfg.target_name, "target_ckpt": cfg.target_ckpt, "attackers": summary}, indent=1))
    run.log({"e1_table": wandb.Table(dataframe=pd.DataFrame(summary).T.reset_index(names="attacker"))})

