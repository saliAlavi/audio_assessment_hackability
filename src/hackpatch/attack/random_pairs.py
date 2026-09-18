"""No-RL baseline: patch with random latent perturbation pairs at the same budget.

Pairs are ``x̂ ≻ dec(z + δ_random)`` with ``rms(δ) = ε``, for every BVCC train
utterance and ``cfg.random_pairs.n_candidates`` random directions. No detector
filter is applied (random perturbations almost never pass the ceiling rule,
so filtering would leave no pairs); the patch budget matches Hack-and-Patch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig

from hackpatch.attack.grpo_attacker import Attacker
from hackpatch.data.manifests import read_manifest
from hackpatch.models.ssl_mos import load_predictor, resolve_ckpt
from hackpatch.utils.audio import save_wav


def _make_pairs(attacker: Attacker, predictor, df: pd.DataFrame, cfg: DictConfig, out_dir: Path) -> pd.DataFrame:
    rows = []
    chunk = cfg.attack.apply.chunk_utts
    for start in range(0, len(df), chunk):
        res = attacker.apply(predictor, df.iloc[start:start + chunk], None, cfg.random_pairs.n_candidates - 1,
                             cfg.attack.apply.batch_size, cfg.attack.apply.max_seconds, random_policy=True)
        for i, r in res.iterrows():
            x, xh, y = attacker.last_audio[(r.utt_id, int(r.cand))]
            pref, rej = out_dir / f"{r.utt_id}_recon.wav", out_dir / f"{r.utt_id}_c{int(r.cand)}_random.wav"
            if not pref.exists():
                save_wav(pref, xh)
            save_wav(rej, y)
            rows.append({"utt_id": r.utt_id, "cand": int(r.cand),
                         "gain": r.score_attacked - r.score_input,
                         "preferred_path": str(pref), "rejected_path": str(rej)})
        attacker.last_audio = {}
    return pd.DataFrame(rows)


def build_random_pairs(cfg: DictConfig, rp, run) -> None:
    """Generates random-latent pairs for BVCC train/val, then patches ``init_ckpt`` with them."""
    from hackpatch.patch.trainer import train_predictor

    torch.manual_seed(cfg.seed)
    predictor = load_predictor(cfg.init_ckpt, cfg.model.pretrained)
    attacker = Attacker(cfg.attack)
    train_df = read_manifest(cfg.data.corpus, "train")
    attacker.fit_latent_stats(train_df)
    audio_dir = Path(rp.root) / "pairs_audio"
    pairs_tr = _make_pairs(attacker, predictor, train_df, cfg, audio_dir)
    val_df = read_manifest(cfg.data.corpus, "val").sample(cfg.loop.val_attack_utts, random_state=cfg.seed)
    pairs_va = _make_pairs(attacker, predictor, val_df, cfg, audio_dir)
    pairs_tr.to_csv(Path(rp.results) / "pairs_train.csv", index=False)
    pairs_va.to_csv(Path(rp.results) / "pairs_val.csv", index=False)
    run.log({"random_pairs/n_train": len(pairs_tr), "random_pairs/mean_gain": float(pairs_tr["gain"].mean()),
             "random_pairs/ceiling_rate": float((pairs_tr["gain"] > cfg.detect.tau).mean())})
    del predictor, attacker
    torch.cuda.empty_cache()
    ckpt = train_predictor(cfg, rp.checkpoints, run, init_ckpt=str(resolve_ckpt(cfg.init_ckpt)), pairs=pairs_tr, val_pairs=pairs_va)
    (Path(rp.results) / "selected_checkpoint.txt").write_text(str(ckpt))
    print(f"[selected] {ckpt}", flush=True)
