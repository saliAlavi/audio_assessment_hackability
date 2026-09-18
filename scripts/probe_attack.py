#!/usr/bin/env python3
"""Probe / ε-pilot for the GRPO latent attacker (VAL only; not a paper run).

Trains a short attacker against a predictor for each ε and codec, attacks a
subset of BVCC *val*, and prints score gains plus cheap signal-level deltas.

    python scripts/probe_attack.py --ckpt R0.pt --eps 0.25,0.5,1.0 --steps 300
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from hackpatch.attack.grpo_attacker import Attacker
from hackpatch.data.manifests import read_manifest
from hackpatch.models.ssl_mos import load_predictor


def si_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref, est = ref - ref.mean(), est - est.mean()
    s = (est @ ref) / (ref @ ref + 1e-8) * ref
    return float(10 * torch.log10((s @ s) / ((est - s) @ (est - s) + 1e-8) + 1e-8))


def best_lag(ref: torch.Tensor, est: torch.Tensor, max_lag: int = 800) -> int:
    """Lag (samples) maximizing cross-correlation; positive = est delayed."""
    n = min(len(ref), len(est))
    r, e = ref[:n].cuda(), est[:n].cuda()
    xc = torch.nn.functional.conv1d(e[None, None], r[None, None, max_lag:n - max_lag]).squeeze()
    return int(xc.argmax()) - max_lag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/hack_loop.yaml")
    ap.add_argument("--eps", default="0.5")
    ap.add_argument("--codecs", default="encodec_24khz")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overrides", nargs="*", default=[], help="attack.* dotlist overrides, e.g. action_hop=0 lr=1e-3")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    predictor = load_predictor(args.ckpt, cfg.model.pretrained)
    train_df = read_manifest("bvcc", "train")
    val_df = read_manifest("bvcc", "val").sample(args.n_val, random_state=0)
    results = []
    for codec_spec in args.codecs.split(","):
        codec, _, flag = codec_spec.partition(":")  # "dac_16khz:q" = decode from quantized latent
        for eps in [float(e) for e in args.eps.split(",")]:
            acfg = OmegaConf.merge(cfg.attack, {"eps": eps, "steps": args.steps,
                                                "codec": {"name": codec, "quantize": flag == "q"}},
                                  OmegaConf.from_dotlist(args.overrides))
            torch.manual_seed(0)
            t0 = time.time()
            att = Attacker(acfg)
            ema = att.train(predictor, train_df, None, log_prefix=f"[{codec} eps={eps}] ")
            t_train = time.time() - t0
            res = att.apply(predictor, val_df, None, n_samples=0, batch_size=16)
            sd_rec = [si_sdr(x, xh) for (x, xh, y) in att.last_audio.values()]
            sd_att = [si_sdr(x, y) for (x, xh, y) in att.last_audio.values()]
            lags = [best_lag(x, xh) for (x, xh, y) in list(att.last_audio.values())[:20]]
            gain = res["score_attacked"] - res["score_input"]
            row = {"codec": codec_spec, "eps": eps, "overrides": " ".join(args.overrides), "train_s": round(t_train, 1),
                   "steps_per_s": round(args.steps / t_train, 2),
                   "train_gain_ema": ema["gain"], "val_mean_gain": float(gain.mean()),
                   "val_ceiling_rate@0.3": float((gain > 0.3).mean()),
                   "val_recon_gain": float((res["score_recon"] - res["score_input"]).mean()),
                   "sisdr_recon": float(np.mean(sd_rec)), "sisdr_attacked": float(np.mean(sd_att)),
                   "recon_lag_median": float(np.median(lags)),
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(json.dumps(row), flush=True)
            results.append(row)
            del att
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
