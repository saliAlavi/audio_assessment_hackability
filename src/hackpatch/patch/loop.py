"""Hack-and-Patch: attack → detect → patch rounds with a replay buffer."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from hackpatch.attack.grpo_attacker import Attacker
from hackpatch.data.manifests import read_manifest
from hackpatch.detect.detector import flag_hacks, panel_scores
from hackpatch.models.ssl_mos import load_predictor, resolve_ckpt
from hackpatch.utils.audio import save_wav


def attack_and_detect(attacker: Attacker, predictor, df: pd.DataFrame, panel, cfg: DictConfig,
                      save_dir: Path | None, n_samples: int, panel_all: bool = False,
                      pair_preferred: str = "recon", random_policy: bool = False, judge=None) -> pd.DataFrame:
    """Attacks a manifest in chunks, runs the detector, and saves flagged pairs.

    The panel is evaluated for every mean-action candidate (for reporting) and
    for every candidate that passes the ceiling rule (the only ones that can
    be hacks), unless ``panel_all``.

    Args:
        attacker: Trained attacker (or any attacker when ``random_policy``).
        predictor: Target predictor.
        df: Utterances to attack.
        panel: ``Panel`` instance.
        cfg: Full config (uses ``detect`` and ``attack.apply``).
        save_dir: Where flagged pairs' audio goes; None = don't save.
        n_samples: Stochastic candidates per utterance beyond the mean action.
        panel_all: Score every candidate with the panel.
        pair_preferred: ``recon`` (codec round trip of the input) or ``input``.
        random_policy: Random directions at the full budget instead of the policy.
        judge: Optional ``ExternalMetrics``; scores ``x̂`` and ``y`` of mean-action
            candidates into ``judge_{measure}_{recon,attacked}`` columns.

    Returns:
        One row per candidate with scores, panel measures, rule flags and, for
        flagged hacks, ``preferred_path``/``rejected_path``.
    """
    out = []
    chunk = cfg.attack.apply.chunk_utts
    for start in range(0, len(df), chunk):
        res = attacker.apply(predictor, df.iloc[start:start + chunk], None, n_samples,
                             cfg.attack.apply.batch_size, cfg.attack.apply.max_seconds, random_policy)
        audio = attacker.last_audio
        need = res["cand"].eq(0) | (res["score_attacked"] - res["score_input"] > cfg.detect.tau) \
            if not panel_all else pd.Series(True, index=res.index)
        keys = list(zip(res.loc[need, "utt_id"], res.loc[need, "cand"]))
        if keys:
            pan = panel_scores(panel, audio, keys, cfg.detect.panel_batch)
            res = res.merge(pan, on=["utt_id", "cand"], how="left")
        res = flag_hacks(res, cfg.detect)
        if judge is not None:
            jk = list(zip(res.loc[res["cand"] == 0, "utt_id"], res.loc[res["cand"] == 0, "cand"]))
            j_rec = judge([audio[k][1] for k in jk])
            j_att = judge([audio[k][2] for k in jk])
            jdf = pd.DataFrame({"utt_id": [k[0] for k in jk], "cand": [k[1] for k in jk]})
            for c in j_rec.columns:
                jdf[f"judge_{c}_recon"], jdf[f"judge_{c}_attacked"] = j_rec[c].values, j_att[c].values
            res = res.merge(jdf, on=["utt_id", "cand"], how="left")
        res["preferred_path"], res["rejected_path"] = "", ""
        if save_dir is not None:
            for i in res.index[res["hack"]]:
                uid, k = res.at[i, "utt_id"], int(res.at[i, "cand"])
                x, xh, y = audio[(uid, k)]
                rej = Path(save_dir) / f"{uid}_c{k}_attacked.wav"
                save_wav(rej, y)
                if pair_preferred == "recon":
                    pref = Path(save_dir) / f"{uid}_recon.wav"
                    if not pref.exists():
                        save_wav(pref, xh)
                else:
                    pref = res.at[i, "input_path"]
                res.at[i, "preferred_path"], res.at[i, "rejected_path"] = str(pref), str(rej)
        out.append(res)
        attacker.last_audio = {}
        print(f"  attacked {min(start + chunk, len(df))}/{len(df)} utts, hacks so far "
              f"{sum(int(r['hack'].sum()) for r in out)}", flush=True)
    return pd.concat(out, ignore_index=True)


def pgd_attack_and_detect(predictor, df: pd.DataFrame, panel, cfg: DictConfig, pcfg: DictConfig,
                          save_dir: Path | None) -> pd.DataFrame:
    """Waveform-PGD counterpart of :func:`attack_and_detect` (reference = the input itself)."""
    from torch.utils.data import DataLoader

    from hackpatch.data.datasets import ManifestDataset, collate
    from hackpatch.patch.trainer import pgd_attack

    ds = ManifestDataset(df.reset_index(drop=True), max_seconds=cfg.attack.apply.max_seconds, cache_in_ram=False)
    out = []
    rows, audio = [], {}
    for b in DataLoader(ds, batch_size=pcfg.batch_size, collate_fn=collate, num_workers=8):
        wav, lengths = b["wav"].cuda(), b["lengths"].cuda()
        adv = pgd_attack(predictor, wav, lengths, pcfg.eps, pcfg.steps, pcfg.step_size)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s_in = predictor(wav, lengths).float()
            s_adv = predictor(adv, lengths).float()
        for i, idx in enumerate(b["idx"].tolist()):
            uid, n = ds.df.at[idx, "utt_id"], int(lengths[i])
            audio[(uid, 0)] = (wav[i, :n].cpu(), wav[i, :n].cpu(), adv[i, :n].cpu())
            rows.append({"utt_id": uid, "cand": 0, "input_path": ds.df.at[idx, "path"],
                         "score_input": s_in[i].item(), "score_recon": s_in[i].item(),
                         "score_attacked": s_adv[i].item(), "rms": float("nan")})
        if len(rows) >= cfg.attack.apply.chunk_utts:
            out.append(_finish_pgd_chunk(pd.DataFrame(rows), audio, panel, cfg, save_dir))
            rows, audio = [], {}
    if rows:
        out.append(_finish_pgd_chunk(pd.DataFrame(rows), audio, panel, cfg, save_dir))
    return pd.concat(out, ignore_index=True)


def _finish_pgd_chunk(res: pd.DataFrame, audio: dict, panel, cfg: DictConfig, save_dir: Path | None) -> pd.DataFrame:
    """Scores one PGD chunk with the panel, flags hacks and saves the flagged pairs."""
    res = res.merge(panel_scores(panel, audio, list(zip(res.utt_id, res.cand)), cfg.detect.panel_batch),
                    on=["utt_id", "cand"], how="left")
    res = flag_hacks(res, cfg.detect)
    res["preferred_path"], res["rejected_path"] = "", ""
    if save_dir is not None:
        for i in res.index[res["hack"]]:
            uid = res.at[i, "utt_id"]
            rej = Path(save_dir) / f"{uid}_pgd.wav"
            save_wav(rej, audio[(uid, 0)][2])
            res.at[i, "preferred_path"], res.at[i, "rejected_path"] = res.at[i, "input_path"], str(rej)
    print(f"  pgd: {len(res)} utts, {int(res['hack'].sum())} hacks", flush=True)
    return res


def summarize(res: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Round-level statistics on the mean-action candidate and on all candidates."""
    mean_act = res[res["cand"] == 0]
    stats = {
        f"{prefix}n_utts": float(len(mean_act)),
        f"{prefix}n_candidates": float(len(res)),
        f"{prefix}asr": float(mean_act["hack"].mean()),
        f"{prefix}ceiling_rate": float(mean_act["ceiling"].mean()),
        f"{prefix}mean_gain": float(mean_act["gain"].mean()),
        f"{prefix}mean_gain_vs_recon": float((mean_act["score_attacked"] - mean_act["score_recon"]).mean()),
        f"{prefix}recon_gain": float((mean_act["score_recon"] - mean_act["score_input"]).mean()),
        f"{prefix}any_cand_hack_rate": float(res.groupby("utt_id")["hack"].any().mean()),
        f"{prefix}n_hacks": float(res["hack"].sum()),
    }
    for m in ("estoi", "sisdr", "wer", "sbs"):
        if f"{m}_attacked" in mean_act:
            stats[f"{prefix}{m}_delta"] = float((mean_act[f"{m}_attacked"] - mean_act[f"{m}_recon"]).mean())
    return stats


def run_loop(cfg: DictConfig, rp, run) -> None:
    """Runs ``cfg.loop.rounds`` rounds; resumable via per-round ``done.json`` markers.

    Round k: for every attack space in ``cfg.loop.attack_spaces`` train a fresh
    attacker against ``R_{k-1}`` on BVCC train (or run waveform PGD), attack
    BVCC train/val, flag hacks, add pairs to the replay buffer, and fine-tune
    ``R_{k-1}`` → ``R_k`` with MSE + margin ranking over the whole buffer.
    """
    from hackpatch.metrics.panel import Panel
    from hackpatch.patch.trainer import train_predictor

    train_df = read_manifest(cfg.data.corpus, "train")
    val_df = read_manifest(cfg.data.corpus, "val")
    attack_train_df = train_df.sample(cfg.loop.max_train_utts, random_state=cfg.seed) \
        if cfg.loop.get("max_train_utts") else train_df
    if cfg.loop.get("val_attack_utts"):
        val_df = val_df.sample(min(cfg.loop.val_attack_utts, len(val_df)), random_state=cfg.seed)
    prev_ckpt = str(resolve_ckpt(cfg.init_ckpt))
    buf_train, buf_val = [], []
    history = []
    for k in range(1, cfg.loop.rounds + 1):
        rd = Path(rp.root) / f"round{k}"
        rd.mkdir(exist_ok=True)
        done = rd / "done.json"
        if done.exists():
            info = json.loads(done.read_text())
            prev_ckpt = info["ckpt"]
            buf_train.append(pd.read_csv(rd / "pairs_train.csv"))
            buf_val.append(pd.read_csv(rd / "pairs_val.csv"))
            history.append(info)
            print(f"[round {k}] already done -> {prev_ckpt}", flush=True)
            continue

        print(f"[round {k}] attackers vs {prev_ckpt}", flush=True)
        predictor = load_predictor(prev_ckpt, cfg.model.pretrained)
        panel = Panel(**cfg.panel)
        stats, pairs_tr_all, pairs_va_all = {}, [], []
        cols = ["utt_id", "cand", "gain", "preferred_path", "rejected_path"]
        for si, space in enumerate(cfg.loop.attack_spaces):
            torch.manual_seed(cfg.seed * 1000 + k * 10 + si)
            name = space.name
            if space.type == "grpo":
                acfg = OmegaConf.merge(cfg.attack, space.get("attack_overrides") or {})
                sub = OmegaConf.merge(cfg, {"attack": acfg})
                attacker = Attacker(acfg)
                attacker.train(predictor, train_df, run, log_prefix=f"round{k}/{name}/",
                               save_path=rd / f"attacker_{name}.pt")
                res_tr = attack_and_detect(attacker, predictor, attack_train_df, panel, sub, rd / "pairs_audio",
                                           cfg.attack.apply.n_samples, pair_preferred=cfg.loop.pair_preferred)
                res_va = attack_and_detect(attacker, predictor, val_df, panel, sub, rd / "pairs_audio",
                                           cfg.attack.apply.n_samples, pair_preferred=cfg.loop.pair_preferred)
                del attacker
                torch.cuda.empty_cache()
            elif space.type == "pgd":
                res_tr = pgd_attack_and_detect(predictor, attack_train_df, panel, cfg, space.pgd, rd / "pairs_audio")
                res_va = pgd_attack_and_detect(predictor, val_df, panel, cfg, space.pgd, rd / "pairs_audio")
            else:
                raise ValueError(f"unknown attack space type {space.type}")
            res_tr.to_csv(rd / f"attack_train_{name}.csv", index=False)
            res_va.to_csv(rd / f"attack_val_{name}.csv", index=False)
            pairs_tr_all.append(res_tr.loc[res_tr["hack"], cols].assign(round=k, space=name))
            pairs_va_all.append(res_va.loc[res_va["hack"], cols].assign(round=k, space=name))
            stats |= summarize(res_tr, f"round{k}/{name}/train_") | summarize(res_va, f"round{k}/{name}/val_")
        pairs_tr = pd.concat(pairs_tr_all, ignore_index=True)
        pairs_va = pd.concat(pairs_va_all, ignore_index=True)
        pairs_tr.to_csv(rd / "pairs_train.csv", index=False)
        pairs_va.to_csv(rd / "pairs_val.csv", index=False)
        buf_train.append(pairs_tr)
        buf_val.append(pairs_va)
        first = cfg.loop.attack_spaces[0].name
        stats |= {f"round{k}/train_asr": stats[f"round{k}/{first}/train_asr"],
                  f"round{k}/val_asr": stats[f"round{k}/{first}/val_asr"],
                  f"round{k}/val_mean_gain": stats[f"round{k}/{first}/val_mean_gain"],
                  f"round{k}/val_ceiling_rate": stats[f"round{k}/{first}/val_ceiling_rate"]}
        print(json.dumps(stats, indent=1), flush=True)
        run.log(stats)
        del predictor, panel
        gc.collect()
        torch.cuda.empty_cache()

        buffer = pd.concat(buf_train, ignore_index=True)
        val_buffer = pd.concat(buf_val, ignore_index=True)
        if len(buffer) == 0:
            print(f"[round {k}] no hacks found; predictor unchanged", flush=True)
            new_ckpt = prev_ckpt
        else:
            patch_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            patch_cfg.train = cfg.patch
            new_ckpt = str(train_predictor(patch_cfg, rd / "predictor", run, init_ckpt=prev_ckpt,
                                           pairs=buffer, val_pairs=val_buffer if len(val_buffer) else None,
                                           log_prefix=f"round{k}/patch/"))
        info = {"round": k, "ckpt": new_ckpt, "prev_ckpt": prev_ckpt, "n_pairs_round": len(pairs_tr),
                "n_pairs_buffer": len(buffer), **stats}
        done.write_text(json.dumps(info, indent=1))
        history.append(info)
        prev_ckpt = new_ckpt
    (Path(rp.results) / "loop_history.json").write_text(json.dumps(history, indent=1))
    (Path(rp.results) / "final_checkpoint.txt").write_text(str(prev_ckpt))
    print(f"[loop] final predictor {prev_ckpt}", flush=True)
