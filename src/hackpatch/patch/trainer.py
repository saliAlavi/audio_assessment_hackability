"""Predictor training: plain MOS regression (R0) and patching with ranking pairs.

One loop serves every predictor variant; the config decides the pair source:

* ``none``   – MSE on BVCC only (R0).
* ``buffer`` – margin ranking loss over stored preference pairs (hack replay
  buffer for Hack-and-Patch, or random-latent pairs for the no-RL baseline).
* ``noise``  – online pairs ``x ≻ x + Gaussian noise`` (noise-augmentation baseline).
* ``pgd``    – online pairs ``x ≻ PGD_adv(x)`` in the waveform domain
  (waveform adversarial-training baseline).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hackpatch.data.datasets import ManifestDataset, PairDataset, collate, collate_pairs
from hackpatch.data.manifests import read_manifest
from hackpatch.models.ssl_mos import SSLMOS, resolve_ckpt


@torch.no_grad()
def predict_manifest(model: SSLMOS, df: pd.DataFrame, batch_size: int = 16,
                     max_seconds: float | None = None, device: str = "cuda") -> np.ndarray:
    """Scores every utterance of a manifest (length-sorted batches, bf16).

    Args:
        model: Predictor.
        df: Manifest frame.
        batch_size: Utterances per forward pass.
        max_seconds: Optional truncation for very long files.
        device: Device.

    Returns:
        Predictions aligned with ``df`` rows.
    """
    was_training = model.training
    model.eval()
    ds = ManifestDataset(df, max_seconds=max_seconds, train=False, cache_in_ram=False)
    preds = np.zeros(len(df), dtype=np.float32)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate, num_workers=8)
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(batch["wav"].to(device), batch["lengths"].to(device))
        preds[batch["idx"].numpy()] = p.float().cpu().numpy()
    model.train(was_training)
    return preds


def ranking_loss(pos: torch.Tensor, neg: torch.Tensor, margin: float) -> torch.Tensor:
    """Margin ranking loss ``max(0, margin - (s_pos - s_neg))``."""
    return F.relu(margin - (pos - neg)).mean()


def _add_noise(wav: torch.Tensor, lengths: torch.Tensor, snr_db: tuple[float, float]) -> torch.Tensor:
    """Adds white Gaussian noise at a per-utterance SNR drawn uniformly from ``snr_db``."""
    mask = (torch.arange(wav.shape[1], device=wav.device)[None] < lengths[:, None].to(wav.device)).float()
    power = (wav ** 2 * mask).sum(1) / mask.sum(1)
    snr = torch.empty(wav.shape[0], device=wav.device).uniform_(*snr_db)
    noise_power = power / (10 ** (snr / 10))
    return (wav + torch.randn_like(wav) * noise_power.sqrt()[:, None]) * mask


def pgd_attack(model: SSLMOS, wav: torch.Tensor, lengths: torch.Tensor, eps: float,
               steps: int, step_size: float) -> torch.Tensor:
    """L-inf PGD in the waveform domain that maximizes the predicted score.

    Args:
        model: Predictor (its parameters get no gradient from this call).
        wav: [B, T] clean batch.
        lengths: [B] valid lengths.
        eps: L-inf budget on the perturbation (waveform units).
        steps: PGD iterations.
        step_size: Signed-gradient step size.

    Returns:
        Adversarial batch, detached.
    """
    was_training = model.training
    model.eval()
    mask = (torch.arange(wav.shape[1], device=wav.device)[None] < lengths[:, None].to(wav.device)).float()
    delta = torch.empty_like(wav).uniform_(-eps, eps) * mask
    for _ in range(steps):
        delta.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            score = model(wav + delta, lengths).float().sum()
        (grad,) = torch.autograd.grad(score, delta)
        delta = (delta.detach() + step_size * grad.sign()).clamp(-eps, eps) * mask
    model.train(was_training)
    return (wav + delta).clamp(-1, 1).detach()


def evaluate_val(model: SSLMOS, val_df: pd.DataFrame, val_pairs: pd.DataFrame | None,
                 cfg: DictConfig, device: str) -> dict[str, float]:
    """Validation metrics: BVCC val agreement and (optionally) hack-pair accuracy."""
    val_df = val_df.copy()
    val_df["pred"] = predict_manifest(model, val_df, cfg.eval.batch_size, device=device)
    out = {f"val/{k}": v for k, v in _point_report(val_df).items()}
    if val_pairs is not None and len(val_pairs):
        pos = val_pairs.rename(columns={"preferred_path": "path"}).assign(mos=np.nan, system="x")
        neg = val_pairs.rename(columns={"rejected_path": "path"}).assign(mos=np.nan, system="x")
        sp = predict_manifest(model, pos, cfg.eval.batch_size, device=device)
        sn = predict_manifest(model, neg, cfg.eval.batch_size, device=device)
        out["val/pair_acc"] = float(np.mean(sp > sn))
        out["val/pair_gap"] = float(np.mean(sp - sn))
    return out


def _point_report(df: pd.DataFrame) -> dict[str, float]:
    from scipy import stats
    sys_df = df.groupby("system")[["pred", "mos"]].mean()
    return {
        "utt_mse": float(np.mean((df["pred"] - df["mos"]) ** 2)),
        "utt_srcc": float(stats.spearmanr(df["pred"], df["mos"])[0]),
        "utt_lcc": float(stats.pearsonr(df["pred"], df["mos"])[0]),
        "sys_srcc": float(stats.spearmanr(sys_df["pred"], sys_df["mos"])[0]),
        "sys_lcc": float(stats.pearsonr(sys_df["pred"], sys_df["mos"])[0]),
        "sys_mse": float(np.mean((sys_df["pred"] - sys_df["mos"]) ** 2)),
    }


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def train_predictor(cfg: DictConfig, out_dir: Path, run=None, init_ckpt: str | None = None,
                    pairs: pd.DataFrame | None = None, val_pairs: pd.DataFrame | None = None,
                    log_prefix: str = "") -> Path:
    """Trains or patches an SSL-MOS predictor.

    Reads ``cfg.model``, ``cfg.data``, ``cfg.train`` (optimizer, steps, loss
    weights, pair source) and ``cfg.eval``.

    Args:
        cfg: Resolved config.
        out_dir: Where ``best.pt`` / ``last.pt`` go.
        run: Active W&B run or None.
        init_ckpt: Checkpoint to start from (patching); None = fresh R0.
        pairs: Preference pairs for ``pair_source=buffer``.
        val_pairs: Held-out pairs for validation pair accuracy.
        log_prefix: Prefix for W&B keys (e.g. ``round2/``).

    Returns:
        Path of the checkpoint selected by ``cfg.train.select`` (``best`` or ``last``).
    """
    device = "cuda"
    tc = cfg.train
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SSLMOS(**cfg.model).to(device)
    if init_ckpt:
        model.load(resolve_ckpt(init_ckpt))
    model.train()

    train_df = read_manifest(cfg.data.corpus, "train")
    val_df = read_manifest(cfg.data.corpus, "val")
    mos_loader = DataLoader(ManifestDataset(train_df, cfg.data.max_seconds, train=True),
                            batch_size=tc.batch_size, shuffle=True, collate_fn=collate,
                            num_workers=cfg.data.num_workers, drop_last=True, persistent_workers=True)
    mos_iter = _infinite(mos_loader)
    pair_iter = None
    if tc.pair_source == "buffer":
        if pairs is None or not len(pairs):
            raise ValueError("pair_source=buffer needs a non-empty pair set")
        pair_loader = DataLoader(PairDataset(pairs, cfg.data.max_seconds), batch_size=tc.pair_batch_size,
                                 shuffle=True, collate_fn=collate_pairs, num_workers=cfg.data.num_workers,
                                 drop_last=len(pairs) >= tc.pair_batch_size)
        pair_iter = _infinite(pair_loader)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": tc.lr_encoder},
                             {"params": model.head.parameters(), "lr": tc.lr_head}],
                            weight_decay=tc.weight_decay)
    warmup = int(tc.warmup_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * (
            0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warmup) / max(1, tc.steps - warmup))))
            if tc.cosine else 1.0))

    best, best_path, last_path = math.inf, out_dir / "best.pt", out_dir / "last.pt"
    for step in range(1, tc.steps + 1):
        batch = next(mos_iter)
        wav, lengths, mos = batch["wav"].to(device), batch["lengths"].to(device), batch["mos"].to(device)
        logs = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(wav, lengths).float()
        loss_mse = F.mse_loss(pred, mos)
        loss = tc.w_mse * loss_mse
        logs["loss/mse"] = loss_mse.item()

        if tc.pair_source != "none" and tc.w_rank > 0:
            if tc.pair_source == "buffer":
                pb = next(pair_iter)
                pos_w, neg_w = pb["pos"].to(device), pb["neg"].to(device)
                pos_l, neg_l = pb["pos_len"].to(device), pb["neg_len"].to(device)
            elif tc.pair_source == "noise":
                pos_w, pos_l = wav, lengths
                neg_w, neg_l = _add_noise(wav, lengths, tuple(tc.noise_snr_db)), lengths
            elif tc.pair_source == "pgd":
                pos_w, pos_l = wav, lengths
                neg_w = pgd_attack(model, wav, lengths, tc.pgd.eps, tc.pgd.steps, tc.pgd.step_size)
                neg_l = lengths
            else:
                raise ValueError(f"unknown pair_source {tc.pair_source}")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(torch.cat([pos_w, neg_w]), torch.cat([pos_l, neg_l])).float()
            s_pos, s_neg = s[: len(pos_w)], s[len(pos_w):]
            loss_rank = ranking_loss(s_pos, s_neg, tc.margin)
            loss = loss + tc.w_rank * loss_rank
            logs.update({"loss/rank": loss_rank.item(), "pairs/gap": (s_pos - s_neg).mean().item(),
                         "pairs/acc": (s_pos > s_neg).float().mean().item()})

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        sched.step()
        logs.update({"loss/total": loss.item(), "opt/grad_norm": gnorm.item(),
                     "opt/lr_encoder": sched.get_last_lr()[0], "trainer/step": step})
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"non-finite loss at step {step}")

        if step % tc.log_every == 0 and run is not None:
            run.log({f"{log_prefix}{k}": v for k, v in logs.items()})
        if step % tc.eval_every == 0 or step == tc.steps:
            val = evaluate_val(model, val_df, val_pairs, cfg, device)
            print(f"{log_prefix}step {step} " + " ".join(f"{k}={v:.4f}" for k, v in val.items()), flush=True)
            if run is not None:
                run.log({f"{log_prefix}{k}": v for k, v in val.items()} | {f"{log_prefix}trainer/step": step})
            model.save(last_path, {"step": step, "val": val})
            if val["val/utt_mse"] < best:
                best = val["val/utt_mse"]
                model.save(best_path, {"step": step, "val": val})
    return best_path if tc.select == "best" else last_path
