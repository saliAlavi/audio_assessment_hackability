"""E3 tasks: supervised pre-training, GRPO post-training and evaluation of the enhancement model.

* :func:`pretrain` – SI-SDR + compressed-magnitude L1 on VoiceBank-DEMAND.
* :func:`grpo`     – GRPO on a frozen SSL-MOS reward (R0 or R3), starting from
  the pre-trained model, with periodic external-metric evaluation on VAL.
* :func:`evaluate` – full test-set evaluation of several checkpoints with every
  predictor and the external measures, with bootstrap CIs.
"""
from __future__ import annotations

import copy
import math
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hackpatch.models.ssl_mos import load_predictor
from hackpatch.se.data import PairedCropDataset, collate_crops, fixed_subset, load_utterances, vbd_manifest
from hackpatch.se.model import BandResidualPolicy, MaskNet, flatten_rnns, load_se_model, num_parameters
from hackpatch.se.scoring import (PSEUDO_CHECKPOINTS, MetricSuite, bootstrap_summary, enhance_list,
                                  score_predictor)
from hackpatch.utils.audio import SR, save_wav

DEVICE = "cuda"


# ---------------------------------------------------------------------- losses
def time_mask(lengths: torch.Tensor, n: int) -> torch.Tensor:
    """Float mask ``[B, n]`` with 1 inside each signal."""
    return (torch.arange(n, device=lengths.device)[None] < lengths[:, None]).float()


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB for zero-padded batches (padding must be zero in both).

    Args:
        est: ``[B, T]`` estimate.
        ref: ``[B, T]`` reference.
        eps: Numerical floor.

    Returns:
        ``[B]`` SI-SDR.
    """
    alpha = (est * ref).sum(-1, keepdim=True) / ((ref ** 2).sum(-1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10((target ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps)


def supervised_loss(model: MaskNet, noisy: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor,
                    w_sisdr: float, w_mag: float, compress: float) -> tuple[torch.Tensor, dict[str, float]]:
    """``-w_sisdr · SI-SDR + w_mag · L1(|Ŝ|^c, |S|^c)`` on a crop batch.

    Args:
        model: Enhancement model.
        noisy: ``[B, T]`` noisy crops (zero beyond ``lengths``).
        clean: ``[B, T]`` clean crops.
        lengths: ``[B]`` valid sample counts.
        w_sisdr: Weight of the negative SI-SDR term.
        w_mag: Weight of the compressed-magnitude L1 term.
        compress: Magnitude compression exponent ``c``.

    Returns:
        (loss, logs).
    """
    tm = time_mask(lengths, noisy.shape[1])
    est = model(noisy) * tm
    sdr = si_sdr(est, clean * tm)
    s_est, s_ref = model.stft(est), model.stft(clean * tm)
    fm = model.frame_mask(lengths, s_ref.shape[-1])[:, None, :]
    diff = (s_est.abs() + 1e-8).pow(compress) - (s_ref.abs() + 1e-8).pow(compress)
    mag = (diff.abs() * fm).sum() / (fm.sum() * s_ref.shape[1])
    loss = -w_sisdr * sdr.mean() + w_mag * mag
    return loss, {"loss/total": loss.item(), "loss/sisdr": sdr.mean().item(), "loss/mag_l1": mag.item()}


def _lr_lambda(warmup: int, total: int, cosine: bool, final_frac: float = 0.0):
    def f(step: int) -> float:
        w = min(1.0, (step + 1) / max(1, warmup))
        if not cosine:
            return w
        prog = min(1.0, max(0, step - warmup) / max(1, total - warmup))
        return w * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
    return f


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def _pesq_stoi(args: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    from pesq import pesq
    from pystoi import stoi

    ref, deg = args
    n = min(len(ref), len(deg))
    try:
        p = pesq(SR, ref[:n], deg[:n], "wb")
    except Exception:  # PESQ raises on signals without detectable speech
        p = float("nan")
    return p, stoi(ref[:n], deg[:n], SR, extended=True)


def quick_val(model: MaskNet, noisy: list[np.ndarray], clean: list[np.ndarray], with_pesq: bool = False,
              workers: int = 8) -> dict[str, float]:
    """Validation of the supervised model on full utterances.

    Args:
        model: Enhancement model.
        noisy: Noisy utterances.
        clean: Clean references.
        with_pesq: Also compute wide-band PESQ and ESTOI (CPU, slower).
        workers: CPU processes for PESQ/ESTOI.

    Returns:
        ``val/sisdr`` (and ``val/pesq``, ``val/estoi``) means, plus per-utterance
        arrays under ``_per_utt``.
    """
    was = model.training
    model.eval()
    enh = enhance_list(model, noisy, DEVICE)
    model.train(was)
    sdr = np.array([si_sdr(torch.from_numpy(e)[None], torch.from_numpy(c[: len(e)])[None]).item()
                    for e, c in zip(enh, clean)])
    out: dict = {"val/sisdr": float(sdr.mean()), "_per_utt": {"sisdr": sdr}}
    if with_pesq:
        with Pool(workers) as pool:
            ps = np.array(pool.map(_pesq_stoi, list(zip(clean, enh))))
        out["val/pesq"], out["val/estoi"] = float(np.nanmean(ps[:, 0])), float(ps[:, 1].mean())
        out["_per_utt"].update({"pesq": ps[:, 0], "estoi": ps[:, 1]})
    return out


# -------------------------------------------------------------------- pretrain
def pretrain(cfg: DictConfig, rp, run) -> None:
    """Supervised pre-training on ``vbd_train``; selects the best ``vbd_val`` SI-SDR checkpoint.

    Reads ``cfg.model``, ``cfg.data`` and ``cfg.pretrain``. Writes
    ``checkpoints/{best,last}.pt`` and ``results/val_final.csv`` (per-utterance
    PESQ / ESTOI / SI-SDR of the best checkpoint).

    Args:
        cfg: Resolved config.
        rp: Run paths.
        run: W&B run.
    """
    tc = cfg.pretrain
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = MaskNet(**cfg.model).to(DEVICE).train()
    n_params = num_parameters(model)
    print(f"[model] MaskNet {n_params / 1e6:.2f}M parameters", flush=True)
    run.summary["model/n_params"] = n_params

    t0 = time.time()
    train_df = vbd_manifest("train")
    if cfg.data.get("max_train_utts"):
        train_df = fixed_subset(train_df, cfg.data.max_train_utts, cfg.data.subset_seed)
    ds = PairedCropDataset(train_df, cfg.data.crop_seconds, train=True)
    val_df = fixed_subset(vbd_manifest("val"), cfg.data.get("max_val_utts"), cfg.data.subset_seed)
    val_noisy, val_clean = load_utterances(val_df, "noisy"), load_utterances(val_df, "clean")
    print(f"[data] train {len(ds)} val {len(val_df)} utts loaded in {time.time() - t0:.0f}s", flush=True)
    loader = DataLoader(ds, batch_size=tc.batch_size, shuffle=True, collate_fn=collate_crops,
                        num_workers=cfg.data.num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=cfg.data.num_workers > 0)
    it = _infinite(loader)

    params = [p for n, p in model.named_parameters() if n != "log_std"]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(tc.warmup_steps, tc.steps, tc.cosine,
                                                              tc.final_lr_frac))
    best, t_last, n_skipped = -math.inf, time.time(), 0
    for step in range(1, tc.steps + 1):
        b = next(it)
        noisy, clean = b["noisy"].to(DEVICE, non_blocking=True), b["clean"].to(DEVICE, non_blocking=True)
        lengths = b["lengths"].to(DEVICE)
        loss, logs = supervised_loss(model, noisy, clean, lengths, tc.w_sisdr, tc.w_mag, tc.mag_compress)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        if not math.isfinite(logs["loss/total"]):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if torch.isfinite(gnorm):  # skip the rare step whose gradient overflows (near-silent crops)
            opt.step()
        else:
            n_skipped += 1
            logs["opt/skipped_steps"] = n_skipped
        sched.step()
        if step % tc.log_every == 0:
            dt = time.time() - t_last
            t_last = time.time()
            logs.update({"opt/grad_norm": gnorm.item(), "opt/lr": sched.get_last_lr()[0],
                         "perf/steps_per_s": tc.log_every / dt, "trainer/step": step})
            run.log(logs)
            if step % (tc.log_every * 10) == 0:
                print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in logs.items()), flush=True)
        if step % tc.eval_every == 0 or step == tc.steps:
            val = quick_val(model, val_noisy, val_clean)
            val.pop("_per_utt")
            print(f"[val] step {step} sisdr={val['val/sisdr']:.3f}", flush=True)
            run.log(val | {"trainer/step": step})
            model.save(rp.checkpoints / "last.pt", {"step": step, "val": val})
            if val["val/sisdr"] > best:
                best = val["val/sisdr"]
                model.save(rp.checkpoints / "best.pt", {"step": step, "val": val})

    best_model = load_se_model(rp.checkpoints / "best.pt", device=DEVICE)
    final = quick_val(best_model, val_noisy, val_clean, with_pesq=True, workers=cfg.data.metric_workers)
    per = pd.DataFrame({"utt_id": val_df["utt_id"], **final.pop("_per_utt")})
    per.to_csv(rp.results / "val_final.csv", index=False)
    ckpt_step = torch.load(rp.checkpoints / "best.pt", map_location="cpu", weights_only=False)["step"]
    final = {k.replace("val/", "val_final/"): v for k, v in final.items()} | {"val_final/best_step": ckpt_step}
    noisy_ref = quick_val(torch.nn.Identity(), val_noisy, val_clean, with_pesq=True,
                          workers=cfg.data.metric_workers) if tc.report_noisy else {}
    final |= {k.replace("val/", "val_noisy/"): v for k, v in noisy_ref.items() if k != "_per_utt"}
    for k, v in final.items():
        run.summary[k] = v
    pd.Series(final).to_csv(rp.results / "val_final_summary.csv", header=["value"])
    print("[final] " + " ".join(f"{k}={v:.4f}" for k, v in final.items()), flush=True)
    (rp.results / "selected_checkpoint.txt").write_text(str(rp.checkpoints / "best.pt"))


# ------------------------------------------------------------------------ GRPO
def _masked_mean(x: torch.Tensor, fmask: torch.Tensor) -> torch.Tensor:
    """Per-utterance mean of ``[B, F, T']`` over valid frames → ``[B]``."""
    m = fmask[:, None, :].expand_as(x)
    return (x * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)


def _energy_db(y: torch.Tensor | np.ndarray, x: torch.Tensor | np.ndarray) -> float:
    """Output/input energy ratio in dB."""
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    return float(10 * np.log10(((y ** 2).sum() + 1e-10) / ((x ** 2).sum() + 1e-10)))


class ValEvaluator:
    """Fixed-subset evaluation of the deterministic policy during GRPO.

    Args:
        cfg: Full config (reads ``eval``, ``metrics``, ``reward``).
        predictor: Reward predictor.
        cross: Additional fixed predictors ``{name: model}``.
    """

    def __init__(self, cfg: DictConfig, predictor, cross: dict):
        self.cfg, self.predictor, self.cross = cfg, predictor, cross
        self.df = fixed_subset(vbd_manifest(cfg.eval.split), cfg.eval.n_utts, cfg.eval.subset_seed)
        self.noisy = load_utterances(self.df, "noisy")
        self.clean = load_utterances(self.df, "clean")
        self.texts = self.df["text"].astype(str).tolist()
        self.suite = MetricSuite(**cfg.metrics, device=DEVICE)

    def score(self, wavs: list[np.ndarray]) -> pd.DataFrame:
        """All measures of a list of outputs aligned with the subset.

        Args:
            wavs: Outputs aligned with ``self.df``.

        Returns:
            Per-utterance frame.
        """
        df = pd.DataFrame({"utt_id": self.df["utt_id"]})
        df["score_reward"] = score_predictor(self.predictor, wavs, DEVICE)
        for name, p in self.cross.items():
            df[f"score_{name}"] = score_predictor(p, wavs, DEVICE)
        df["energy_db"] = [_energy_db(w, x) for w, x in zip(wavs, self.noisy)]
        return pd.concat([df, self.suite(wavs, self.clean, self.texts)], axis=1)

    @staticmethod
    def means(df: pd.DataFrame) -> dict[str, float]:
        """Column means of the numeric measures."""
        return {c: float(np.nanmean(df[c].to_numpy(dtype=float))) for c in df.columns if c != "utt_id"}


def grpo(cfg: DictConfig, rp, run) -> None:
    """GRPO post-training of the enhancement policy on a frozen predictor reward.

    For each of ``batch_size`` noisy crops the policy samples ``group_size``
    actions (full-resolution mask logits, or ERB-band logit offsets on the frozen
    pre-trained model when ``policy.action_bands > 0``, see
    :mod:`hackpatch.se.model`); each output is scored by the reward predictor. The
    reward is ``score − kl_coef · KL̂`` when ``kl_mode = reward`` (per-sample
    estimate ``mean_bins[log π(a) − log π_ref(a)]``; the analytic KL is constant
    within a group and would cancel in the advantage), and the plain score when
    ``kl_mode = loss`` (analytic KL to the reference added to the loss, as in
    DeepSeek-GRPO). Advantages are normalized within each group; the policy is
    updated with a per-bin PPO-clip objective for ``ppo_epochs`` epochs.

    Reads ``cfg.init_ckpt``, ``cfg.policy``, ``cfg.reward``, ``cfg.grpo``,
    ``cfg.data``, ``cfg.eval`` and ``cfg.metrics``. Writes a checkpoint per
    evaluation, per-utterance VAL results per evaluation, and
    ``results/val_history.csv``.

    Args:
        cfg: Resolved config.
        rp: Run paths.
        run: W&B run.
    """
    gc = cfg.grpo
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pc = cfg.policy
    sigma_kw = {k: float(pc[k]) for k in ("init_log_std", "min_log_std", "max_log_std")}
    policy = load_se_model(cfg.init_ckpt, sigma_kw, DEVICE)
    if pc.action_bands:
        policy = BandResidualPolicy(policy, n_bands=int(pc.action_bands), sr=SR, **sigma_kw).to(DEVICE)
    with torch.no_grad():
        policy.log_std.fill_(sigma_kw["init_log_std"])
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    for m in (policy, ref):  # deepcopy breaks cuDNN's flat GRU weight buffer
        flatten_rnns(m)
    policy.train()
    policy.log_std.requires_grad_(bool(gc.learn_sigma))
    print(f"[policy] {type(policy).__name__} action channels={policy.log_std.numel()} "
          f"trainable={sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6:.2f}M", flush=True)

    predictor = load_predictor(cfg.reward.ckpt, cfg.reward.pretrained, DEVICE).requires_grad_(False)
    cross = {name: load_predictor(ck, cfg.reward.pretrained, DEVICE).requires_grad_(False)
             for name, ck in (cfg.reward.get("cross_ckpts") or {}).items()}

    train_df = vbd_manifest(cfg.data.train_split)
    if cfg.data.get("max_train_utts"):
        train_df = fixed_subset(train_df, cfg.data.max_train_utts, cfg.data.subset_seed)
    ds = PairedCropDataset(train_df, gc.crop_seconds, train=True, with_clean=False)
    loader = DataLoader(ds, batch_size=gc.batch_size, shuffle=True, collate_fn=collate_crops,
                        num_workers=cfg.data.num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=cfg.data.num_workers > 0)
    it = _infinite(loader)

    groups = [{"params": [p for n, p in policy.named_parameters() if n != "log_std" and p.requires_grad],
               "lr": gc.lr}]
    if gc.learn_sigma:
        groups.append({"params": [policy.log_std], "lr": gc.lr_log_std})
    opt = torch.optim.Adam(groups, betas=tuple(gc.betas))
    trainable = [p for g in groups for p in g["params"]]

    evaluator = ValEvaluator(cfg, predictor, cross)
    history: list[dict] = []
    for name in cfg.eval.reference_rows:
        wavs = evaluator.noisy if name == "noisy" else evaluator.clean
        ref_means = ValEvaluator.means(evaluator.score(wavs))
        run.summary.update({f"val_ref_{name}/{k}": v for k, v in ref_means.items()})
        history.append({"step": -1, "system": name, **ref_means})
        print(f"[val-ref] {name} " + " ".join(f"{k}={v:.3f}" for k, v in ref_means.items()), flush=True)

    def run_eval(step: int) -> None:
        policy.eval()
        enh = enhance_list(policy, evaluator.noisy, DEVICE)
        policy.train()
        df = evaluator.score(enh)
        df.to_csv(rp.results / f"val_step{step:06d}.csv", index=False)
        m = ValEvaluator.means(df)
        base = next((h for h in history if h["system"] == "policy" and h["step"] == 0), None)
        logs = {f"val/{k}": v for k, v in m.items()}
        if base is not None:
            logs |= {f"val_delta/{k}": v - base[k] for k, v in m.items()}
        run.log(logs | {"trainer/step": step})
        history.append({"step": step, "system": "policy", **m})
        pd.DataFrame(history).to_csv(rp.results / "val_history.csv", index=False)
        policy.save(rp.checkpoints / f"step{step:06d}.pt", {"step": step, "val": m})
        policy.save(rp.checkpoints / "last.pt", {"step": step, "val": m})
        print(f"[val] step {step} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)

    run_eval(0)
    G, ema, t_last = int(gc.group_size), {}, time.time()
    for step in range(1, gc.steps + 1):
        b = next(it)
        wav, lengths = b["noisy"].to(DEVICE, non_blocking=True), b["lengths"].to(DEVICE)
        B, T = wav.shape
        with torch.no_grad():
            spec = policy.stft(wav)
            fmask = policy.frame_mask(lengths, spec.shape[-1])
            tmask = time_mask(lengths, T)
            mu_old, std_old = policy.mean_logits(spec), policy.std()
            mu_ref, std_ref = ref.mean_logits(spec), ref.std()
            spec_g, mu_g = spec.repeat_interleave(G, 0), mu_old.repeat_interleave(G, 0)
            fmask_g, len_g, tmask_g = (x.repeat_interleave(G, 0) for x in (fmask, lengths, tmask))
            action = mu_g + std_old * torch.randn_like(mu_g)
            old_lp = policy.gaussian_log_prob(action, mu_g, std_old)
            y = policy.synthesize(spec_g, action, T) * tmask_g
            y_det = policy.synthesize(spec, mu_old, T) * tmask
            with torch.autocast("cuda", dtype=torch.bfloat16):
                score = predictor(y, len_g).float()
                det_score = predictor(y_det, lengths).float()
            ref_lp = policy.gaussian_log_prob(action, mu_ref.repeat_interleave(G, 0), std_ref)
            kl_sample = _masked_mean(old_lp - ref_lp, fmask_g)
            reward = score - gc.kl_coef * kl_sample if gc.kl_mode == "reward" else score
            r = reward.view(B, G)
            adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + gc.adv_eps)).view(-1)

        m3 = fmask_g[:, None, :].expand_as(action)
        n_valid = m3.sum()
        for _ in range(gc.ppo_epochs):
            mu_new = policy.mean_logits(spec)
            std_new = policy.std()
            lp = policy.gaussian_log_prob(action, mu_new.repeat_interleave(G, 0), std_new)
            ratio = (lp - old_lp).exp()
            a = adv[:, None, None]
            obj = torch.minimum(ratio * a, ratio.clamp(1 - gc.clip, 1 + gc.clip) * a)
            loss_pg = -(obj * m3).sum() / n_valid
            kl = _masked_mean(MaskNet.gaussian_kl(mu_new, std_new, mu_ref, std_ref), fmask).mean()
            loss = loss_pg + (gc.kl_coef * kl if gc.kl_mode == "loss" else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, gc.grad_clip)
            if torch.isfinite(gnorm):
                opt.step()

        with torch.no_grad():
            clip_frac = (((ratio - 1).abs() > gc.clip).float() * m3).sum() / n_valid
            mask_det = policy.mask(spec, mu_old)
            stats = {
                "reward/mean": reward.mean().item(), "reward/group_std": r.std(1).mean().item(),
                "score/sample_mean": score.mean().item(), "score/det_mean": det_score.mean().item(),
                "score/best_of_group": score.view(B, G).max(1).values.mean().item(),
                "kl/analytic": kl.item(), "kl/sample": kl_sample.mean().item(),
                "policy/sigma_mean": policy.std().mean().item(),
                "policy/mask_mean": _masked_mean(mask_det, fmask).mean().item(),
                "policy/mask_gt1_frac": _masked_mean((mask_det > 1).float(), fmask).mean().item(),
                "policy/energy_db": (10 * torch.log10((y_det ** 2).sum(1) / ((wav ** 2).sum(1) + 1e-10)
                                                      + 1e-10)).mean().item(),
                "opt/loss": loss.item(), "opt/loss_pg": loss_pg.item(), "opt/grad_norm": gnorm.item(),
                "opt/clip_frac": clip_frac.item(), "opt/ratio_dev": ((ratio - 1).abs() * m3).sum().item() / n_valid.item(),
            }
        if not math.isfinite(stats["opt/loss"]):
            raise FloatingPointError(f"non-finite GRPO loss at step {step}")
        for k, v in stats.items():
            ema[k] = v if k not in ema else 0.95 * ema[k] + 0.05 * v
        if step % gc.log_every == 0:
            dt, t_last = time.time() - t_last, time.time()
            run.log(stats | {"perf/steps_per_s": gc.log_every / dt, "trainer/step": step})
        if step % (gc.log_every * 10) == 0:
            keys = ("reward/mean", "score/det_mean", "kl/analytic", "policy/sigma_mean", "policy/energy_db",
                    "opt/clip_frac")
            print(f"step {step} " + " ".join(f"{k}={ema[k]:.4f}" for k in keys), flush=True)
        if step % gc.eval_every == 0 or step == gc.steps:
            run_eval(step)
    (rp.results / "selected_checkpoint.txt").write_text(str(rp.checkpoints / "last.pt"))


# -------------------------------------------------------------------- evaluate
def evaluate(cfg: DictConfig, rp, run) -> None:
    """Evaluates enhancement checkpoints on the VBD test set with every measure.

    ``cfg.checkpoints`` maps a system name to a checkpoint path or to one of the
    pseudo-checkpoints ``noisy`` / ``clean`` (unprocessed input / reference).
    ``cfg.predictors`` maps a predictor name to an SSL-MOS checkpoint; each
    gives a ``score_{name}`` column. Writes ``results/per_utterance.csv``,
    ``results/summary.csv`` (mean + 95% bootstrap CI per system and metric) and
    ``results/summary_wide.csv``, and logs them as W&B tables.

    Args:
        cfg: Resolved config.
        rp: Run paths.
        run: W&B run.
    """
    import wandb

    ec = cfg.eval
    df = fixed_subset(vbd_manifest(ec.split), ec.get("n_utts"), ec.subset_seed)
    noisy, clean = load_utterances(df, "noisy"), load_utterances(df, "clean")
    texts = df["text"].astype(str).tolist()
    predictors = {name: load_predictor(ck, cfg.predictor_pretrained, DEVICE).requires_grad_(False)
                  for name, ck in cfg.predictors.items()}
    suite = MetricSuite(**cfg.metrics, device=DEVICE)
    frames = []
    for name, path in cfg.checkpoints.items():
        t0 = time.time()
        if str(path) in PSEUDO_CHECKPOINTS:
            wavs = noisy if str(path) == "noisy" else clean
        else:
            model = load_se_model(path, device=DEVICE)
            wavs = enhance_list(model, noisy, DEVICE)
            del model
        out = pd.DataFrame({"checkpoint": name, "utt_id": df["utt_id"]})
        for pname, p in predictors.items():
            out[f"score_{pname}"] = score_predictor(p, wavs, DEVICE)
        out["energy_db"] = [_energy_db(w, x) for w, x in zip(wavs, noisy)]
        out = pd.concat([out, suite(wavs, clean, texts)], axis=1)
        frames.append(out)
        if ec.save_audio_n:
            for i in range(min(ec.save_audio_n, len(wavs))):
                save_wav(rp.artifacts / "audio" / name / f"{df.at[i, 'utt_id']}.wav", wavs[i])
        means = out.drop(columns=["checkpoint", "utt_id"]).mean(numeric_only=True)
        print(f"[eval] {name} ({time.time() - t0:.0f}s) " + " ".join(f"{k}={v:.3f}" for k, v in means.items()),
              flush=True)
    per = pd.concat(frames, ignore_index=True)
    per.to_csv(rp.results / "per_utterance.csv", index=False)
    metrics = [c for c in per.columns if c not in ("checkpoint", "utt_id")]
    summary = bootstrap_summary(per, "checkpoint", metrics, n_boot=ec.n_boot, seed=cfg.seed)
    summary.to_csv(rp.results / "summary.csv", index=False)
    wide = summary.pivot(index="checkpoint", columns="metric", values="mean").reindex(list(cfg.checkpoints))
    wide.to_csv(rp.results / "summary_wide.csv")
    print(wide.round(3).to_string(), flush=True)
    run.log({"eval/summary": wandb.Table(dataframe=summary),
             "eval/summary_wide": wandb.Table(dataframe=wide.reset_index())})
