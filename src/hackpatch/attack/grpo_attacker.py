"""GRPO attacker: a Gaussian policy over codec-latent perturbations.

For an input latent ``z`` the policy proposes an action ``a ~ N(μ_θ(z), σ²)`` in
per-channel-standardized latent units. To keep the policy-gradient variance
manageable the action is low-dimensional in time: one ``C``-dim vector per
``action_hop`` latent frames (linearly interpolated to frame rate), or a single
time-invariant ``C``-dim offset when ``action_hop = 0``. The perturbation
``δ = expand(a)`` is projected to an RMS budget ``ε`` (the fixed attack strength), decoded, and scored by the
target predictor. Reward: ``R(dec(z + δ)) - λ·rms(δ)²``. Advantages are
normalized within each group of ``G`` samples of the same input (GRPO).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hackpatch.attack.codecs import LatentCodec
from hackpatch.data.datasets import ManifestDataset, collate
from hackpatch.models.ssl_mos import SSLMOS
from hackpatch.utils.audio import save_wav


class LatentPolicy(nn.Module):
    """Fully-convolutional Gaussian policy over latent frames.

    Built from ``attack.policy``.

    Args:
        channels: Latent channels C.
        hidden: Hidden width.
        layers: Number of hidden conv layers.
        kernel: Temporal kernel size.
        init_log_std: Initial log σ (state-independent, per channel).
    """

    def __init__(self, channels: int, hidden: int = 256, layers: int = 4, kernel: int = 5,
                 init_log_std: float = -1.0):
        super().__init__()
        mods, c = [], channels
        for _ in range(layers):
            mods += [nn.Conv1d(c, hidden, kernel, padding=kernel // 2), nn.GELU()]
            c = hidden
        self.body = nn.Sequential(*mods)
        self.out = nn.Conv1d(c, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.log_std = nn.Parameter(torch.full((channels,), init_log_std))

    def forward(self, zn: torch.Tensor) -> torch.Tensor:
        """Returns the mean perturbation μ for standardized latents ``[B, C, F]``."""
        return self.out(self.body(zn))

    def std(self) -> torch.Tensor:
        """Per-channel σ broadcastable to ``[B, C, F]``."""
        return self.log_std.clamp(-6, 1).exp()[None, :, None]


def project_rms(delta: torch.Tensor, eps: float, frame_mask: torch.Tensor | None = None,
                sphere: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Scales each perturbation so its RMS over valid frames is at most (or exactly) ``eps``.

    Args:
        delta: ``[B, C, F]`` standardized perturbation.
        eps: RMS budget.
        frame_mask: ``[B, F]`` valid-frame mask, or None.
        sphere: If True every perturbation is rescaled to RMS exactly ``eps``
            (the policy searches directions at a fixed attack strength).

    Returns:
        (projected delta, its RMS ``[B]``).
    """
    if frame_mask is None:
        frame_mask = torch.ones(delta.shape[0], delta.shape[2], device=delta.device)
    m = frame_mask[:, None, :]
    delta = delta * m
    rms = ((delta ** 2).sum((1, 2)) / (m.sum((1, 2)) * delta.shape[1]).clamp(min=1)).sqrt()
    scale = eps / (rms + 1e-8) if sphere else torch.clamp(eps / (rms + 1e-8), max=1.0)
    return delta * scale[:, None, None], rms * scale


class Attacker:
    """Bundles codec, latent statistics and policy; trains and applies attacks.

    Args:
        cfg: The ``attack`` config section.
        device: Device.
    """

    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        self.cfg, self.device = cfg, device
        self.codec = LatentCodec(**cfg.codec).to(device)
        self.policy: LatentPolicy | None = None
        self.z_mean: torch.Tensor | None = None
        self.z_std: torch.Tensor | None = None

    # ------------------------------------------------------------------ utils
    def _frame_mask(self, lengths: torch.Tensor, n_frames: int) -> torch.Tensor:
        frames = torch.ceil(lengths.float() / self.codec.hop).long().clamp(max=n_frames)
        return (torch.arange(n_frames, device=self.device)[None] < frames[:, None].to(self.device)).float()

    @torch.no_grad()
    def fit_latent_stats(self, df: pd.DataFrame, n_utts: int = 300) -> None:
        """Per-channel latent mean/std from a sample of utterances (standardization only)."""
        ds = ManifestDataset(df.sample(min(n_utts, len(df)), random_state=0), max_seconds=4.0, cache_in_ram=False)
        zs = []
        for b in DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=4):
            z = self.codec.encode(b["wav"].to(self.device))
            m = self._frame_mask(b["lengths"], z.shape[2])
            zs.append(z.permute(0, 2, 1)[m.bool()])
        zs = torch.cat(zs)
        self.z_mean, self.z_std = zs.mean(0)[None, :, None], zs.std(0)[None, :, None].clamp(min=1e-3)

    def action_mean(self, zn: torch.Tensor, fmask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pools frame-rate policy means to action rate.

        Returns:
            (μ_a ``[B, C, A]``, action mask ``[B, A]``).
        """
        mu, m = self.policy(zn), fmask[:, None, :]
        hop = int(self.cfg.action_hop)
        if hop == 0:
            return (mu * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1), fmask[:, :1]
        pad = (-mu.shape[-1]) % hop
        mu_p, m_p = torch.nn.functional.pad(mu * m, (0, pad)), torch.nn.functional.pad(m, (0, pad))
        num = torch.nn.functional.avg_pool1d(mu_p, hop)
        den = torch.nn.functional.avg_pool1d(m_p, hop)
        return num / den.clamp(min=1e-6), (den[:, 0] > 0).float()

    def expand(self, a: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Maps an action ``[B, C, A]`` to a frame-rate perturbation ``[B, C, F]``."""
        hop = int(self.cfg.action_hop)
        if hop == 0:
            return a.expand(-1, -1, n_frames)
        up = torch.nn.functional.interpolate(a, size=a.shape[-1] * hop, mode="linear", align_corners=False)
        return up[..., :n_frames]

    def perturb(self, z: torch.Tensor, delta_n: torch.Tensor) -> torch.Tensor:
        """Maps a standardized perturbation to latent units and adds it."""
        return z + delta_n * self.z_std

    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "z_mean": self.z_mean, "z_std": self.z_std,
                "cfg": dict(self.cfg)}

    def load_state_dict(self, sd: dict) -> None:
        self.z_mean, self.z_std = sd["z_mean"].to(self.device), sd["z_std"].to(self.device)
        self.policy = LatentPolicy(self.z_std.shape[1], **self.cfg.policy).to(self.device)
        self.policy.load_state_dict(sd["policy"])

    # --------------------------------------------------------------- training
    def train(self, predictor: SSLMOS, train_df: pd.DataFrame, run=None, log_prefix: str = "",
              save_path: Path | None = None) -> dict[str, float]:
        """Trains the policy with GRPO against a frozen predictor.

        Args:
            predictor: Target predictor (frozen, eval mode).
            train_df: Manifest of attack-training utterances.
            run: W&B run or None.
            log_prefix: W&B key prefix.
            save_path: Where to store the trained attacker.

        Returns:
            Final running averages of the logged training statistics.
        """
        cfg = self.cfg
        predictor.eval().requires_grad_(False)
        self.fit_latent_stats(train_df)
        z_dim = self.z_std.shape[1]
        self.policy = LatentPolicy(z_dim, **cfg.policy).to(self.device)
        opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        loader = DataLoader(ManifestDataset(train_df, cfg.crop_seconds, train=True), batch_size=cfg.batch_size,
                            shuffle=True, collate_fn=collate, num_workers=8, drop_last=True,
                            persistent_workers=True)
        it = iter(loader)
        G, ema = cfg.group_size, {}
        for step in range(1, cfg.steps + 1):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            wav, lengths = b["wav"].to(self.device), b["lengths"].to(self.device)
            with torch.no_grad():
                z = self.codec.encode(wav)
                zn = (z - self.z_mean) / self.z_std
                fmask = self._frame_mask(lengths, z.shape[2])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    base = predictor(wav, lengths).float()
                zn_g, z_g = zn.repeat_interleave(G, 0), z.repeat_interleave(G, 0)
                len_g, fmask_g = lengths.repeat_interleave(G, 0), fmask.repeat_interleave(G, 0)
                mu, amask_g = self.action_mean(zn_g, fmask_g)
                sample = mu + self.policy.std() * torch.randn_like(mu)
                old_logp = torch.distributions.Normal(mu, self.policy.std()).log_prob(sample)
                delta, rms = project_rms(self.expand(sample, z.shape[2]), cfg.eps, fmask_g, cfg.budget_mode == "sphere")
                y = self.codec.decode(self.perturb(z_g, delta), wav.shape[1])
                y = y * (torch.arange(y.shape[1], device=self.device)[None] < len_g[:, None])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    score = predictor(y, len_g).float()
                reward = score - cfg.budget_penalty * rms ** 2
                r = reward.view(-1, G)
                adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + cfg.adv_eps)).view(-1)

            m3 = amask_g[:, None, :]
            for _ in range(cfg.ppo_epochs):
                mu_new, _ = self.action_mean(zn_g, fmask_g)
                logp = torch.distributions.Normal(mu_new, self.policy.std()).log_prob(sample)
                ratio = (logp - old_logp).exp()
                a = adv[:, None, None]
                obj = torch.minimum(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
                loss = -(obj * m3).sum() / (m3.sum() * z_dim)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
                opt.step()

            stats = {"reward": reward.mean().item(), "score": score.mean().item(),
                     "gain": (score.view(-1, G) - base[:, None]).mean().item(),
                     "gain_max": (score.view(-1, G).max(1).values - base).mean().item(),
                     "rms": rms.mean().item(), "sigma": self.policy.std().mean().item(),
                     "mu_rms": (mu ** 2).mean().sqrt().item(), "loss": loss.item()}
            for k, v in stats.items():
                ema[k] = v if k not in ema else 0.95 * ema[k] + 0.05 * v
            if run is not None and step % cfg.log_every == 0:
                run.log({f"{log_prefix}attack/{k}": v for k, v in stats.items()})
            if step % (cfg.log_every * 10) == 0:
                print(f"{log_prefix}attack step {step} " + " ".join(f"{k}={v:.3f}" for k, v in ema.items()),
                      flush=True)
            if not math.isfinite(stats["loss"]):
                raise FloatingPointError("non-finite attacker loss")
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), save_path)
        return ema

    # --------------------------------------------------------------- applying
    @torch.no_grad()
    def apply(self, predictor: SSLMOS, df: pd.DataFrame, out_dir: Path | None, n_samples: int = 0,
              batch_size: int = 16, max_seconds: float = 20.0, random_policy: bool = False) -> pd.DataFrame:
        """Attacks every utterance of a manifest (full length) and scores the results.

        Candidate 0 is the deterministic mean action; candidates ``1..n_samples``
        are stochastic policy samples. With ``random_policy`` every candidate is
        instead a random Gaussian direction at the full budget (the no-RL baseline).

        Args:
            predictor: The predictor to score with (the attack target).
            df: Manifest of utterances to attack.
            out_dir: If set, writes ``{utt_id}_recon.wav`` and ``{utt_id}_c{k}.wav``.
            n_samples: Extra stochastic candidates per utterance.
            batch_size: Utterances per batch.
            max_seconds: Truncation for long files.
            random_policy: Use random directions instead of the policy.

        Returns:
            One row per (utterance, candidate) with paths and scores:
            ``utt_id, cand, input_path, recon_path, attacked_path, score_input,
            score_recon, score_attacked, rms``. Waveforms are returned in-memory
            through ``self.last_audio`` for the detector when ``out_dir`` is None.
        """
        predictor.eval()
        df = df.reset_index(drop=True)
        order = np.argsort([0] * len(df))  # keep manifest order; loader pads within batch
        ds = ManifestDataset(df.iloc[order].reset_index(drop=True), max_seconds=max_seconds,
                             cache_in_ram=False)
        rows, audio = [], {}
        n_cand = 1 + n_samples
        for b in DataLoader(ds, batch_size=batch_size, collate_fn=collate, num_workers=8):
            wav, lengths = b["wav"].to(self.device), b["lengths"].to(self.device)
            z = self.codec.encode(wav)
            fmask = self._frame_mask(lengths, z.shape[2])
            recon = self.codec.decode(z, wav.shape[1])
            tmask = (torch.arange(wav.shape[1], device=self.device)[None] < lengths[:, None]).float()
            recon = recon * tmask
            cands = []
            if random_policy:
                n_act = 1 if int(self.cfg.action_hop) == 0 else -(-z.shape[2] // int(self.cfg.action_hop))
                for _ in range(n_cand):
                    a = torch.randn(z.shape[0], z.shape[1], n_act, device=self.device)
                    # scaled up so the projection lands exactly on the budget sphere
                    cands.append(project_rms(self.expand(a * 1e3, z.shape[2]), self.cfg.eps, fmask))
            else:
                zn = (z - self.z_mean) / self.z_std
                mu, _ = self.action_mean(zn, fmask)
                cands.append(project_rms(self.expand(mu, z.shape[2]), self.cfg.eps, fmask, self.cfg.budget_mode == "sphere"))
                for _ in range(n_samples):
                    a = mu + self.policy.std() * torch.randn_like(mu)
                    cands.append(project_rms(self.expand(a, z.shape[2]), self.cfg.eps, fmask, self.cfg.budget_mode == "sphere"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_in = predictor(wav, lengths).float()
                s_rec = predictor(recon, lengths).float()
            for k, (delta, rms) in enumerate(cands):
                y = self.codec.decode(self.perturb(z, delta), wav.shape[1]) * tmask
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s_att = predictor(y, lengths).float()
                for i, idx in enumerate(b["idx"].tolist()):
                    uid = ds.df.at[idx, "utt_id"]
                    n = int(lengths[i])
                    row = {"utt_id": uid, "cand": k, "input_path": ds.df.at[idx, "path"],
                           "score_input": s_in[i].item(), "score_recon": s_rec[i].item(),
                           "score_attacked": s_att[i].item(), "rms": rms[i].item(),
                           "recon_path": "", "attacked_path": ""}
                    if out_dir is not None:
                        rp, ap = Path(out_dir) / f"{uid}_recon.wav", Path(out_dir) / f"{uid}_c{k}.wav"
                        if k == 0:
                            save_wav(rp, recon[i, :n])
                        save_wav(ap, y[i, :n])
                        row["recon_path"], row["attacked_path"] = str(rp), str(ap)
                    audio[(uid, k)] = (wav[i, :n].cpu(), recon[i, :n].cpu(), y[i, :n].cpu())
                    rows.append(row)
        self.last_audio = audio
        return pd.DataFrame(rows)
