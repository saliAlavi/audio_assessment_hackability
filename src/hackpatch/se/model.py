"""Small STFT mask-based speech enhancement network and its Gaussian policies for GRPO.

Signal path (16 kHz): ``STFT(noisy) → log-power features → Conv1d → (Bi)GRU →
Linear → mask logits μ ∈ R^{F×T} → mask = mask_max · sigmoid(logits) →
mask · noisy STFT (noisy phase) → iSTFT``. With ``mask_max = 2`` a logit of 0
is the identity mask and the enhancer may amplify bins by up to +6 dB, leaving
room for reward hacking (e.g. spectral boosting) that a [0, 1] mask forbids.

Two Gaussian policies share :class:`GaussianPolicyMixin` (diagonal Gaussian with
a learned, state-independent ``log σ`` per action channel; the deterministic
output uses the mean):

* :class:`MaskNet` itself (``policy.action_bands = 0``): actions are the full
  F×T mask logits. Exact, but exploration lives in ~10^5 dimensions for a 3 s
  crop, so the policy-gradient signal of one utterance-level reward is weak.
* :class:`BandResidualPolicy` (``policy.action_bands = K``): the pre-trained
  network is frozen and provides base logits; a trainable copy of it with a
  K-channel head outputs per-frame ERB-band logit offsets
  ``a ∈ R^{K×T}``, which are interpolated to the F bins and added:
  ``logits = μ_base(x) + U·a``. The policy density is exact in the K×T space
  (``K ≪ F``), exploration is spectrally smooth (the directions a predictor
  can be hacked along, e.g. tilt, over-suppression, loudness), and the head is
  zero-initialized so the initial deterministic output equals the pre-trained
  model.

PPO ratios are taken per action element (token-level GRPO).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOG_2PI = math.log(2 * math.pi)


class GaussianPolicyMixin:
    """Diagonal Gaussian policy helpers; needs ``log_std``, ``min_log_std`` and ``max_log_std``."""

    def std(self) -> torch.Tensor:
        """Clamped per-channel σ broadcastable to actions ``[B, C, T']``."""
        return self.log_std.clamp(self.min_log_std, self.max_log_std).exp()[None, :, None]

    def sample(self, mu: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draws an action ``a = μ + σ·ε``.

        Args:
            mu: Policy mean ``[B, F, T']``.
            generator: Optional RNG.

        Returns:
            Sampled logits, detached.
        """
        eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        return (mu + self.std().detach() * eps).detach()

    @staticmethod
    def gaussian_log_prob(action: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Element-wise diagonal-Gaussian log-density.

        Args:
            action: ``[B, F, T']``.
            mu: Mean, same shape.
            std: Standard deviation broadcastable to ``mu``.

        Returns:
            ``[B, F, T']`` log-densities.
        """
        return -((action - mu) ** 2) / (2 * std ** 2) - std.log() - 0.5 * LOG_2PI

    def log_prob(self, action: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Element-wise log π(a | x) under this policy's current σ."""
        return self.gaussian_log_prob(action, mu, self.std())

    @staticmethod
    def gaussian_kl(mu_p: torch.Tensor, std_p: torch.Tensor, mu_q: torch.Tensor, std_q: torch.Tensor) -> torch.Tensor:
        """Element-wise ``KL(N(μ_p, σ_p²) || N(μ_q, σ_q²))``."""
        return torch.log(std_q / std_p) + (std_p ** 2 + (mu_p - mu_q) ** 2) / (2 * std_q ** 2) - 0.5


class MaskNet(GaussianPolicyMixin, nn.Module):
    """STFT-domain mask estimator with a Gaussian policy over mask logits.

    Built from the ``model`` config section.

    Args:
        n_fft: FFT size (Hann window of the same length).
        hop: Hop size in samples.
        hidden: Width of the Conv1d projection.
        rnn_hidden: GRU hidden size per direction.
        rnn_layers: Number of GRU layers.
        bidirectional: Bidirectional GRU (non-causal) if True.
        conv_kernel: Temporal kernel of the input Conv1d.
        mask_max: Upper bound of the mask (``mask_max · sigmoid``).
        init_log_std: Initial per-frequency log σ of the policy.
        min_log_std: Lower clamp of log σ.
        max_log_std: Upper clamp of log σ.
    """

    def __init__(self, n_fft: int = 512, hop: int = 128, hidden: int = 384, rnn_hidden: int = 256,
                 rnn_layers: int = 2, bidirectional: bool = True, conv_kernel: int = 3, mask_max: float = 2.0,
                 init_log_std: float = -1.0, min_log_std: float = -5.0, max_log_std: float = 1.0):
        super().__init__()
        #: Constructor kwargs, stored in every checkpoint so it can be rebuilt.
        self.config = dict(n_fft=n_fft, hop=hop, hidden=hidden, rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
                           bidirectional=bidirectional, conv_kernel=conv_kernel, mask_max=mask_max,
                           init_log_std=init_log_std, min_log_std=min_log_std, max_log_std=max_log_std)
        self.n_fft, self.hop, self.mask_max = n_fft, hop, float(mask_max)
        self.n_freq = n_fft // 2 + 1
        self.min_log_std, self.max_log_std = float(min_log_std), float(max_log_std)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.inp = nn.Sequential(nn.Conv1d(self.n_freq, hidden, conv_kernel, padding=conv_kernel // 2),
                                 nn.GroupNorm(1, hidden), nn.PReLU(hidden))
        self.rnn = nn.GRU(hidden, rnn_hidden, num_layers=rnn_layers, batch_first=True,
                          bidirectional=bidirectional)
        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.out = nn.Linear(out_dim, self.n_freq)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)  # logit 0 → identity mask when mask_max = 2
        self.log_std = nn.Parameter(torch.full((self.n_freq,), float(init_log_std)))

    # ------------------------------------------------------------------ signal
    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """Complex STFT ``[B, F, T']`` of a ``[B, T]`` waveform batch."""
        return torch.stft(wav.float(), self.n_fft, self.hop, window=self.window, center=True, return_complex=True)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Inverse STFT back to ``[B, length]``."""
        return torch.istft(spec, self.n_fft, self.hop, window=self.window, center=True, length=length)

    def frame_mask(self, lengths: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Valid-frame mask ``[B, T']`` for zero-padded inputs (``center=True`` framing).

        Args:
            lengths: Valid sample counts ``[B]``.
            n_frames: Number of STFT frames of the padded batch.

        Returns:
            Float mask with 1 for frames whose centre lies inside the signal.
        """
        frames = (lengths.to(self.window.device) // self.hop + 1).clamp(max=n_frames)
        return (torch.arange(n_frames, device=self.window.device)[None] < frames[:, None]).float()

    # ----------------------------------------------------------------- network
    def mean_logits(self, spec: torch.Tensor) -> torch.Tensor:
        """Policy mean μ in mask-logit space.

        Args:
            spec: Complex noisy STFT ``[B, F, T']``.

        Returns:
            ``[B, F, T']`` mask logits.
        """
        feat = torch.log(spec.abs().pow(2) + 1e-10) / 10.0
        h = self.inp(feat).transpose(1, 2)  # [B, T', H]
        h, _ = self.rnn(h)
        return self.out(h).transpose(1, 2)

    def mask_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Maps logits to a mask in ``[0, mask_max]``."""
        return self.mask_max * torch.sigmoid(logits)

    def synthesize(self, spec: torch.Tensor, logits: torch.Tensor, length: int) -> torch.Tensor:
        """Applies a mask (given as logits) to the noisy STFT and resynthesizes with the noisy phase.

        Args:
            spec: Complex noisy STFT ``[B, F, T']``.
            logits: Mask logits ``[B, F, T']`` (the mean or a sampled action).
            length: Output length in samples.

        Returns:
            Enhanced waveform ``[B, length]``.
        """
        return self.istft(spec * self.mask_from_logits(logits), length)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Deterministic enhancement (mean action).

        Args:
            wav: Noisy ``[B, T]`` waveform at 16 kHz.

        Returns:
            Enhanced ``[B, T]`` waveform.
        """
        spec = self.stft(wav)
        return self.synthesize(spec, self.mean_logits(spec), wav.shape[-1])

    def mask(self, spec: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Mask ``[B, F, T']`` produced by an action (here: the logits themselves)."""
        return self.mask_from_logits(action)

    # --------------------------------------------------------------------- I/O
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        """Saves weights and constructor kwargs atomically (tmp file + rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"state_dict": self.state_dict(), "model_cfg": dict(self.config),
                    "policy_cfg": {"type": "bins"}, **(extra or {})}, tmp)
        tmp.rename(path)


def num_parameters(model: nn.Module, include_log_std: bool = False) -> int:
    """Counts trainable parameters of the enhancement network.

    Args:
        model: The module.
        include_log_std: Whether to count the policy's ``log_std``.

    Returns:
        Parameter count.
    """
    return sum(p.numel() for n, p in model.named_parameters() if include_log_std or n != "log_std")


def erb_interpolation(n_freq: int, n_bands: int, sr: int) -> torch.Tensor:
    """Linear interpolation from ERB-spaced band centres to STFT bins.

    Args:
        n_freq: Number of STFT bins (``n_fft // 2 + 1``).
        n_bands: Number of bands K (centres uniformly spaced on the ERB-rate
            scale from 0 Hz to Nyquist, at least one bin apart).
        sr: Sample rate.

    Returns:
        ``U`` of shape ``[F, K]`` with non-negative rows summing to 1 (hat functions).
    """
    nyq = sr / 2

    def erb_rate(f):
        return 21.4 * np.log10(1 + 0.00437 * f)

    centres = (10 ** (np.linspace(0, erb_rate(nyq), n_bands) / 21.4) - 1) / 0.00437
    pos = centres / nyq * (n_freq - 1)
    for k in range(1, n_bands):
        pos[k] = max(pos[k], pos[k - 1] + 1.0)
    pos = pos * (n_freq - 1) / pos[-1]
    bins = np.arange(n_freq)
    eye = np.eye(n_bands)
    U = np.stack([np.interp(bins, pos, eye[k]) for k in range(n_bands)], axis=1)
    return torch.from_numpy(U / U.sum(1, keepdims=True)).float()


class BandResidualPolicy(GaussianPolicyMixin, nn.Module):
    """Frozen pre-trained enhancer plus a trainable ERB-band logit-offset actor.

    ``logits = μ_base(x) + U · a``, ``a ~ N(δ_θ(x), diag(σ²))`` with
    ``a, δ ∈ R^{K×T'}``. The actor is a copy of the base network whose output
    layer is replaced by a zero-initialized K-channel head.

    Args:
        base: Pre-trained :class:`MaskNet` (frozen here).
        n_bands: Number of ERB bands K.
        init_log_std: Initial per-band log σ.
        min_log_std: Lower clamp of log σ.
        max_log_std: Upper clamp of log σ.
        sr: Sample rate (for the ERB scale).
    """

    def __init__(self, base: MaskNet, n_bands: int = 32, init_log_std: float = -1.0, min_log_std: float = -5.0,
                 max_log_std: float = 1.0, sr: int = 16000):
        super().__init__()
        self.policy_cfg = dict(type="band_residual", n_bands=n_bands, init_log_std=init_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std, sr=sr)
        self.base = base.eval().requires_grad_(False)
        self.actor = copy.deepcopy(base).requires_grad_(True)
        self.actor.log_std.requires_grad_(False)  # unused: the band σ below is the policy's σ
        self.actor.out = nn.Linear(self.actor.out.in_features, n_bands)
        nn.init.zeros_(self.actor.out.weight)
        nn.init.zeros_(self.actor.out.bias)
        self.register_buffer("U", erb_interpolation(base.n_freq, n_bands, sr), persistent=False)
        self.min_log_std, self.max_log_std = float(min_log_std), float(max_log_std)
        self.log_std = nn.Parameter(torch.full((n_bands,), float(init_log_std)))
        self.config = base.config
        self.hop, self.n_freq = base.hop, base.n_freq

    def train(self, mode: bool = True) -> "BandResidualPolicy":
        """Sets train mode on the actor only; the base always stays in eval mode."""
        super().train(mode)
        self.base.eval()
        return self

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """Complex STFT of the base model."""
        return self.base.stft(wav)

    def frame_mask(self, lengths: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Valid-frame mask of the base model."""
        return self.base.frame_mask(lengths, n_frames)

    def mean_logits(self, spec: torch.Tensor) -> torch.Tensor:
        """Action mean δ_θ(x) ``[B, K, T']`` (band logit offsets)."""
        return self.actor.mean_logits(spec)

    def logits(self, spec: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Full-resolution mask logits ``μ_base(x) + U·a``."""
        with torch.no_grad():
            base = self.base.mean_logits(spec)
        return base + torch.einsum("fk,bkt->bft", self.U, action)

    def mask(self, spec: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Mask ``[B, F, T']`` produced by an action."""
        return self.base.mask_from_logits(self.logits(spec, action))

    def synthesize(self, spec: torch.Tensor, action: torch.Tensor, length: int) -> torch.Tensor:
        """Enhanced waveform ``[B, length]`` for a band action."""
        return self.base.istft(spec * self.mask(spec, action), length)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Deterministic enhancement (mean action)."""
        spec = self.stft(wav)
        return self.synthesize(spec, self.mean_logits(spec), wav.shape[-1])

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        """Saves weights, base architecture and policy config atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"state_dict": self.state_dict(), "model_cfg": dict(self.config),
                    "policy_cfg": dict(self.policy_cfg), **(extra or {})}, tmp)
        tmp.rename(path)


def flatten_rnns(module: nn.Module) -> None:
    """Re-compacts cuDNN GRU weights (needed after ``copy.deepcopy``)."""
    for m in module.modules():
        if isinstance(m, nn.GRU):
            m.flatten_parameters()


def load_se_model(path: str | Path, model_cfg: dict | None = None,
                  device: str = "cuda") -> MaskNet | BandResidualPolicy:
    """Loads an enhancement checkpoint (pre-trained :class:`MaskNet` or a GRPO policy).

    The architecture comes from the ``model_cfg``/``policy_cfg`` stored in the
    checkpoint; ``model_cfg`` overrides MaskNet kwargs (only policy-side keys
    such as ``init_log_std`` or clamps should differ).

    Args:
        path: Checkpoint written by ``save``.
        model_cfg: Optional MaskNet kwargs overriding the stored ones.
        device: Target device.

    Returns:
        The model in eval mode.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    kwargs = dict(ckpt.get("model_cfg") or {})
    kwargs.update(model_cfg or {})
    policy_cfg = dict(ckpt.get("policy_cfg") or {"type": "bins"})
    model: MaskNet | BandResidualPolicy = MaskNet(**kwargs)
    if policy_cfg.pop("type") == "band_residual":
        model = BandResidualPolicy(model, **policy_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    flatten_rnns(model)
    return model
