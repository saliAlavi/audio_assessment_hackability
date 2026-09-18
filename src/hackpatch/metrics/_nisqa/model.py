"""Minimal inference re-implementation of NISQA v2.0 (Mittag et al., Interspeech 2021).

Derived from ``nisqa/NISQA_lib.py`` of https://github.com/gabrielmittag/NISQA
(commit fe84f0f, MIT licence, see ``LICENSE`` in this directory). Only the
configuration used by the released ``nisqa.tar`` checkpoint is kept:
AdaptCNN framewise model -> 2-layer self-attention -> five attention-pooling
heads (MOS, NOI, DIS, COL, LOUD). Module and parameter names are identical to
the original so the official state dict loads with ``strict=True``.

Changes w.r.t. the original, none of which alter the outputs:
    * ``pack_padded_sequence`` in the framewise stage is replaced by boolean
      masking (same per-segment computation);
    * masks are created on the input device;
    * the librosa mel-spectrogram is computed with ``torch.stft`` using the
      same window, padding, librosa mel filterbank and dB conversion, so it
      can run on GPU.

The model weights (``nisqa.tar``) are licensed separately under CC BY-NC-SA 4.0.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptCNN(nn.Module):
    """Six-layer CNN with adaptive max-pooling applied to each 48x15 mel segment."""

    def __init__(self, input_channels: int = 1, c_out_1: int = 16, c_out_2: int = 32, c_out_3: int = 64,
                 kernel_size: tuple[int, int] = (3, 3), dropout: float = 0.2,
                 pool_1: tuple[int, int] = (24, 7), pool_2: tuple[int, int] = (12, 5),
                 pool_3: tuple[int, int] = (6, 3), fc_out_h: int | None = None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.pool_1, self.pool_2, self.pool_3 = list(pool_1), list(pool_2), list(pool_3)
        self.fc_out_h = fc_out_h
        self.dropout = nn.Dropout2d(p=dropout)
        kernel_size_last = (kernel_size[0], pool_3[1])
        cnn_pad = (1, 0) if kernel_size[1] == 1 else (1, 1)
        self.conv1 = nn.Conv2d(input_channels, c_out_1, kernel_size, padding=cnn_pad)
        self.bn1 = nn.BatchNorm2d(c_out_1)
        self.conv2 = nn.Conv2d(c_out_1, c_out_2, kernel_size, padding=cnn_pad)
        self.bn2 = nn.BatchNorm2d(c_out_2)
        self.conv3 = nn.Conv2d(c_out_2, c_out_3, kernel_size, padding=cnn_pad)
        self.bn3 = nn.BatchNorm2d(c_out_3)
        self.conv4 = nn.Conv2d(c_out_3, c_out_3, kernel_size, padding=cnn_pad)
        self.bn4 = nn.BatchNorm2d(c_out_3)
        self.conv5 = nn.Conv2d(c_out_3, c_out_3, kernel_size, padding=cnn_pad)
        self.bn5 = nn.BatchNorm2d(c_out_3)
        self.conv6 = nn.Conv2d(c_out_3, c_out_3, kernel_size_last, padding=(1, 0))
        self.bn6 = nn.BatchNorm2d(c_out_3)
        if fc_out_h:
            self.fc = nn.Linear(c_out_3 * pool_3[0], fc_out_h)
            self.fan_out = fc_out_h
        else:
            self.fan_out = c_out_3 * pool_3[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps segments [N, 1, 48, 15] to features [N, fan_out]."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_max_pool2d(x, output_size=self.pool_1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool2d(x, output_size=self.pool_2)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_max_pool2d(x, output_size=self.pool_3)
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, self.conv6.out_channels * self.pool_3[0])
        if self.fc_out_h:
            x = self.fc(x)
        return x


class Framewise(nn.Module):
    """Applies the CNN independently to every valid segment of a padded batch."""

    def __init__(self, **cnn_kwargs):
        super().__init__()
        self.model = AdaptCNN(**cnn_kwargs)

    def forward(self, x: torch.Tensor, n_wins: torch.Tensor) -> torch.Tensor:
        """Maps [B, L, 1, 48, 15] segments to [B, L, fan_out] (zeros at padding)."""
        bs, length = x.shape[:2]
        valid = torch.arange(length, device=x.device)[None, :] < n_wins[:, None]
        feats = self.model(x[valid])
        out = feats.new_zeros(bs, length, feats.shape[-1])
        out[valid] = feats
        return out


class SelfAttentionLayer(nn.Module):
    """Post-norm transformer encoder layer with key padding mask."""

    def __init__(self, d_model: int, nhead: int, sa_h: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, sa_h)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(sa_h, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, n_wins: torch.Tensor) -> torch.Tensor:
        """Encodes [L, B, D] (sequence first, as in the original)."""
        mask = ~(torch.arange(src.shape[0], device=src.device)[None, :] < n_wins[:, None])
        src2 = self.self_attn(src, src, src, key_padding_mask=mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class SelfAttention(nn.Module):
    """Linear projection + LayerNorm + stacked self-attention layers."""

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 1, num_layers: int = 2,
                 sa_h: int = 64, dropout: float = 0.1):
        super().__init__()
        layer = SelfAttentionLayer(d_model, nhead, sa_h, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src: torch.Tensor, n_wins: torch.Tensor) -> torch.Tensor:
        """Maps [B, L, input_size] to [B, L, d_model]."""
        out = self.norm1(self.linear(src).transpose(1, 0))
        for mod in self.layers:
            out = mod(out, n_wins)
        return out.transpose(1, 0)


class TimeDependency(nn.Module):
    """Wrapper kept for state-dict key compatibility (``time_dependency.model.*``)."""

    def __init__(self, **sa_kwargs):
        super().__init__()
        self.model = SelfAttention(**sa_kwargs)

    def forward(self, x: torch.Tensor, n_wins: torch.Tensor) -> torch.Tensor:
        return self.model(x, n_wins)


class PoolAttFF(nn.Module):
    """Attention pooling with a feed-forward attention network."""

    def __init__(self, d_input: int, output_size: int = 1, h: int = 128, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)
        self.linear3 = nn.Linear(d_input, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, n_wins: torch.Tensor) -> torch.Tensor:
        """Pools [B, L, D] to [B, output_size] over valid steps."""
        att = self.linear2(self.dropout(F.relu(self.linear1(x)))).transpose(2, 1)
        mask = torch.arange(att.shape[2], device=x.device)[None, :] < n_wins[:, None]
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        att = F.softmax(att, dim=2)
        return self.linear3(torch.bmm(att, x).squeeze(1))


class Pooling(nn.Module):
    """Wrapper kept for state-dict key compatibility (``pool_layers.i.model.*``)."""

    def __init__(self, d_input: int, h: int, dropout: float):
        super().__init__()
        self.model = PoolAttFF(d_input, 1, h=h, dropout=dropout)

    def forward(self, x: torch.Tensor, n_wins: torch.Tensor) -> torch.Tensor:
        return self.model(x, n_wins)


class NISQA_DIM(nn.Module):  # noqa: N801 - original class name
    """NISQA with quality dimensions: outputs [MOS, NOI, DIS, COL, LOUD]."""

    def __init__(self, args: dict):
        super().__init__()
        assert args["cnn_model"] == "adapt" and args["td"] == "self_att" and args["td_2"] == "skip" \
            and args["pool"] == "att" and args["pool_att_h"], "unsupported NISQA configuration"
        self.cnn = Framewise(
            input_channels=1, c_out_1=args["cnn_c_out_1"], c_out_2=args["cnn_c_out_2"],
            c_out_3=args["cnn_c_out_3"], kernel_size=args["cnn_kernel_size"], dropout=args["cnn_dropout"],
            pool_1=args["cnn_pool_1"], pool_2=args["cnn_pool_2"], pool_3=args["cnn_pool_3"],
            fc_out_h=args["cnn_fc_out_h"])
        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out, d_model=args["td_sa_d_model"], nhead=args["td_sa_nhead"],
            num_layers=args["td_sa_num_layers"], sa_h=args["td_sa_h"], dropout=args["td_sa_dropout"])
        pool = Pooling(args["td_sa_d_model"], h=args["pool_att_h"], dropout=args["pool_att_dropout"])
        self.pool_layers = nn.ModuleList([copy.deepcopy(pool) for _ in range(5)])

    def forward(self, x: torch.Tensor, n_wins: torch.Tensor) -> torch.Tensor:
        """Predicts [B, 5] scores from padded segments [B, L, 1, 48, 15] and counts [B]."""
        x = self.cnn(x, n_wins)
        x = self.time_dependency(x, n_wins)
        return torch.cat([mod(x, n_wins) for mod in self.pool_layers], dim=1)


class NISQAPredictor(nn.Module):
    """Waveform-in NISQA v2.0 predictor with batched GPU feature extraction.

    Args:
        checkpoint: Path to the official ``nisqa.tar``.
        sr: Sample rate of the input waveforms. The checkpoint has
            ``ms_sr=None``, i.e. the original computes features at the file's
            native rate, which we reproduce.
    """

    def __init__(self, checkpoint: str | Path, sr: int = 16000):
        super().__init__()
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.args = args = ck["args"]
        if args.get("model") != "NISQA_DIM":
            raise ValueError(f"expected a NISQA_DIM checkpoint, got {args.get('model')}")
        if args.get("ms_sr") not in (None, sr):
            raise ValueError(f"checkpoint expects sr={args['ms_sr']}, got {sr}")
        self.model = NISQA_DIM(args)
        self.model.load_state_dict(ck["model_state_dict"], strict=True)
        self.model.eval()
        self.sr = sr
        self.n_fft = int(args["ms_n_fft"])
        self.hop = int(sr * args["ms_hop_length"])
        self.win = int(sr * args["ms_win_length"])
        self.seg_length = int(args["ms_seg_length"])
        self.seg_hop = int(args["ms_seg_hop_length"])
        from hackpatch.metrics._utils import librosa_mel_filterbank

        mel = librosa_mel_filterbank(sr=sr, n_fft=self.n_fft, n_mels=args["ms_n_mels"], fmin=0.0,
                                     fmax=args["ms_fmax"])
        self.register_buffer("mel_fb", torch.from_numpy(mel.astype(np.float32)), persistent=False)
        self.register_buffer("window", torch.hann_window(self.win, periodic=True), persistent=False)

    def min_samples(self) -> int:
        """Shortest input (in samples) that yields at least one segment."""
        return max(self.n_fft // 2 + 1, (self.seg_length - 1) * self.hop)

    def melspec(self, wav: torch.Tensor) -> torch.Tensor:
        """Log-mel spectrogram [n_mels, T] matching NISQA's ``get_librosa_melspec``.

        Args:
            wav: 1-D float32 waveform on the model device.
        """
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.window, center=True, pad_mode="reflect", return_complex=True).abs()
        mel = torch.matmul(self.mel_fb, spec)  # power=1.0 magnitude mel spectrogram
        db = 20.0 * torch.log10(mel.clamp_min(1e-4))  # amplitude_to_db(ref=1.0, amin=1e-4)
        return torch.maximum(db, db.max() - 80.0)  # top_db=80

    def segments(self, spec: torch.Tensor) -> torch.Tensor:
        """Splits [n_mels, T] into hopped segments [n_wins, 1, n_mels, seg_length]."""
        return spec.unfold(1, self.seg_length, self.seg_hop).permute(1, 0, 2).unsqueeze(1)

    @torch.no_grad()
    def forward(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        """Predicts [MOS, NOI, DIS, COL, LOUD] for a list of waveforms.

        Args:
            wavs: 1-D float32 tensors on the model device, each at least
                :meth:`min_samples` long.

        Returns:
            [B, 5] predictions.
        """
        segs = [self.segments(self.melspec(w)) for w in wavs]
        n_wins = torch.tensor([s.shape[0] for s in segs], device=self.mel_fb.device)
        x = segs[0].new_zeros(len(segs), int(n_wins.max()), *segs[0].shape[1:])
        for i, s in enumerate(segs):
            x[i, : s.shape[0]] = s
        return self.model(x, n_wins)
