"""SSL-MOS: wav2vec 2.0 encoder + mean pooling + linear head (Cooper et al., 2022)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class SSLMOS(nn.Module):
    """Utterance-level MOS predictor on a self-supervised speech encoder.

    Built from the ``model`` config section.

    Args:
        pretrained: HF model id of the SSL encoder.
        freeze_feature_encoder: Freeze the CNN waveform front-end (standard
            practice for wav2vec 2.0 fine-tuning).
        freeze_encoder: Freeze the whole SSL encoder and train only the head
            (the "patch head only" fallback in the plan).
    """

    def __init__(self, pretrained: str = "facebook/wav2vec2-base",
                 freeze_feature_encoder: bool = True, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(pretrained)
        # SpecAugment-style masking inside HF wav2vec2 would randomize scores
        # during attacks; MOS fine-tuning recipes do not use it either.
        self.encoder.config.mask_time_prob = 0.0
        self.encoder.config.mask_feature_prob = 0.0
        self.encoder.config.layerdrop = 0.0
        if freeze_feature_encoder:
            self.encoder.feature_extractor._freeze_parameters()
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Predicts MOS for a zero-padded batch.

        Args:
            wav: [B, T] 16 kHz waveforms.
            lengths: [B] valid sample counts; None means no padding.

        Returns:
            [B] predicted MOS on the 1-5 scale (unbounded linear output).
        """
        hidden = self.encoder(wav).last_hidden_state  # [B, F, D]
        if lengths is None:
            pooled = hidden.mean(dim=1)
        else:
            frames = self.encoder._get_feat_extract_output_lengths(lengths.to(wav.device))
            frames = frames.clamp(min=1, max=hidden.shape[1])
            mask = (torch.arange(hidden.shape[1], device=wav.device)[None] < frames[:, None])
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        return self.head(pooled).squeeze(-1) + 3.0  # center the init at mid-scale

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        """Saves weights atomically (tmp file + rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"state_dict": self.state_dict(), **(extra or {})}, tmp)
        tmp.rename(path)

    def load(self, path: str | Path) -> dict:
        """Loads weights saved by :meth:`save`; returns the checkpoint dict."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["state_dict"])
        return ckpt


def resolve_ckpt(path: str | Path) -> Path:
    """Resolves a checkpoint pointer at run time.

    Jobs are submitted before their inputs exist, so a checkpoint may be given
    as a ``.txt`` file containing the path (``selected_checkpoint.txt``,
    ``final_checkpoint.txt``) or a loop ``done.json`` with a ``ckpt`` key.
    """
    import json

    path = Path(path)
    if path.suffix == ".txt":
        return Path(path.read_text().strip())
    if path.suffix == ".json":
        return Path(json.loads(path.read_text())["ckpt"])
    return path


def load_predictor(path: str | Path, pretrained: str = "facebook/wav2vec2-base",
                   device: str = "cuda") -> SSLMOS:
    """Loads a trained SSL-MOS checkpoint in eval mode.

    Args:
        path: Checkpoint written by :meth:`SSLMOS.save`.
        pretrained: Encoder id used to build the architecture.
        device: Target device.

    Returns:
        The model in eval mode with gradients enabled for its parameters left
        untouched (callers decide whether to freeze).
    """
    model = SSLMOS(pretrained=pretrained)
    model.load(resolve_ckpt(path))
    return model.to(device).eval()
