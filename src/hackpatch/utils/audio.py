"""Audio I/O helpers shared by every task.

All models in the project consume 16 kHz mono float32 waveforms in [-1, 1].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

#: Sample rate every predictor, detector and enhancement model operates at.
SR = 16000


def load_wav(path: str | Path, sr: int = SR, max_seconds: float | None = None) -> np.ndarray:
    """Loads an audio file as mono float32 at ``sr``.

    Args:
        path: Audio file path (any format soundfile reads).
        sr: Target sample rate.
        max_seconds: If set, keep only the first ``max_seconds`` seconds.

    Returns:
        1-D float32 array.
    """
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = AF.resample(torch.from_numpy(wav), file_sr, sr).numpy()
    if max_seconds is not None:
        wav = wav[: int(max_seconds * sr)]
    return np.ascontiguousarray(wav, dtype=np.float32)


def pad_batch(wavs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pads a list of 1-D waveforms into a batch.

    Args:
        wavs: List of 1-D tensors.

    Returns:
        (batch [B, T_max], lengths [B]) tensors.
    """
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    out = torch.zeros(len(wavs), int(lengths.max()), dtype=wavs[0].dtype)
    for i, w in enumerate(wavs):
        out[i, : w.shape[-1]] = w
    return out, lengths


def save_wav(path: str | Path, wav: np.ndarray | torch.Tensor, sr: int = SR) -> None:
    """Writes a mono waveform as 16-bit PCM WAV, creating parent dirs.

    Args:
        path: Destination path.
        wav: 1-D waveform in [-1, 1].
        sr: Sample rate.
    """
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1, 1), sr, subtype="PCM_16")
