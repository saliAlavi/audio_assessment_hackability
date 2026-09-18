"""Torch datasets over manifests and preference pairs."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hackpatch.utils.audio import SR, load_wav, pad_batch


class ManifestDataset(Dataset):
    """Labeled audio from a manifest, optionally random-cropped.

    Built from the ``data`` config section.

    Args:
        df: Manifest frame (see ``hackpatch.data.manifests``).
        max_seconds: Random crop length during training; None keeps full audio.
        train: If True, crops are random; otherwise the first window is used.
        cache_in_ram: Decode each file once and keep it in memory.
    """

    def __init__(self, df: pd.DataFrame, max_seconds: float | None = None,
                 train: bool = False, cache_in_ram: bool = True):
        self.df = df.reset_index(drop=True)
        self.max_len = int(max_seconds * SR) if max_seconds else None
        self.train = train
        self.cache: dict[int, np.ndarray] | None = {} if cache_in_ram else None

    def __len__(self) -> int:
        return len(self.df)

    def _wav(self, i: int) -> np.ndarray:
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        wav = load_wav(self.df.at[i, "path"])
        if self.cache is not None:
            self.cache[i] = wav
        return wav

    def __getitem__(self, i: int) -> dict:
        wav = self._wav(i)
        if self.max_len and len(wav) > self.max_len:
            start = random.randint(0, len(wav) - self.max_len) if self.train else 0
            wav = wav[start:start + self.max_len]
        mos = self.df.at[i, "mos"]
        return {"wav": torch.from_numpy(wav.copy()), "mos": float(mos) if pd.notna(mos) else float("nan"),
                "idx": i}


def collate(items: list[dict]) -> dict:
    """Pads waveforms and stacks labels."""
    wav, lengths = pad_batch([it["wav"] for it in items])
    return {"wav": wav, "lengths": lengths,
            "mos": torch.tensor([it["mos"] for it in items], dtype=torch.float32),
            "idx": torch.tensor([it["idx"] for it in items])}


class PairDataset(Dataset):
    """Preference pairs ``preferred ≻ rejected`` stored as wav paths.

    Both sides are cropped to the same window so the ranking loss compares
    time-aligned audio.

    Args:
        pairs: Frame with ``preferred_path`` and ``rejected_path`` columns.
        max_seconds: Aligned crop length; None keeps full audio.
    """

    def __init__(self, pairs: pd.DataFrame, max_seconds: float | None = None):
        self.pairs = pairs.reset_index(drop=True)
        self.max_len = int(max_seconds * SR) if max_seconds else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        a = load_wav(self.pairs.at[i, "preferred_path"])
        b = load_wav(self.pairs.at[i, "rejected_path"])
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if self.max_len and n > self.max_len:
            s = random.randint(0, n - self.max_len)
            a, b = a[s:s + self.max_len], b[s:s + self.max_len]
        return {"pos": torch.from_numpy(a.copy()), "neg": torch.from_numpy(b.copy())}


def collate_pairs(items: list[dict]) -> dict:
    """Pads both sides of each pair to a shared length."""
    wav, lengths = pad_batch([it["pos"] for it in items] + [it["neg"] for it in items])
    n = len(items)
    return {"pos": wav[:n], "neg": wav[n:], "pos_len": lengths[:n], "neg_len": lengths[n:]}
