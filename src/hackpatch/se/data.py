"""VoiceBank-DEMAND manifests and paired (noisy, clean) datasets for the E3 enhancement model.

Manifests are built by ``scripts/build_vbd_manifests.py`` and have columns
``utt_id, noisy_path, clean_path, text, split``. When the script was run with
``--pack``, audio is read from memory-mapped packed float32 arrays in
``paths.cache_dir("vbd16k")``; otherwise it falls back to the wav files.
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from hackpatch.utils import paths
from hackpatch.utils.audio import SR


def vbd_manifest(split: str) -> pd.DataFrame:
    """Reads a VBD manifest.

    Args:
        split: ``train``, ``val`` or ``test``.

    Returns:
        The manifest frame.

    Raises:
        FileNotFoundError: If ``scripts/build_vbd_manifests.py`` has not been run.
    """
    path = paths.cache_dir("manifests") / f"vbd_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/build_vbd_manifests.py")
    return pd.read_csv(path, keep_default_na=False)


def fixed_subset(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Draws a deterministic utterance subset (independent of the run seed).

    Args:
        df: Manifest.
        n: Subset size; None or ``>= len(df)`` keeps everything.
        seed: Sampling seed.

    Returns:
        The subset sorted by ``utt_id``.
    """
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.sort_values("utt_id").reset_index(drop=True)


def _read_float32(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return wav[:, 0].copy()


class AudioStore:
    """Read access to VBD audio by ``(split, kind, utt_id)``.

    Uses the packed memory-mapped arrays when present (fast, fork-safe, shared
    page cache across DataLoader workers) and the original wav files otherwise.

    Args:
        pack_dir: Directory with packed arrays; None uses ``paths.cache_dir("vbd16k")``.
    """

    def __init__(self, pack_dir: str | Path | None = None):
        self.dir = Path(pack_dir) if pack_dir else paths.cache_dir("vbd16k", create=False)
        self._packs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, int]] | None] = {}

    def _pack(self, split: str, kind: str):
        key = (split, kind)
        if key not in self._packs:
            files = [self.dir / f"{split}_{kind}.npy", self.dir / f"{split}_{kind}_offsets.npy",
                     self.dir / f"{split}_utt_ids.txt"]
            if all(f.exists() for f in files):
                ids = files[2].read_text().split()
                self._packs[key] = (np.load(files[0], mmap_mode="r"), np.load(files[1]),
                                    {u: i for i, u in enumerate(ids)})
            else:
                self._packs[key] = None
        return self._packs[key]

    def packed(self, split: str) -> bool:
        """Whether packed arrays exist for ``split``."""
        return self._pack(split, "noisy") is not None and self._pack(split, "clean") is not None

    def get(self, row: pd.Series, kind: str) -> np.ndarray:
        """Returns the float32 waveform of one manifest row.

        Args:
            row: Manifest row (needs ``utt_id``, ``split`` and ``{kind}_path``).
            kind: ``noisy`` or ``clean``.

        Returns:
            1-D float32 array (a read-only view for packed audio).
        """
        pack = self._pack(str(row["split"]), kind)
        if pack is not None and row["utt_id"] in pack[2]:
            data, offsets, index = pack
            i = index[row["utt_id"]]
            return data[offsets[i]:offsets[i + 1]]
        return _read_float32(row[f"{kind}_path"])


class PairedCropDataset(Dataset):
    """Aligned random crops of (noisy, clean) pairs.

    Built from the ``data`` config section.

    Args:
        df: VBD manifest.
        crop_seconds: Crop length. Shorter utterances are zero-padded and their
            true length is returned in ``lengths``.
        train: Random crop offsets if True, else the first window.
        with_clean: Also return the clean crop (False for GRPO, which only
            needs the noisy input).
        store: Audio source; None creates a default :class:`AudioStore`.
    """

    def __init__(self, df: pd.DataFrame, crop_seconds: float, train: bool = True, with_clean: bool = True,
                 store: AudioStore | None = None):
        self.df = df.reset_index(drop=True)
        self.crop = int(crop_seconds * SR)
        self.train = train
        self.with_clean = with_clean
        self.store = store or AudioStore()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        noisy = self.store.get(row, "noisy")
        clean = self.store.get(row, "clean") if self.with_clean else None
        n = len(noisy) if clean is None else min(len(noisy), len(clean))
        start = random.randint(0, n - self.crop) if (self.train and n > self.crop) else 0
        length = min(self.crop, n)
        out = {"length": length, "idx": i}
        for key, w in (("noisy", noisy), ("clean", clean)):
            if w is None:
                continue
            seg = np.zeros(self.crop, dtype=np.float32)
            seg[:length] = w[start:start + length]
            out[key] = torch.from_numpy(seg)
        return out


def collate_crops(items: list[dict]) -> dict:
    """Stacks equal-length crops.

    Args:
        items: Outputs of :meth:`PairedCropDataset.__getitem__`.

    Returns:
        Dict with ``noisy`` [B, T], optional ``clean`` [B, T], ``lengths`` [B], ``idx`` [B].
    """
    out = {"noisy": torch.stack([it["noisy"] for it in items]),
           "lengths": torch.tensor([it["length"] for it in items], dtype=torch.long),
           "idx": torch.tensor([it["idx"] for it in items], dtype=torch.long)}
    if "clean" in items[0]:
        out["clean"] = torch.stack([it["clean"] for it in items])
    return out


def load_utterances(df: pd.DataFrame, kind: str, threads: int = 16,
                    store: AudioStore | None = None) -> list[np.ndarray]:
    """Loads full-length utterances as float32.

    Args:
        df: Manifest.
        kind: ``noisy`` or ``clean``.
        threads: Reader threads (only matters without packed audio).
        store: Audio source; None creates a default :class:`AudioStore`.

    Returns:
        List of 1-D float32 arrays in manifest order.
    """
    store = store or AudioStore()

    def one(i: int) -> np.ndarray:
        return np.array(store.get(df.iloc[i], kind), dtype=np.float32)

    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(one, range(len(df))))
