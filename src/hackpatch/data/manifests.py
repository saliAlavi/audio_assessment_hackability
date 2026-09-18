"""Standard manifest format shared by every corpus.

A manifest is a CSV with columns::

    utt_id, path, mos, system, corpus, split

``mos`` may be NaN for unlabeled audio (e.g. attack inputs), ``system`` is the
system/condition id used for system-level correlation. Manifests live in
``paths.cache_dir("manifests")/{corpus}_{split}.csv``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hackpatch.utils import paths

COLUMNS = ["utt_id", "path", "mos", "system", "corpus", "split"]


def manifest_path(corpus: str, split: str) -> Path:
    """Returns the canonical manifest location for ``corpus``/``split``."""
    return paths.cache_dir("manifests") / f"{corpus}_{split}.csv"


def write_manifest_csv(df: pd.DataFrame, corpus: str, split: str) -> Path:
    """Validates columns and writes a manifest.

    Args:
        df: Frame with at least :data:`COLUMNS`.
        corpus: Corpus id (e.g. ``bvcc``).
        split: Split id (e.g. ``train``).

    Returns:
        The written path.

    Raises:
        ValueError: If columns are missing or audio files do not exist.
    """
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns {missing}")
    absent = [p for p in df["path"].head(50) if not Path(p).exists()]
    if absent:
        raise ValueError(f"audio files not found, e.g. {absent[:3]}")
    out = manifest_path(corpus, split)
    df[COLUMNS].to_csv(out, index=False)
    return out


def read_manifest(corpus: str, split: str) -> pd.DataFrame:
    """Reads a manifest written by :func:`write_manifest_csv`."""
    return pd.read_csv(manifest_path(corpus, split))


def build_bvcc(root: str) -> None:
    """Builds BVCC (VMC'22 main track) train/val/test manifests.

    Args:
        root: The BVCC ``main/DATA`` directory with ``wav/`` and ``sets/``.
    """
    root = Path(root)
    for split, fname in [("train", "train_mos_list.txt"), ("val", "val_mos_list.txt"),
                         ("test", "test_mos_list.txt")]:
        df = pd.read_csv(root / "sets" / fname, header=None, names=["wav", "mos"])
        df["utt_id"] = df["wav"].str.replace(".wav", "", regex=False)
        df["path"] = df["wav"].map(lambda w: str(root / "wav" / w))
        df["system"] = df["wav"].str.split("-").str[0]
        df["corpus"] = "bvcc"
        df["split"] = split
        print(split, len(df), "->", write_manifest_csv(df, "bvcc", split))
