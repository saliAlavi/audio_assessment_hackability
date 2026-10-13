#!/usr/bin/env python3
"""Builds the VoiceBank-DEMAND (16 kHz) manifests for the E3 enhancement experiment.

Writes ``paths.cache_dir("manifests")/vbd_{train,val,test}.csv`` with columns::

    utt_id, noisy_path, clean_path, text, split

Splits:

* ``train``: the 28-speaker training set minus :data:`VAL_SPEAKERS`.
* ``val``:   the utterances of :data:`VAL_SPEAKERS` (held out from the training
  set; used for model selection and every hyperparameter decision).
* ``test``:  the official 824-utterance test set (speakers p232, p257); only
  touched by the final ``se_eval`` runs.

With ``--pack`` it also decodes every file once into packed float32 arrays in
``paths.cache_dir("vbd16k")`` (``{split}_{noisy,clean}.npy`` + ``_offsets.npy``
+ ``{split}_utt_ids.txt``), which :mod:`hackpatch.se.data` memory-maps instead
of opening ~24k small files on the (slow, shared) project file system.
Per-file opens on ``/fs/ess`` take ~0.3 s under load, so run this once in a
job (it is I/O-bound; many threads help)::

    source scripts/env.sh
    python scripts/build_vbd_manifests.py --pack --threads 48
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from hackpatch.utils import paths

#: VoiceBank-DEMAND resampled to 16 kHz (``{clean,noisy}_{trainset_28spk,testset}_wav_16k/`` + ``*_txt/``),
#: under ``HACKPATCH_DATA_ROOT`` (default: the original cluster's read-only project storage).
VBD_ROOT = Path(os.environ.get("HACKPATCH_DATA_ROOT", "/fs/ess/PAS2301/Data/Speech")) / "datasets_voicebank" / \
    "noisy-vctk-16k"
#: Training speakers held out for validation (one female, one male).
VAL_SPEAKERS = ("p226", "p287")
#: Output columns of every VBD manifest.
COLUMNS = ["utt_id", "noisy_path", "clean_path", "text", "split"]
#: Expected utterance count of the official test set.
N_TEST = 824
#: Sample rate of the corpus copy.
SR = 16000


def _read_text(path: Path) -> str:
    """Reads a transcript file and collapses whitespace."""
    return " ".join(path.read_text(encoding="utf-8", errors="replace").split())


def _build(root: Path, subset: str, threads: int) -> pd.DataFrame:
    """Lists one VBD subset (``trainset_28spk`` or ``testset``).

    Args:
        root: VBD root directory.
        subset: Subset name as used in the directory names.
        threads: Reader threads for the transcripts.

    Returns:
        Frame with ``utt_id, noisy_path, clean_path, text`` (sorted by ``utt_id``).

    Raises:
        FileNotFoundError: If a noisy file, clean file or transcript is missing.
    """
    clean_dir, noisy_dir, txt_dir = (root / f"clean_{subset}_wav_16k", root / f"noisy_{subset}_wav_16k",
                                     root / f"{subset}_txt")
    clean = sorted(p.stem for p in clean_dir.glob("*.wav"))
    noisy = {p.stem for p in noisy_dir.glob("*.wav")}
    txt = {p.stem for p in txt_dir.glob("*.txt")}
    missing = [u for u in clean if u not in noisy or u not in txt]
    if missing:
        raise FileNotFoundError(f"{subset}: {len(missing)} utterances lack noisy audio or text, e.g. {missing[:3]}")
    with ThreadPoolExecutor(threads) as ex:
        texts = list(ex.map(_read_text, (txt_dir / f"{u}.txt" for u in clean)))
    return pd.DataFrame({"utt_id": clean, "noisy_path": [str(noisy_dir / f"{u}.wav") for u in clean],
                         "clean_path": [str(clean_dir / f"{u}.wav") for u in clean], "text": texts})


def _read_float32(path: str) -> np.ndarray:
    # The corpus copy is stored as 32-bit float WAV: never read it as int16
    # (libsndfile does not rescale float→int and returns ~0).
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return wav[:, 0]


def pack_split(df: pd.DataFrame, split: str, threads: int) -> None:
    """Decodes one split into packed float32 arrays (atomic writes).

    Args:
        df: Manifest of the split (row order defines the packed order).
        split: Split name.
        threads: Reader threads.
    """
    out = paths.cache_dir("vbd16k")
    for kind in ("noisy", "clean"):
        t0 = time.time()
        with ThreadPoolExecutor(threads) as ex:
            wavs = list(ex.map(_read_float32, df[f"{kind}_path"]))
        offsets = np.zeros(len(wavs) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(w) for w in wavs])
        for name, arr in ((f"{split}_{kind}.npy", np.concatenate(wavs)), (f"{split}_{kind}_offsets.npy", offsets)):
            tmp = out / (name + ".tmp.npy")
            np.save(tmp, arr)
            tmp.rename(out / name)
        print(f"  packed {split}/{kind}: {len(wavs)} files, {offsets[-1] / SR / 3600:.2f} h "
              f"in {time.time() - t0:.0f}s", flush=True)
    (out / f"{split}_utt_ids.txt").write_text("\n".join(df["utt_id"]) + "\n")


def main() -> None:
    """Builds and writes the three manifests (and optionally the packed audio)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=VBD_ROOT)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--pack", action="store_true", help="Also write packed float32 audio to the cache.")
    args = ap.parse_args()

    t0 = time.time()
    train_all = _build(args.root, "trainset_28spk", args.threads)
    spk = train_all["utt_id"].str.split("_").str[0]
    is_val = spk.isin(VAL_SPEAKERS)
    splits = {"train": train_all[~is_val].copy(), "val": train_all[is_val].copy(),
              "test": _build(args.root, "testset", args.threads)}
    if len(splits["test"]) != N_TEST:
        raise ValueError(f"expected {N_TEST} test utterances, found {len(splits['test'])}")
    if set(splits["val"]["utt_id"].str.split("_").str[0]) != set(VAL_SPEAKERS):
        raise ValueError("validation speakers missing from the training set")
    out_dir = paths.cache_dir("manifests")
    for split, df in splits.items():
        df["split"] = split
        out = out_dir / f"vbd_{split}.csv"
        df[COLUMNS].to_csv(out, index=False)
        n_spk = df["utt_id"].str.split("_").str[0].nunique()
        print(f"{split:5s} {len(df):6d} utts {n_spk:2d} speakers -> {out} ({time.time() - t0:.0f}s)", flush=True)
    if args.pack:
        for split, df in splits.items():
            pack_split(df.reset_index(drop=True), split, args.threads)


if __name__ == "__main__":
    main()
