#!/usr/bin/env python3
"""Builds the E2 evaluation manifests (MOS-predictor vs. human correlation).

Deterministically (re)creates, via :func:`hackpatch.data.manifests.write_manifest_csv`:

* ``bvcc_ood_{train,val,test}``   VMC'22 OOD track (Chinese Blizzard 2019).
* ``nisqa_test_{for,p501,livetalk}``  NISQA corpus test sets (overall ``mos``).
* ``somos_test``                   SOMOS split1 *clean* test list, audios_16000.
* ``tmhintqi_test``                TMHINT-QI official test directory.
* ``vmc23_track{1a,1b,2,3}``       VoiceMOS 2023 test sets with released answers.
* ``urgent24_sqa_test`` / ``urgent25_sqa_test``  URGENT SQA ``blind_test_mos``.

Usage::

    source scripts/env.sh
    python scripts/build_eval_manifests.py                 # all corpora + checks + doc
    python scripts/build_eval_manifests.py --only nisqa vmc23
    python scripts/build_eval_manifests.py --no-doc

URGENT audio is stored in HF parquet shards; the first run extracts the raw
(already-encoded) audio bytes to ``data/audio/`` next to the shards (no
re-encoding, idempotent).  Run ``download.sh`` in the dataset dir first.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from hackpatch.data.manifests import COLUMNS, write_manifest_csv
from hackpatch.utils import paths

SPEECH = Path("/fs/ess/PAS2301/Data/Speech")
MOS = SPEECH / "MOS_datasets"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC = PROJECT_ROOT / "docs" / "eval_corpora.md"


@dataclass
class Built:
    """One manifest plus the provenance needed for the summary table."""

    corpus: str
    split: str
    df: pd.DataFrame
    expected: int | None
    source: str


def _frame(utt_id, path, mos, system, corpus: str, split: str) -> pd.DataFrame:
    df = pd.DataFrame({"utt_id": list(utt_id), "path": list(path),
                       "mos": np.asarray(mos, dtype=float), "system": list(system)})
    df["corpus"], df["split"] = corpus, split
    return df.sort_values("utt_id", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Local corpora
# --------------------------------------------------------------------------- #
def build_bvcc_ood() -> list[Built]:
    """VMC'22 OOD track (BC2019, Mandarin). Official labeled train/val/test lists.

    The 540-utt ``unlabeled_mos_list_LABELS.txt`` (unlabeled-train, labels
    revealed after the challenge) is intentionally not exported.
    """
    root = MOS / "bc19" / "ood" / "DATA"
    out = []
    for split, fname, n in [("train", "train_mos_list.txt", 136),
                            ("val", "val_mos_list.txt", 136),
                            ("test", "test_mos_list.txt", 540)]:
        lst = pd.read_csv(root / "sets" / fname, header=None, names=["wav", "mos"])
        utt = lst["wav"].str.replace(".wav", "", regex=False)
        df = _frame(utt, [str(root / "wav" / w) for w in lst["wav"]], lst["mos"],
                    utt.str.split("-").str[0], "bvcc_ood", split)
        out.append(Built("bvcc_ood", split, df, n,
                         f"local `{root}/sets/{fname}` (VMC'22 OOD release)"))
    return out


def build_nisqa() -> list[Built]:
    """NISQA_TEST_{FOR,P501,LIVETALK}: per-file overall MOS, system = condition ``con``."""
    root = SPEECH / "NISQA_Corpus"
    out = []
    for db, n in [("FOR", 240), ("P501", 240), ("LIVETALK", 232)]:
        name = f"NISQA_TEST_{db}"
        f = pd.read_csv(root / name / f"{name}_file.csv")
        utt = name + "/" + f["filename_deg"].str.replace(".wav", "", regex=False)
        system = name + "_c" + f["con"].astype(int).astype(str)
        df = _frame(utt, [str(root / p) for p in f["filepath_deg"]], f["mos"], system,
                    "nisqa", f"test_{db.lower()}")
        out.append(Built("nisqa", f"test_{db.lower()}", df, n,
                         f"local `{root}/{name}/{name}_file.csv` (48 kHz deg)"))
    return out


def build_somos() -> list[Built]:
    """SOMOS split1 test (3000 utts, 201 systems), *clean* MOS labels, 16 kHz audio."""
    root = MOS / "somos"
    lst = pd.read_csv(root / "training_files" / "split1" / "clean" / "test_mos_list.txt")
    utt = lst["utteranceId"].str.replace(".wav", "", regex=False)
    df = _frame(utt, [str(root / "audios_16000" / w) for w in lst["utteranceId"]],
                lst["mean"], utt.str.rsplit("_", n=1).str[1], "somos", "test")
    return [Built("somos", "test", df, 3000,
                  f"local `{root}/training_files/split1/clean/test_mos_list.txt`")]


def build_tmhintqi() -> list[Built]:
    """TMHINT-QI test dir; MOS = mean of raw ``quality_score`` per file.

    Follows the SHEET/MOS-Bench recipe: system = ``{method}-SNR{snr}-{noise}``;
    rows of ``raw_data.csv`` whose ``file_name`` has no wav in ``test/`` are ignored
    (e.g. the un-prefixed clean ``TMHINT_*`` rows).
    """
    root = SPEECH / "TMHINTQI"
    raw = pd.read_csv(root / "raw_data.csv", keep_default_na=False)
    raw = raw[raw["quality_score"] != ""]
    raw["quality_score"] = raw["quality_score"].astype(float).astype(int)
    wavs = {f[:-4] for f in os.listdir(root / "test") if f.endswith(".wav")}
    raw = raw[raw["file_name"].isin(wavs)]
    raw["system"] = raw["method"] + "-SNR" + raw["snr"] + "-" + raw["noise"]
    g = raw.groupby("file_name").agg(mos=("quality_score", "mean"),
                                     system=("system", "first"),
                                     n_sys=("system", "nunique")).reset_index()
    assert (g["n_sys"] == 1).all(), "file mapped to more than one system"
    df = _frame(g["file_name"], [str(root / "test" / f"{u}.wav") for u in g["file_name"]],
                g["mos"], g["system"], "tmhintqi", "test")
    return [Built("tmhintqi", "test", df, 1978, f"local `{root}/raw_data.csv` + `test/`")]


def build_vmc23() -> list[Built]:
    """VoiceMOS Challenge 2023 test sets with the organizers' released answers.

    Official track naming (Cooper et al., ASRU 2023, arXiv 2310.02640):
    1a = Blizzard'23 French hub task (speaker NEB), 1b = spoke task (speaker AD),
    2 = SVCC'23 singing voice conversion, 3 = TMHINT-QI(S) noisy/enhanced Mandarin.
    """
    root = MOS / "vmc23"
    out = []
    specs = [("track1a", 1, 882), ("track1b", 1, 578), ("track2", 2, 4040), ("track3", 3, 1960)]
    for split, t, n in specs:
        ans = pd.read_csv(root / "answers" / f"track{t}_answer.txt", header=None,
                          names=["utt", "mos"])
        wavdir = root / f"track{t}"
        wavs = {f[:-4] for f in os.listdir(wavdir) if f.endswith(".wav")}
        # track3 answers also list 255 un-prefixed "TMHINT_*" ids with no audio.
        ans = ans[ans["utt"].isin(wavs)]
        body = ans["utt"].str.replace(f"VoiceMOS2023Track{t}-", "", regex=False)
        if t == 1:
            system = body.str.split("_").str[0]              # e.g. "A-NEB"
            keep = system.str.endswith("-NEB" if split == "track1a" else "-AD")
            ans, system = ans[keep], system[keep]
        elif t == 2:
            system = body.str.split("-").str[0]              # team id, e.g. "B01"
        else:
            system = body.str.rsplit("_", n=3).str[0]        # e.g. "CMGAN_snr-2_babble_TMHINT"
        df = _frame(ans["utt"], [str(wavdir / f"{u}.wav") for u in ans["utt"]],
                    ans["mos"], system, "vmc23", split)
        out.append(Built("vmc23", split, df, n,
                         f"local `{root}/answers/track{t}_answer.txt` (= SHEET egs/vmc23/answers)"))
    return out


# --------------------------------------------------------------------------- #
# URGENT (HF parquet -> audio files)
# --------------------------------------------------------------------------- #
def _extract_urgent(year: int) -> pd.DataFrame:
    """Extracts audio bytes + labels from the HF ``blind_test_mos`` parquet shards.

    Returns:
        Metadata frame (``sample_id, system_id, fileid, mos, n_ratings, path``).
    """
    import pyarrow.parquet as pq

    base = paths.raw_data_dir(f"urgent{year}_sqa", create=False) / "data"
    meta_csv = base / "metadata.csv"
    if meta_csv.exists():
        return pd.read_csv(meta_csv)
    shards = sorted(glob.glob(str(base / "hf" / "data" / "blind_test_mos" / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {base}/hf; run download.sh")
    rows = []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            tab = pf.read_row_group(rg, columns=["audio", "sample_id", "system_id", "mos",
                                                  "raw_ratings"])
            for r in tab.to_pylist():
                fileid = re.search(r"fileid_(\d+)", r["sample_id"]).group(1)
                ext = Path(r["audio"]["path"] or "x.flac").suffix or ".flac"
                out = base / "audio" / r["system_id"] / f"fileid_{fileid}{ext}"
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(out.suffix + ".tmp")
                    tmp.write_bytes(r["audio"]["bytes"])
                    tmp.rename(out)
                rr = r["raw_ratings"]
                rows.append({"sample_id": r["sample_id"], "system_id": r["system_id"],
                             "fileid": fileid, "mos": r["mos"],
                             "n_ratings": len(rr) if rr is not None else np.nan,
                             "path": str(out)})
        print(f"  extracted {shard}", flush=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(meta_csv, index=False)
    return meta


def build_urgent() -> list[Built]:
    """URGENT 2024 (6900 = 23 sys x 300) and 2025 (23400 = 26 sys x 900) SQA MOS sets."""
    out = []
    for year, n in [(24, 6900), (25, 23400)]:
        meta = _extract_urgent(year)
        utt = meta["system_id"] + "__fileid_" + meta["fileid"].astype(str).str.zfill(6)
        corpus = f"urgent{year}_sqa"
        df = _frame(utt, meta["path"], meta["mos"], meta["system_id"], corpus, "test")
        out.append(Built(corpus, "test", df, n,
                         f"HF `urgent-challenge/urgent20{year}-sqa` split `blind_test_mos`"))
    return out


BUILDERS = {"bvcc_ood": build_bvcc_ood, "nisqa": build_nisqa, "somos": build_somos,
            "tmhintqi": build_tmhintqi, "vmc23": build_vmc23, "urgent": build_urgent}


# --------------------------------------------------------------------------- #
# Checks + summary
# --------------------------------------------------------------------------- #
def check(b: Built, n_spot: int = 5) -> dict:
    """Sanity checks; raises on hard failures, returns summary stats."""
    df = b.df
    tag = f"{b.corpus}_{b.split}"
    assert list(df.columns) == COLUMNS, tag
    assert df["utt_id"].is_unique, f"{tag}: duplicate utt_id"
    assert df["path"].is_unique, f"{tag}: duplicate path"
    if b.expected is not None:
        assert len(df) == b.expected, f"{tag}: {len(df)} rows != official {b.expected}"
    assert df["mos"].notna().all(), f"{tag}: NaN MOS"
    assert df["mos"].between(1.0, 5.0).all(), f"{tag}: MOS outside [1, 5]"
    missing = [p for p in df["path"] if not os.path.exists(p)]
    assert not missing, f"{tag}: {len(missing)} missing files, e.g. {missing[:2]}"
    rng = random.Random(0)
    srs = set()
    for p in rng.sample(list(df["path"]), min(n_spot, len(df))):
        x, sr = sf.read(p, dtype="float32")
        assert x.size > 0 and np.isfinite(x).all(), f"{tag}: bad audio {p}"
        srs.add(sr)
    return {"corpus": b.corpus, "split": b.split, "n_utt": len(df),
            "n_sys": df["system"].nunique(), "mos_mean": df["mos"].mean(),
            "mos_std": df["mos"].std(), "mos_min": df["mos"].min(),
            "mos_max": df["mos"].max(), "sr": "/".join(str(s) for s in sorted(srs)),
            "source": b.source}


DOC_HEADER = """# E2 evaluation corpora

Generated by `scripts/build_eval_manifests.py`; manifests at
`{mdir}/{{corpus}}_{{split}}.csv` (columns `utt_id, path, mos, system, corpus, split`).
All labels are 5-point ACR MOS (1-5) as released; no rescaling. `sr (spot)` is the
native sample rate of the 5 spot-checked files (the loader resamples to 16 kHz).
Every manifest passed: row count == official count, unique ids, MOS in [1, 5],
all paths exist, 5 random files decode with soundfile.

| corpus | split | n_utt | n_sys | MOS mean | MOS std | MOS range | sr (spot) | source |
|---|---|---:|---:|---:|---:|---|---|---|
"""

DOC_NOTES = """
## Notes

- **bvcc_ood** = VMC'22 OOD track (Blizzard 2019 Mandarin); `system` = BVCC-style
  anonymised system id. The labeled 136-utt `train` and `val` lists are exported
  too; the 540-utt "unlabeled train" list (`unlabeled_mos_list_LABELS.txt`) is not.
  BVCC main (`bvcc_*`) is English only, so no utterance overlap with this set.
- **nisqa**: `mos` = overall-quality MOS column of `NISQA_TEST_*_file.csv`; `system`
  = `{db}_c{con}` (condition). Original 48 kHz `deg/` files are used.
- **somos**: split1 *clean* test list (listener-filtered MOS). The *full* list has the
  same 3000 utterances with different means (2969/3000 differ). `system` = the
  3-digit SOMOS system id (`000` = natural LJSpeech). Audio = `audios_16000`.
  Text overlap with BVCC main train: 127 SOMOS-test utterances (71 Blizzard sentence
  ids, e.g. `news_2008_0044`) share the *sentence* with BVCC-train samples (different
  voices/systems, so no audio duplicates); no LJSpeech sentence overlap.
- **tmhintqi**: MOS = mean of raw per-listener `quality_score` over rows of
  `raw_data.csv` whose wav is in the official `test/` dir (SHEET/MOS-Bench
  convention); `system` = `{method}-SNR{snr}-{noise}`.
- **vmc23**: split names follow the official challenge (arXiv 2310.02640):
  `track1a` = Blizzard 2023 French hub task (speaker NEB, 21 sys x 42),
  `track1b` = Blizzard 2023 spoke task (speaker AD, 17 sys x 34),
  `track2` = SVCC 2023 singing voice conversion (system = team id, e.g. `B01`,
  merging its in-domain/cross-domain conversions; `SOU` = source singing),
  `track3` = TMHINT-QI(S) noisy/enhanced Mandarin (97 sys = noise x SNR x SE method,
  incl. `clean_TMHINT`; ~20 utts each). Audio came from the
  organizers' Dropbox/Google-Drive packages; answers are the organizers' released
  keys (byte-identical to `unilight/sheet` `egs/vmc23/answers/`). The track3 answer
  file lists 255 extra un-prefixed `TMHINT_*` ids without audio; they are dropped.
  Of the 1596 track1 wavs, 136 (unrated extra utterances) have no answer and are dropped.
- **urgent24_sqa / urgent25_sqa**: HF `urgent-challenge/urgent2024-sqa` and
  `urgent-challenge/urgent2025-sqa`, split `blind_test_mos` (P.808 ACR via MTurk).
  2024: 23 system ids x 300 files, exactly 8 raw ratings each (card: noisy input +
  22 enhanced versions; all ids are anonymised `submission_*`, the noisy one is not
  flagged). 2025: 26 systems x 900 files (multilingual), no raw ratings released.
  `system` = submission id; `utt_id` = `{system_id}__fileid_XXXXXX` (the HF
  `sample_id` alone is not unique across systems). Audio bytes extracted unchanged
  (flac, 8-48 kHz) to
  `/fs/scratch/PAS2301/alialavi/datasets/audio_assessment_hackability_urgent2{4,5}_sqa/data/audio`.
  URGENT 2026 Track 2 uses `urgent2024-sqa/blind_test_mos` as its public test set.
"""


def write_doc(stats: list[dict], extra_notes: str = "") -> None:
    mdir = paths.cache_dir("manifests")
    lines = [DOC_HEADER.format(mdir=mdir)]
    for s in stats:
        lines.append(f"| {s['corpus']} | {s['split']} | {s['n_utt']} | {s['n_sys']} | "
                     f"{s['mos_mean']:.3f} | {s['mos_std']:.3f} | "
                     f"{s['mos_min']:.2f}-{s['mos_max']:.2f} | {s['sr']} | {s['source']} |\n")
    DOC.write_text("".join(lines) + DOC_NOTES + extra_notes)
    print("wrote", DOC)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=sorted(BUILDERS), default=sorted(BUILDERS))
    ap.add_argument("--no-doc", action="store_true", help="do not rewrite docs/eval_corpora.md")
    args = ap.parse_args()

    order = [k for k in BUILDERS if k in args.only]
    stats = []
    for key in order:
        print(f"== {key}", flush=True)
        for b in BUILDERS[key]():
            s = check(b)
            path = write_manifest_csv(b.df, b.corpus, b.split)
            stats.append(s)
            print(f"  {b.corpus}_{b.split}: n={s['n_utt']} sys={s['n_sys']} "
                  f"mos={s['mos_mean']:.3f}+-{s['mos_std']:.3f} "
                  f"[{s['mos_min']:.2f},{s['mos_max']:.2f}] sr={s['sr']} -> {path}", flush=True)
    if not args.no_doc:
        if len(order) < len(BUILDERS):
            print("partial build: not rewriting doc (run without --only)", file=sys.stderr)
        else:
            write_doc(stats)


if __name__ == "__main__":
    main()
