#!/usr/bin/env python3
"""Collects per-predictor benchmark summaries into analysis/out/bench_summary.csv."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hackpatch.utils import paths  # noqa: E402

OUT = ROOT / "analysis" / "out"
PROJ = paths.SCRATCH_DIR / "projects" / paths.project_name()
ORDER = ["sslmos", "utmos", "nisqa", "dnsmos"]


def main() -> None:
    reg = json.loads((PROJ / "_campaign.json").read_text())
    rows = []
    for key, v in reg.items():
        if v.get("kind") != "bench":
            continue
        f = Path(v["run_dir"]) / "results" / "bench_summary.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    if not rows:
        print("no benchmark summaries yet")
        return
    df = pd.DataFrame(rows)
    df["order"] = df.target.map({t: i for i, t in enumerate(ORDER)}).fillna(99)
    df = df.sort_values("order").drop(columns="order")
    df.to_csv(OUT / "bench_summary.csv", index=False)
    cols = ["target", "recon_gain_mean", "asr_ref_recon_mean", "asr_ref_recon_max", "asr_ref_recon_min",
            "gain_ref_recon_mean", "asr_ref_input_mean", "n_attackers", "n_utts"]
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
