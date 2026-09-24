#!/usr/bin/env python3
"""Recomputes E1 from stored per-utterance attack CSVs under both ceiling references.

The ceiling rule as run compares the attacked score with the *raw* input,
``R(y)-R(x)>tau``. Because patching also moves where the predictor places the
zero-perturbation codec reconstruction ``x_hat``, that reference confounds
"the attack worked" with "the defence shifted its reconstruction baseline".
This script recomputes every E1 statistic with ``R(y)-R(x_hat)>tau`` as well,
using the panel columns already stored per utterance. No GPU, no re-running.

    python analysis/recompute_e1.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hackpatch.detect.detector import MEASURE_SIGN  # noqa: E402
from hackpatch.repro import load_registry  # noqa: E402
from hackpatch.utils import paths  # noqa: E402

# Overridable so the notebooks can point the analysis at released or recomputed run outputs.
OUT = Path(os.environ.get("HACKPATCH_ANALYSIS_OUT", ROOT / "analysis" / "out"))
PROJ = Path(os.environ.get("HACKPATCH_RUNS_DIR", paths.SCRATCH_DIR / "projects" / paths.project_name()))
TAU = 0.3
MARGINS = {"estoi": 0.02, "sisdr": 1.0, "wer": 0.0, "sbs": 0.02}


def panel_fires(df: pd.DataFrame) -> pd.Series:
    """Recomputes the panel rule from the stored measure columns."""
    fires = pd.Series(False, index=df.index)
    for m, sign in MEASURE_SIGN.items():
        if f"{m}_attacked" in df and f"{m}_recon" in df:
            fires |= sign * (df[f"{m}_recon"] - df[f"{m}_attacked"]) > MARGINS[m]
    return fires


def attacker_variance(reg: dict) -> pd.DataFrame:
    """Attacker-seed control: five fresh EnCodec attackers against one fixed predictor (seed 0)."""
    rows = []
    for key, v in reg.items():
        if v.get("kind") != "e1_attacker_seeds":
            continue
        for f in sorted(Path(v["run_dir"], "results").glob("attack_grpo_encodec_a*.csv")):
            d = pd.read_csv(f)
            m = d[d.cand == 0]
            panel = panel_fires(m)
            rows.append({"target": key.split("_")[1], "attacker": f.stem.rsplit("_", 1)[1],
                         "asr_ref_recon": float((((m.score_attacked - m.score_recon) > TAU) & panel).mean()),
                         "asr_ref_input": float((((m.score_attacked - m.score_input) > TAU) & panel).mean())})
    av = pd.DataFrame(rows)
    if len(av):
        av.to_csv(OUT / "e1_attacker_variance.csv", index=False)
    return av


def main() -> None:
    reg = load_registry(PROJ)
    rows = []
    for key, v in reg.items():
        if v.get("kind") != "e1":
            continue
        seed = int(key.rsplit("_s", 1)[1])
        for f in sorted(Path(v["run_dir"], "results").glob("attack_*.csv")):
            d = pd.read_csv(f)
            m = d[d.cand == 0].copy()
            if not len(m):
                continue
            panel = panel_fires(m)
            gain_x = m.score_attacked - m.score_input
            gain_r = m.score_attacked - m.score_recon
            row = {
                "target": v["target"], "attacker": f.stem[len("attack_"):], "seed": seed, "n_utt": len(m),
                "recon_gain": float((m.score_recon - m.score_input).mean()),
                "asr_ref_input": float(((gain_x > TAU) & panel).mean()),
                "asr_ref_recon": float(((gain_r > TAU) & panel).mean()),
                "ceiling_ref_input": float((gain_x > TAU).mean()),
                "ceiling_ref_recon": float((gain_r > TAU).mean()),
                "panel_rate": float(panel.mean()),
                "gain_ref_input": float(gain_x.mean()),
                "gain_ref_recon": float(gain_r.mean()),
            }
            for m_ in MEASURE_SIGN:
                if f"{m_}_attacked" in m:
                    row[f"{m_}_delta"] = float((m[f"{m_}_attacked"] - m[f"{m_}_recon"]).mean())
            for c in [c for c in m.columns if c.startswith("judge_") and c.endswith("_attacked")]:
                base = c[len("judge_"):-len("_attacked")]
                row[f"judge_{base}_delta"] = float((m[c] - m[f"judge_{base}_recon"]).mean())
            rows.append(row)
    e1 = pd.DataFrame(rows)
    e1.to_csv(OUT / "e1_both_refs.csv", index=False)

    def tab(value: str) -> pd.DataFrame:
        return e1.pivot_table(index="target", columns="attacker", values=value)

    order = ["R0", "R1", "R2", "R3", "R3ms", "baseline_noise", "baseline_pgd", "baseline_random"]
    lines = ["# E1 under both ceiling references", "",
             f"tau={TAU}; panel margins {MARGINS}; mean over seeds; mean-action candidate only.", ""]
    for name, val in [("Predictor's own reconstruction offset R(x_hat)-R(x)", "recon_gain"),
                      ("ASR, ceiling referenced to raw input x (as run)", "asr_ref_input"),
                      ("ASR, ceiling referenced to codec reconstruction x_hat (corrected)", "asr_ref_recon"),
                      ("Attacker gain over x_hat", "gain_ref_recon")]:
        t = tab(val).reindex(order).dropna(how="all")
        lines += [f"## {name}", "", t.round(3).to_markdown(), ""]
    (OUT / "e1_both_refs.md").write_text("\n".join(lines))

    # Threshold sweep: is the ranking of defences an artifact of tau (patch margin m=0.5 > tau=0.3)?
    sweep = []
    for key, v in reg.items():
        if v.get("kind") != "e1":
            continue
        seed = int(key.rsplit("_s", 1)[1])
        f = Path(v["run_dir"], "results", "attack_grpo_encodec.csv")
        if not f.exists():
            continue
        d = pd.read_csv(f)
        m = d[d.cand == 0]
        panel = panel_fires(m)
        gap = m.score_attacked - m.score_recon
        for t in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            sweep.append({"target": v["target"], "seed": seed, "tau": t,
                          "asr": float(((gap > t) & panel).mean())})
        sweep.append({"target": v["target"], "seed": seed, "tau": "median_gap", "asr": float(gap.median())})
        sweep.append({"target": v["target"], "seed": seed, "tau": "p90_gap", "asr": float(gap.quantile(0.9))})
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "e1_tau_sweep.csv", index=False)
    piv = sw.pivot_table(index="target", columns="tau", values="asr")
    (OUT / "e1_tau_sweep.md").write_text(
        "# Corrected-reference ASR vs detection threshold (fresh EnCodec attacker, mean over seeds)\n\n"
        + piv.round(3).to_markdown() + "\n")
    print(piv.round(3).to_string(), "\n")

    # In-loop rounds, recomputed under both references so the round figure is consistent.
    lrows = []
    for key, v in reg.items():
        if v.get("kind") not in ("hnp", "multispace", "ablation"):
            continue
        for k in (1, 2, 3):
            rd = Path(v["run_dir"]) / f"round{k}"
            for f in sorted(rd.glob("attack_train*.csv")):
                d = pd.read_csv(f)
                m = d[d.cand == 0]
                if not len(m):
                    continue
                space = f.stem.replace("attack_train", "").lstrip("_") or "encodec"
                panel = panel_fires(m)
                lrows.append({"run": key, "round": k, "space": space,
                              "asr_ref_input": float((((m.score_attacked - m.score_input) > TAU) & panel).mean()),
                              "asr_ref_recon": float((((m.score_attacked - m.score_recon) > TAU) & panel).mean()),
                              "recon_gain": float((m.score_recon - m.score_input).mean()),
                              "n_pairs": int(m.hack.sum()) if "hack" in m else np.nan})
    pd.DataFrame(lrows).to_csv(OUT / "loop_rounds_both_refs.csv", index=False)

    attacker_variance(reg)

    # Paired per-seed differences against R0 under each reference.
    report = []
    for ref in ("asr_ref_input", "asr_ref_recon"):
        w = e1[e1.attacker == "grpo_encodec"].pivot(index="seed", columns="target", values=ref)
        for t in ("R3", "R3ms", "baseline_random"):
            if t in w.columns:
                d = (w["R0"] - w[t]).dropna()
                rng = np.random.default_rng(0)
                lo, hi = (np.percentile([rng.choice(d, len(d)).mean() for _ in range(10000)], [2.5, 97.5])
                          if len(d) > 1 else (np.nan, np.nan))
                report.append(f"{ref}: ASR(R0)-ASR({t}) per seed {np.round(d.values, 3).tolist()}, "
                              f"mean {d.mean():+.3f}, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    (OUT / "e1_reference_check.md").write_text("\n".join(report))
    print("\n".join(report))
    print("\nwrote", OUT / "e1_both_refs.csv", OUT / "e1_both_refs.md")


if __name__ == "__main__":
    main()
