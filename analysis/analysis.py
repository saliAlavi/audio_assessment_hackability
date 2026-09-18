#!/usr/bin/env python3
"""Regenerates every reported number from run outputs listed in the campaign registry.

Outputs (markdown + CSV) go to ``analysis/out/``:
    r0_fidelity.md      R0 vs the VMC'22 SSL-MOS reference on BVCC test
    loop_rounds.md      per-round hack counts / ASR in the H&P loop
    table1_e1.md        hackability (ASR, score gain, judge deltas) per target × attacker
    table2_e2.md        correlation per predictor × corpus
    table3_e3.md        downstream GRPO enhancement results (if present)
    stats.md            decision-rule checks from preregistration.md

    python analysis/analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hackpatch.utils import paths  # noqa: E402

OUT = ROOT / "analysis" / "out"
PROJ = paths.SCRATCH_DIR / "projects" / paths.project_name()
SEEDS = [0, 1, 2]
TARGET_ORDER = ["R0", "R1", "R2", "R3", "R3ms", "baseline_noise", "baseline_pgd", "baseline_random"]


def registry() -> dict:
    return json.loads((PROJ / "_campaign.json").read_text())


def mean_sd(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "–"
    if len(x) == 1:
        return f"{x.iloc[0]:.3f} (n=1)"
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f} (n={len(x)})"


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def loop_rounds(reg: dict) -> pd.DataFrame:
    rows = []
    for key, v in reg.items():
        if v.get("kind") not in ("hnp", "ablation", "multispace"):
            continue
        for k in (1, 2, 3):
            f = Path(v["run_dir"]) / f"round{k}" / "done.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            row = {"run": key, "round": k, "n_pairs_round": d["n_pairs_round"],
                   "n_pairs_buffer": d["n_pairs_buffer"],
                   "train_asr": d.get(f"round{k}/train_asr"), "val_asr": d.get(f"round{k}/val_asr"),
                   "val_mean_gain": d.get(f"round{k}/val_mean_gain"),
                   "val_ceiling_rate": d.get(f"round{k}/val_ceiling_rate")}
            # Multi-space rounds log one block of statistics per attack space.
            for space in ("encodec", "dac", "pgd"):
                if f"round{k}/{space}/train_asr" in d:
                    row[f"train_asr_{space}"] = d[f"round{k}/{space}/train_asr"]
                    row[f"val_asr_{space}"] = d.get(f"round{k}/{space}/val_asr")
            rows.append(row)
    return pd.DataFrame(rows)


def e1_table(reg: dict) -> pd.DataFrame:
    rows = []
    for key, v in reg.items():
        if v.get("kind") != "e1":
            continue
        f = Path(v["run_dir"]) / "results" / "e1_summary.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        seed = int(key.rsplit("_s", 1)[1])
        for att, st in d["attackers"].items():
            rows.append({"target": d["target"], "attacker": att, "seed": seed, **st})
    return pd.DataFrame(rows)


def e2_table(reg: dict) -> pd.DataFrame:
    frames = []
    for key, v in reg.items():
        if v.get("kind") != "e2":
            continue
        f = Path(v["run_dir"]) / "results" / "correlation.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(seed=int(key.rsplit("_s", 1)[1])))
    # predictor names are unique per (predictor, seed) job
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def e3_tables(reg: dict) -> tuple[pd.DataFrame, list[str]]:
    """E3: per-system means and the preregistered count of degraded external measures.

    The primary E3 metric counts, per GRPO system, how many of DNSMOS OVRL, NISQA, UTMOS,
    WER and speaker similarity are worse than the pretrained enhancer with a paired
    bootstrap 95% CI on the per-utterance difference excluding 0. PESQ, ESTOI and SI-SDR
    are reported as secondary intrusive measures.

    Returns:
        (long frame of per-system metrics with seeds, list of decision-rule lines).
    """
    prereg = {"dnsmos_ovrl": 1, "nisqa_mos": 1, "utmos": 1, "spk_sim": 1, "wer": -1}
    secondary = {"pesq": 1, "estoi": 1, "sisdr": 1}
    rows, report = [], []
    for key, v in reg.items():
        if v.get("kind") not in ("e3_eval", "e3_eval_r2"):
            continue
        f = Path(v["run_dir"]) / "results" / "per_utterance.csv"
        if not f.exists():
            continue
        seed = int(key.rsplit("_s", 1)[1])
        df = pd.read_csv(f)
        base = df[df.checkpoint == "pretrained"].set_index("utt_id")
        for sysname in df.checkpoint.unique():
            if not sysname.startswith("grpo_"):
                continue
            cur = df[df.checkpoint == sysname].set_index("utt_id")
            reward = sysname[len("grpo_"):]
            row = {"seed": seed, "reward": reward,
                   "reward_rise": float(cur[f"score_{reward}"].mean() - base[f"score_{reward}"].mean())}
            n_worse = 0
            for m, sign in {**prereg, **secondary}.items():
                d = (cur[m] - base.loc[cur.index, m]).dropna()
                lo, hi = boot_ci(d.to_numpy())
                row[f"{m}_delta"] = float(d.mean())
                worse = bool((sign > 0 and hi < 0) or (sign < 0 and lo > 0))
                row[f"{m}_worse"] = worse
                if m in prereg and worse:
                    n_worse += 1
            row["n_external_worse"] = n_worse
            rows.append(row)
    e3 = pd.DataFrame(rows)
    if len(e3):
        piv = e3.pivot_table(index="reward", values=["n_external_worse", "reward_rise"], aggfunc="mean")
        if {"R0", "R3"} <= set(piv.index):
            report.append(f"H3: degraded external measures (mean over seeds) R0-reward {piv.loc['R0', 'n_external_worse']:.2f} "
                          f"vs R3-reward {piv.loc['R3', 'n_external_worse']:.2f}; reward rise "
                          f"{piv.loc['R0', 'reward_rise']:+.3f} / {piv.loc['R3', 'reward_rise']:+.3f}; supported = "
                          f"{bool(piv.loc['R3', 'n_external_worse'] < piv.loc['R0', 'n_external_worse'] and piv.loc['R3', 'reward_rise'] > 0)}")
    return e3, report


def paper_numbers(reg: dict, e1: pd.DataFrame, e2: pd.DataFrame, e3: pd.DataFrame) -> dict:
    """Dumps every scalar quoted in the paper prose, so the provenance linter can trace it.

    Values are computed here (never typed), rounded the way the prose prints them.
    """
    out: dict[str, float] = {}

    def put(key: str, value, digits: int = 3) -> None:
        out[key] = round(float(value), digits)

    if len(e2):
        e2 = e2.copy()
        e2["set"] = e2.corpus + "/" + e2.split
        ind = e2[e2.set == "bvcc/test"]
        for metric in ("utt_srcc", "sys_srcc"):
            g = ind[ind.predictor == "R0"][metric]
            put(f"r0_bvcc_{metric}", g.mean())
            put(f"r0_bvcc_{metric}_sd", g.std(ddof=1))
        ood = e2[e2.set != "bvcc/test"].groupby(["predictor", "seed"]).sys_srcc.mean().groupby("predictor").mean()
        for p_ in ood.index:
            put(f"ood_sys_srcc_{p_}", ood[p_])
        w = ind.pivot_table(index="seed", columns="predictor", values="sys_srcc")
        if {"R0", "R3"} <= set(w.columns):
            put("bvcc_sys_srcc_R3_minus_R0", (w["R3"] - w["R0"]).mean())
        per_set = e2.pivot_table(index="set", columns="predictor", values="sys_srcc")
        if {"R0", "R3"} <= set(per_set.columns):
            for st in per_set.index:
                put(f"ood_drop_{st}", per_set.loc[st, "R3"] - per_set.loc[st, "R0"])

    if len(e1):
        a = e1.groupby(["target", "attacker"]).asr.agg(["mean", "std"])
        g = e1.groupby(["target", "attacker"]).mean_gain.agg(["mean", "std"])
        for (t, att), row in a.iterrows():
            put(f"asr_{t}_{att}", row["mean"])
            put(f"asr_{t}_{att}_sd", row["std"])
            put(f"asr_{t}_{att}_2dp", row["mean"], 2)
        for _, r in e1.iterrows():  # per-seed values quoted when the seed spread is the point
            put(f"asr_{r['target']}_{r['attacker']}_seed{int(r['seed'])}", r["asr"])
        for (t, att), row in g.iterrows():
            put(f"gain_{t}_{att}", row["mean"], 2)
        w = e1[e1.attacker == "grpo_encodec"].pivot(index="seed", columns="target", values="asr")
        if {"R0", "R3"} <= set(w.columns):
            put("h1_asr_drop", (w["R0"] - w["R3"]).mean())

    if len(e3):
        for reward, g in e3.groupby("reward"):
            for col, digits in [("reward_rise", 3), ("pesq_delta", 3), ("estoi_delta", 3), ("sisdr_delta", 1),
                                ("n_external_worse", 2)]:
                put(f"e3_{reward}_{col}", g[col].mean(), digits)
                # Two-decimal variant for the abstract, so both roundings trace.
                put(f"e3_{reward}_{col}_2dp", g[col].mean(), 2)

    # Cross-scores of the E3 systems (mean over seeds), and in-loop detector statistics.
    cross, loop_stats = [], []
    for key, v in reg.items():
        if v.get("kind") == "e3_eval":
            f = Path(v["run_dir"]) / "results" / "summary_wide.csv"
            if f.exists():
                cross.append(pd.read_csv(f).set_index("checkpoint"))
        if v.get("kind") == "hnp":
            f = Path(v["run_dir"]) / "round1" / "attack_train.csv"
            if f.exists():
                loop_stats.append(pd.read_csv(f))
    if cross:
        c = pd.concat(cross).groupby(level=0).mean(numeric_only=True)
        for sysname in ("grpo_R0", "grpo_R3", "pretrained"):
            if sysname in c.index and "score_R3" in c.columns:
                put(f"e3_R3_scores_{sysname}", c.loc[sysname, "score_R3"], 2)
    if loop_stats:
        d = pd.concat(loop_stats)
        m = d[d.cand == 0]
        put("hack_input_score_mean", m[m.hack].score_input.mean(), 1)
        put("all_input_score_mean", m.score_input.mean(), 1)
        h = m[m.ceiling]
        put("panel_estoi_delta", (h.estoi_attacked - h.estoi_recon).mean(), 2)
        put("panel_sisdr_delta", (h.sisdr_attacked - h.sisdr_recon).mean(), 1)

    # ε pilot (validation only), read from the pilot run directory.
    pilot = PROJ / "pilot_eps3"
    for f in sorted(pilot.glob("h0_e*.json")) if pilot.exists() else []:
        rows = json.loads(f.read_text())
        for r in rows:
            put(f"pilot_ceiling_eps{r['eps']}", 100 * r["val_ceiling_rate@0.3"], 1)

    # Corrected-reference E1 statistics (analysis/recompute_e1.py), both roundings.
    both = OUT / "e1_both_refs.csv"
    if both.exists():
        b = pd.read_csv(both)
        for (t, att), g in b.groupby(["target", "attacker"]):
            for col in ("asr_ref_recon", "asr_ref_input", "recon_gain", "gain_ref_recon", "ceiling_ref_recon",
                        "panel_rate", "estoi_delta", "sisdr_delta", "wer_delta",
                        "judge_utmos_delta", "judge_dnsmos_ovrl_delta"):
                if col in g:
                    for d in (3, 2):
                        put(f"e1_{t}_{att}_{col}_{d}dp", g[col].mean(), d)
                    put(f"e1_{t}_{att}_{col}_sd", g[col].std(ddof=1))
    lrb = OUT / "loop_rounds_both_refs.csv"
    if lrb.exists():
        for _, r in pd.read_csv(lrb).iterrows():
            put(f"loop_{r['run']}_r{int(r['round'])}_{r['space']}_asr", r["asr_ref_recon"])
    lr = OUT / "loop_rounds.csv"
    if lr.exists():
        d = pd.read_csv(lr)
        for run, g in d[d["round"] == 3].groupby("run"):
            put(f"buffer_{run}", g.n_pairs_buffer.iloc[0], 0)
            put(f"buffer_k_{run}", g.n_pairs_buffer.iloc[0] / 1000, 1)

    # Attacker-variance control: five attacker seeds against one fixed predictor.
    av = OUT / "e1_attacker_variance.csv"
    if av.exists():
        a = pd.read_csv(av)
        for t, g in a.groupby("target"):
            put(f"av_{t}_mean", g.asr_ref_recon.mean())
            put(f"av_{t}_sd", g.asr_ref_recon.std(ddof=1))
            put(f"av_{t}_max", g.asr_ref_recon.max())
            put(f"av_{t}_min", g.asr_ref_recon.min())
            for col in ("mean", "sd", "max", "min"):
                put(f"av_{t}_{col}_2dp", getattr(g.asr_ref_recon, col if col != "sd" else "std")(
                    **({"ddof": 1} if col == "sd" else {})), 2)

    # Published-predictor benchmark.
    bs = OUT / "bench_summary.csv"
    if bs.exists():
        for _, r in pd.read_csv(bs).iterrows():
            for c in ("asr_ref_recon_mean", "asr_ref_recon_max", "asr_ref_recon_min", "asr_ref_input_mean",
                      "gain_ref_recon_mean", "recon_gain_mean", "asr_ref_recon_std"):
                if c in r:
                    for d in (3, 2):
                        put(f"bench_{r['target']}_{c}_{d}dp", r[c], d)
                    if c.startswith("asr"):  # percentages as quoted in the prose
                        put(f"bench_{r['target']}_{c}_pct", 100 * r[c], 1)
                        put(f"bench_{r['target']}_{c}_pct0", 100 * r[c], 0)

    # Compute ledger.
    ledger = PROJ / "_compute_gpu_hours.txt"
    if ledger.exists():
        put("gpu_hours", float(ledger.read_text().strip()), 0)
    (OUT / "paper_numbers.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = registry()
    report = []

    lr = loop_rounds(reg)
    if len(lr):
        lr.to_csv(OUT / "loop_rounds.csv", index=False)
        (OUT / "loop_rounds.md").write_text(lr.to_markdown(index=False, floatfmt=".3f"))

    e1 = e1_table(reg)
    if len(e1):
        e1.to_csv(OUT / "e1_long.csv", index=False)
        metrics = ["asr", "mean_gain", "ceiling_rate", "judge_utmos_delta", "judge_dnsmos_ovrl_delta"]
        tab = (e1.groupby(["target", "attacker"])[[m for m in metrics if m in e1]]
               .agg(mean_sd).reset_index())
        tab["target"] = pd.Categorical(tab["target"], TARGET_ORDER, ordered=True)
        tab = tab.sort_values(["attacker", "target"])
        (OUT / "table1_e1.md").write_text(tab.to_markdown(index=False))
        # H1 decision rule: fresh GRPO-EnCodec ASR, R0 - R3 paired by seed.
        p = e1[e1.attacker == "grpo_encodec"].pivot(index="seed", columns="target", values="asr")
        if {"R0", "R3"} <= set(p.columns):
            d = (p["R0"] - p["R3"]).dropna()
            report.append(f"H1: ASR(R0)-ASR(R3) per seed = {d.round(3).tolist()}, mean {d.mean():.3f}; "
                          f"supported = {bool(d.mean() >= 0.10 and (d > 0).all())} (rule: mean ≥ 0.10 and all seeds > 0)")
            for att in sorted(set(e1.attacker) - {"grpo_encodec"}):
                q = e1[e1.attacker == att].pivot(index="seed", columns="target", values="asr")
                if {"R0", "R3"} <= set(q.columns):
                    dd = (q["R0"] - q["R3"]).dropna()
                    report.append(f"H1-transfer [{att}]: ASR drop per seed {dd.round(3).tolist()}, "
                                  f"mean {dd.mean():.3f}; transfers = {bool(dd.mean() >= 0.10)}")

    e2 = e2_table(reg)
    if len(e2):
        e2.to_csv(OUT / "e2_long.csv", index=False)
        cols = ["utt_mse", "utt_lcc", "utt_srcc", "sys_mse", "sys_lcc", "sys_srcc"]
        tab = e2.groupby(["corpus", "split", "predictor"])[[c for c in cols if c in e2]].agg(mean_sd).reset_index()
        (OUT / "table2_e2.md").write_text(tab.to_markdown(index=False))
        b = e2[(e2.corpus == "bvcc") & (e2.split == "test")].pivot(index="seed", columns="predictor", values="sys_srcc")
        if {"R0", "R3"} <= set(b.columns):
            d = (b["R3"] - b["R0"]).dropna()
            report.append(f"H2: BVCC-test sys-SRCC R3-R0 per seed = {d.round(4).tolist()}, mean {d.mean():.4f}; "
                          f"non-inferior = {bool(d.mean() >= -0.02)} (rule: mean ≥ -0.02)")
        # Preregistered secondary analysis: R3-R0 on every corpus, Holm-Bonferroni over corpora.
        from scipy import stats as _st
        rows = []
        for (corpus, split), g in e2.groupby(["corpus", "split"]):
            w = g.pivot_table(index="seed", columns="predictor", values="sys_srcc")
            if not {"R0", "R3"} <= set(w.columns):
                continue
            d = (w["R3"] - w["R0"]).dropna()
            t = _st.ttest_rel(w["R3"].dropna(), w["R0"].dropna()) if len(d) > 1 else None
            rows.append({"corpus": f"{corpus}/{split}", "n_seeds": len(d), "mean_diff": d.mean(),
                         "all_seeds_same_sign": bool((d > 0).all() or (d < 0).all()),
                         "p_raw": float(t.pvalue) if t is not None else float("nan")})
        sec = pd.DataFrame(rows).sort_values("mean_diff")
        if len(sec):
            order = sec["p_raw"].rank(method="first")
            m = len(sec)
            sec["p_holm"] = (sec["p_raw"] * (m - order + 1)).clip(upper=1.0)
            sec.to_csv(OUT / "e2_r3_minus_r0.csv", index=False)
            (OUT / "e2_r3_minus_r0.md").write_text(sec.to_markdown(index=False, floatfmt=".4f"))
            worse = sec[(sec.mean_diff < 0) & (sec.p_holm < 0.05)]
            report.append("H2 secondary: mean sys-SRCC over the 13 eval sets R0 "
                          f"{e2[e2.predictor == 'R0'].groupby(['corpus','split']).sys_srcc.mean().mean():.3f} vs R3 "
                          f"{e2[e2.predictor == 'R3'].groupby(['corpus','split']).sys_srcc.mean().mean():.3f}; "
                          f"R3 significantly worse (Holm p<0.05) on {len(worse)}/{len(sec)} eval sets: "
                          + ", ".join(worse.corpus.tolist()))
        r0 = e2[(e2.corpus == "bvcc") & (e2.split == "test") & (e2.predictor == "R0")]
        (OUT / "r0_fidelity.md").write_text(
            "VMC'22 B01 SSL-MOS reference (BVCC test): utt SRCC ≈ 0.87, sys SRCC ≈ 0.92 (tolerance ±0.03)\n\n"
            + r0[["seed", "utt_mse", "utt_srcc", "sys_mse", "sys_srcc"]].to_markdown(index=False, floatfmt=".3f"))

    e3, e3_report = e3_tables(reg)
    if len(e3):
        e3.to_csv(OUT / "e3_long.csv", index=False)
        cols = ["reward_rise", "dnsmos_ovrl_delta", "nisqa_mos_delta", "utmos_delta", "spk_sim_delta",
                "wer_delta", "pesq_delta", "estoi_delta", "sisdr_delta", "n_external_worse"]
        tab = e3.groupby("reward")[cols].agg(mean_sd).reset_index()
        (OUT / "table3_e3.md").write_text(tab.to_markdown(index=False))
        report += e3_report

    (OUT / "stats.md").write_text("\n\n".join(report) if report else "no decision-rule inputs yet")

    paper_numbers(reg, e1 if len(e1) else pd.DataFrame(), e2 if len(e2) else pd.DataFrame(),
                  e3 if len(e3) else pd.DataFrame())

    # One document with every table, so the report never quotes a number by hand.
    doc = ["# Results (generated by analysis/analysis.py)", "",
           f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Seeds: {SEEDS}. "
           "Every value is mean ± sd over seeds (n given); attack/eval CIs are in the per-run CSVs.", "",
           "## Decision rules (preregistration.md)", ""]
    doc += [f"- {line}" for line in report]
    for title, path in [("R0 baseline reproduction", "r0_fidelity.md"),
                        ("Loop rounds (in-loop attacker vs the current predictor)", "loop_rounds.md"),
                        ("Table 1 — E1 hackability (BVCC test audio)", "table1_e1.md"),
                        ("Table 2 — E2 agreement with human MOS", "table2_e2.md"),
                        ("Table 3 — E3 downstream GRPO enhancement", "table3_e3.md")]:
        f = OUT / path
        if f.exists():
            doc += ["", f"## {title}", "", f.read_text()]
    (ROOT / "docs" / "results.md").write_text("\n".join(doc))
    print("\n".join(report))


if __name__ == "__main__":
    main()
