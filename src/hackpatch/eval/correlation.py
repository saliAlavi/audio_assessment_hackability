"""Utterance- and system-level agreement with human MOS, with bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(np.mean((pred - true) ** 2)),
        "lcc": float(stats.pearsonr(pred, true)[0]),
        "srcc": float(stats.spearmanr(pred, true)[0]),
        "ktau": float(stats.kendalltau(pred, true)[0]),
    }


def correlation_report(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Computes VMC-style metrics with percentile bootstrap 95% CIs.

    Utterance-level CIs resample utterances; system-level CIs resample systems
    (the unit that system-level metrics generalize over).

    Args:
        df: Frame with ``pred``, ``mos`` and ``system`` columns.
        n_boot: Bootstrap resamples.
        seed: RNG seed for the bootstrap.

    Returns:
        Flat dict like ``utt_srcc``, ``utt_srcc_lo``, ``utt_srcc_hi``, ``sys_*``,
        plus ``n_utt`` and ``n_sys``.
    """
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=["pred", "mos"])
    out: dict[str, float] = {"n_utt": len(df)}
    pred, true = df["pred"].to_numpy(), df["mos"].to_numpy()
    point = _metrics(pred, true)
    boots = [_metrics(pred[ix], true[ix])
             for ix in (rng.integers(0, len(df), len(df)) for _ in range(n_boot))]
    for k, v in point.items():
        vals = np.array([b[k] for b in boots])
        out[f"utt_{k}"], out[f"utt_{k}_lo"], out[f"utt_{k}_hi"] = v, *np.nanpercentile(vals, [2.5, 97.5])

    sys_df = df.groupby("system")[["pred", "mos"]].mean()
    out["n_sys"] = len(sys_df)
    if len(sys_df) >= 4:
        sp, st = sys_df["pred"].to_numpy(), sys_df["mos"].to_numpy()
        point = _metrics(sp, st)
        boots = [_metrics(sp[ix], st[ix])
                 for ix in (rng.integers(0, len(sys_df), len(sys_df)) for _ in range(n_boot))]
        for k, v in point.items():
            vals = np.array([b[k] for b in boots])
            out[f"sys_{k}"], out[f"sys_{k}_lo"], out[f"sys_{k}_hi"] = v, *np.nanpercentile(vals, [2.5, 97.5])
    return out
