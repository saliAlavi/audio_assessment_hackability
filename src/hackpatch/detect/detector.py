"""Label-free hack detection: ceiling rule AND panel rule.

* Ceiling rule: ``R(y) - R(x) > τ`` (the attacked output scores above the
  clean, unattacked input by more than τ MOS).
* Panel rule: at least one independent measure is worse for ``y`` than for the
  zero-perturbation codec reconstruction ``x̂``, by more than a per-measure
  margin. Intrusive measures (ESTOI, SI-SDR, SpeechBERTScore) use ``x`` as the
  reference; WER compares Whisper(``y``) and Whisper(``x̂``) against Whisper(``x``).
  Comparing against ``x̂`` rather than ``x`` isolates what the perturbation did
  from the codec's own round-trip loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

#: measure -> sign; +1 means "higher is better".
MEASURE_SIGN = {"estoi": 1, "sisdr": 1, "sbs": 1, "wer": -1}


def panel_scores(panel, audio: dict, keys: list[tuple[str, int]], batch: int = 64) -> pd.DataFrame:
    """Scores reconstruction and attacked audio against the clean input.

    Args:
        panel: A ``hackpatch.metrics.panel.Panel`` instance.
        audio: ``{(utt_id, cand): (x, x_hat, y)}`` 1-D CPU tensors.
        keys: Which entries to score.
        batch: Pairs per panel call.

    Returns:
        Frame indexed like ``keys`` with ``{measure}_recon`` and ``{measure}_attacked``.
    """
    recon_cache: dict[str, dict] = {}
    rows = []
    for i in range(0, len(keys), batch):
        chunk = keys[i:i + batch]
        todo = [k for k in chunk if k[0] not in recon_cache]
        seen = set()
        todo = [k for k in todo if not (k[0] in seen or seen.add(k[0]))]
        if todo:
            rec = panel([audio[k][0] for k in todo], [audio[k][1] for k in todo])
            for k, (_, r) in zip(todo, rec.iterrows()):
                recon_cache[k[0]] = r.to_dict()
        att = panel([audio[k][0] for k in chunk], [audio[k][2] for k in chunk])
        for k, (_, a) in zip(chunk, att.iterrows()):
            row = {"utt_id": k[0], "cand": k[1]}
            row.update({f"{m}_recon": v for m, v in recon_cache[k[0]].items()})
            row.update({f"{m}_attacked": v for m, v in a.to_dict().items()})
            rows.append(row)
    return pd.DataFrame(rows)


def flag_hacks(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Applies the ceiling and panel rules.

    Args:
        df: Attack results merged with :func:`panel_scores` output.
        cfg: ``detect`` config: ``tau``, ``margins`` {measure: margin}, ``measures``.

    Returns:
        ``df`` with boolean ``ceiling``, ``panel_{measure}``, ``panel`` and ``hack`` columns.
    """
    df = df.copy()
    df["gain"] = df["score_attacked"] - df["score_input"]
    df["ceiling"] = df["gain"] > cfg.tau
    worse = []
    for m in cfg.measures:
        sign, margin = MEASURE_SIGN[m], cfg.margins[m]
        col = f"panel_{m}"
        df[col] = sign * (df[f"{m}_recon"] - df[f"{m}_attacked"]) > margin
        worse.append(df[col].to_numpy())
    df["panel"] = np.any(np.stack(worse), axis=0) if worse else True
    df["hack"] = df["ceiling"] & df["panel"]
    return df
