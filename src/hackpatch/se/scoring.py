"""Enhancement, predictor scoring and the external metric suite for E3.

Everything here works on full-length utterances one at a time: the SSL-MOS
predictors (GroupNorm front-end) and the BiGRU enhancement model both change
their output under zero-padding, so unpadded single-utterance passes are the
only exact option (the model is tiny and VBD utterances are short).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from hackpatch.models.ssl_mos import SSLMOS
from hackpatch.se.model import MaskNet

#: Pseudo-checkpoint names that do not load a model.
PSEUDO_CHECKPOINTS = ("noisy", "clean")


@torch.no_grad()
def enhance_list(model: MaskNet, wavs: list[np.ndarray], device: str = "cuda") -> list[np.ndarray]:
    """Enhances utterances with the deterministic (mean) policy.

    Args:
        model: Enhancement model (put in eval mode by the caller).
        wavs: Noisy 1-D float32 waveforms.
        device: Device.

    Returns:
        Enhanced waveforms, same lengths, float32.
    """
    out = []
    for w in wavs:
        y = model(torch.from_numpy(w).to(device)[None])[0]
        out.append(y.float().cpu().numpy())
    return out


@torch.no_grad()
def score_predictor(predictor: SSLMOS, wavs: list[np.ndarray], device: str = "cuda") -> np.ndarray:
    """Scores utterances with an SSL-MOS predictor (bf16, unpadded).

    Args:
        predictor: Predictor in eval mode.
        wavs: 1-D 16 kHz waveforms.
        device: Device.

    Returns:
        ``[N]`` predicted MOS.
    """
    scores = np.zeros(len(wavs), dtype=np.float32)
    for i, w in enumerate(wavs):
        x = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores[i] = predictor(x).float().item()
    return scores


class MetricSuite:
    """External measures of enhanced speech against clean references and transcripts.

    Wraps :class:`hackpatch.metrics.external.ExternalMetrics` (non-intrusive MOS
    estimators, speaker similarity, PESQ) and :class:`hackpatch.metrics.panel.Panel`
    (ESTOI, SI-SDR, Whisper WER against the transcript). Built from the
    ``metrics`` config section.

    Args:
        external: Measures for ``ExternalMetrics`` (subset of dnsmos, nisqa,
            utmos, spk_sim, pesq). Empty disables it.
        panel: Intrusive panel measures computed against the clean reference
            (subset of estoi, sisdr, sbs). Empty disables them.
        wer: Whether to compute Whisper WER against the manifest transcript.
        asr_model: HF id of the Whisper model.
        device: Device.
    """

    def __init__(self, external: list[str] | tuple[str, ...] = ("dnsmos", "nisqa", "utmos", "spk_sim", "pesq"),
                 panel: list[str] | tuple[str, ...] = ("estoi", "sisdr"), wer: bool = True,
                 asr_model: str = "openai/whisper-small", device: str = "cuda"):
        self.external_measures, self.panel_measures, self.use_wer = tuple(external), tuple(panel), bool(wer)
        self.ext = self.panel = None
        if self.external_measures:
            from hackpatch.metrics.external import ExternalMetrics

            self.ext = ExternalMetrics(device=device, measures=self.external_measures)
        if self.panel_measures or self.use_wer:
            from hackpatch.metrics.panel import Panel

            # WER against the transcript goes through ``wer_against``; listing "wer" here
            # would also transcribe the clean reference on every call.
            self.panel = Panel(device=device, asr_model=asr_model, measures=self.panel_measures)

    def __call__(self, deg: list[np.ndarray], clean: list[np.ndarray], texts: list[str]) -> pd.DataFrame:
        """Computes every enabled measure.

        Args:
            deg: Enhanced (or noisy) waveforms.
            clean: Clean references aligned with ``deg``.
            texts: Reference transcripts aligned with ``deg``.

        Returns:
            One row per utterance; columns depend on the enabled measures
            (``dnsmos_sig, dnsmos_bak, dnsmos_ovrl, nisqa_mos, utmos, spk_sim,
            pesq, estoi, sisdr, sbs, wer``).
        """
        cols: list[pd.DataFrame] = []
        if self.ext is not None:
            needs_ref = any(m in self.external_measures for m in ("spk_sim", "pesq"))
            cols.append(self.ext(deg, clean if needs_ref else None).reset_index(drop=True))
        if self.panel is not None and self.panel_measures:
            df = self.panel(clean, deg).reset_index(drop=True)
            cols.append(df[[c for c in df.columns if c in self.panel_measures]])
        if self.panel is not None and self.use_wer:
            cols.append(pd.DataFrame({"wer": np.asarray(self.panel.wer_against(list(texts), deg), dtype=float)}))
        return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=range(len(deg)))


def bootstrap_summary(df: pd.DataFrame, group: str, metrics: list[str], n_boot: int = 1000,
                      seed: int = 0) -> pd.DataFrame:
    """Mean and percentile-bootstrap 95% CI (over utterances) per group and metric.

    Args:
        df: Per-utterance frame.
        group: Column identifying the system (e.g. ``checkpoint``).
        metrics: Metric columns to summarize.
        n_boot: Bootstrap resamples.
        seed: Bootstrap RNG seed.

    Returns:
        Long frame ``group, metric, mean, ci_lo, ci_hi, n``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in df.groupby(group, sort=False):
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            rows.append({group: name, "metric": m, "mean": float(v.mean()), "ci_lo": float(lo),
                         "ci_hi": float(hi), "n": int(len(v))})
    return pd.DataFrame(rows)
