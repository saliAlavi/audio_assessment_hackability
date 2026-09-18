"""Adapters that expose published quality metrics with the SSL-MOS call signature.

The GRPO attacker is black-box: it only needs ``predictor(wav, lengths) -> [B]``
scores. Wrapping UTMOS, DNSMOS and NISQA this way lets the same attacker, the
same detector and the same corrected-reference analysis be run against the
metrics the field actually uses as rewards.
"""
from __future__ import annotations

import numpy as np
import torch

#: measure name -> (ExternalMetrics measure, column, nominal scale)
METRIC_COLUMNS = {
    "utmos": ("utmos", "utmos"),
    "nisqa": ("nisqa", "nisqa_mos"),
    "dnsmos": ("dnsmos", "dnsmos_ovrl"),
}


class MetricPredictor:
    """A published non-intrusive metric used as a predictor (and as an attack target).

    Args:
        name: One of :data:`METRIC_COLUMNS`.
        device: Device for the underlying models.
        nan_fill: Score substituted when the metric declines a clip (too short);
            such clips are counted in :attr:`n_nan`.
    """

    def __init__(self, name: str, device: str = "cuda", nan_fill: float = 3.0):
        from hackpatch.metrics.external import ExternalMetrics

        if name not in METRIC_COLUMNS:
            raise ValueError(f"unknown metric {name}; choose from {sorted(METRIC_COLUMNS)}")
        measure, self.column = METRIC_COLUMNS[name]
        self.name = name
        self.device = device
        self.nan_fill = nan_fill
        self.n_nan = 0
        self._ext = ExternalMetrics(device=device, measures=(measure,))

    # -- the small surface the attacker and the loop use on a predictor ------
    def eval(self) -> "MetricPredictor":
        return self

    def train(self, mode: bool = True) -> "MetricPredictor":
        return self

    def requires_grad_(self, flag: bool = False) -> "MetricPredictor":
        return self

    @property
    def training(self) -> bool:
        return False

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Scores a padded batch.

        Args:
            wav: ``[B, T]`` 16 kHz waveforms.
            lengths: ``[B]`` valid sample counts, or None for unpadded input.

        Returns:
            ``[B]`` float32 scores on the metric's own scale.
        """
        x = wav.detach().float().cpu().numpy()
        clips = [x[i, : int(lengths[i])] if lengths is not None else x[i] for i in range(x.shape[0])]
        scores = self._ext(clips)[self.column].to_numpy(dtype=np.float64)
        self.n_nan += int(np.isnan(scores).sum())
        scores = np.nan_to_num(scores, nan=self.nan_fill)
        return torch.as_tensor(scores, device=wav.device, dtype=torch.float32)


def build_predictor(spec: str, pretrained: str = "facebook/wav2vec2-base", device: str = "cuda"):
    """Builds either a trained SSL-MOS checkpoint or a published metric.

    Args:
        spec: ``metric:<name>`` for a published metric, otherwise a checkpoint
            path (or pointer file) for SSL-MOS.
        pretrained: SSL encoder id for the SSL-MOS case.
        device: Device.

    Returns:
        An object with the ``predictor(wav, lengths) -> [B]`` contract.
    """
    if spec.startswith("metric:"):
        return MetricPredictor(spec.split(":", 1)[1], device=device)
    from hackpatch.models.ssl_mos import load_predictor

    return load_predictor(spec, pretrained, device)
