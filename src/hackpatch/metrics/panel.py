"""Panel of independent, reference-based quality measures for the label-free hack detector.

The panel rule flags an attacked output as a hack when at least one of these
measures gets worse relative to the unattacked input:

* ``estoi``  - extended STOI (Jensen & Taal, 2016), ``pystoi`` at 16 kHz.
* ``sisdr``  - scale-invariant SDR in dB (Le Roux et al., 2019).
* ``wer``    - word error rate of Whisper(deg) against Whisper(ref).
* ``sbs``    - SpeechBERTScore F1 (Saeki et al., Interspeech 2024).

Typical usage::

    panel = Panel(device="cuda")
    df = panel(ref_wavs, deg_wavs)   # columns: estoi, sisdr, wer, sbs
"""
from __future__ import annotations

import contextlib
from collections.abc import Sequence

import jiwer
import numpy as np
import pandas as pd
import torch

from hackpatch.metrics._utils import (
    SR,
    equal_length_batches,
    parallel_map,
    resolve_device,
    to_numpy_list,
    trim_pairs,
    wav_hash,
)

ALL_MEASURES = ("estoi", "sisdr", "wer", "sbs")
#: Whisper consumes at most 30 s of audio per clip.
ASR_MAX_SECONDS = 30.0


def si_sdr(ref: np.ndarray, deg: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-distortion ratio.

    Both signals are made zero-mean; the reference is optimally scaled to the
    degraded signal before computing the energy ratio.

    Args:
        ref: Reference waveform.
        deg: Degraded waveform, same length as ``ref``.
        eps: Stabiliser added to the energies.

    Returns:
        SI-SDR in dB (NaN for empty inputs).
    """
    if len(ref) == 0:
        return float("nan")
    r = ref.astype(np.float64) - ref.mean(dtype=np.float64)
    d = deg.astype(np.float64) - deg.mean(dtype=np.float64)
    alpha = np.dot(d, r) / (np.dot(r, r) + eps)
    target = alpha * r
    noise = d - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def estoi(ref: np.ndarray, deg: np.ndarray) -> float:
    """Extended STOI via ``pystoi`` at 16 kHz.

    Returns NaN instead of pystoi's ``1e-5`` sentinel when fewer than 30
    non-silent STFT frames (~0.4 s of speech) remain (clip too short or silent).
    """
    import warnings

    from pystoi import stoi

    if len(ref) < 512:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = float(stoi(ref, deg, SR, extended=True))
    return float("nan") if val == 1e-5 else val


@contextlib.contextmanager
def _quiet_transformers():
    """Temporarily silences per-call ``generate`` notices (max_length/attention-mask/logits-processor)."""
    from transformers.utils import logging as hf_logging

    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(level)


def word_error_rate(ref_text: str, hyp_text: str) -> float:
    """Utterance WER on already-normalised text.

    Args:
        ref_text: Reference transcript.
        hyp_text: Hypothesis transcript.

    Returns:
        (S + D + I) / N; if the reference is empty, 0 when the hypothesis is
        also empty and 1 otherwise.
    """
    if not ref_text.split():
        return 0.0 if not hyp_text.split() else 1.0
    return float(jiwer.wer(ref_text, hyp_text))


class Panel:
    """Independent quality measures for the label-free hack detector.

    Models load lazily, only for the requested measures.

    Args:
        device: Torch device for Whisper and the SSL encoder.
        asr_model: Hugging Face Whisper checkpoint (English transcription, fp16 on GPU).
        sbs_model: Hugging Face SSL encoder for SpeechBERTScore.
        sbs_layer: Transformer layer whose output is compared (``hidden_states[layer]``).
            8 is the best-performing HuBERT-base layer in Saeki et al. (2024), Fig. 2.
        measures: Subset of ``("estoi", "sisdr", "wer", "sbs")`` to compute.
        asr_batch_size: Clips per Whisper ``generate`` call.
        sbs_batch_seconds: Audio seconds per SSL forward pass.
        n_jobs: CPU workers for ESTOI (default: available CPUs, max 16).
    """

    def __init__(self, device: str = "cuda", asr_model: str = "openai/whisper-small",
                 sbs_model: str = "facebook/hubert-base-ls960", sbs_layer: int = 8,
                 measures: tuple[str, ...] = ALL_MEASURES, asr_batch_size: int = 64,
                 sbs_batch_seconds: float = 320.0, n_jobs: int | None = None):
        unknown = set(measures) - set(ALL_MEASURES)
        if unknown:
            raise ValueError(f"unknown measures {sorted(unknown)}; choose from {ALL_MEASURES}")
        self.device = resolve_device(device)
        self.asr_model_name = asr_model
        self.sbs_model_name = sbs_model
        self.sbs_layer = sbs_layer
        self.measures = tuple(m for m in ALL_MEASURES if m in measures)
        self.asr_batch_size = asr_batch_size
        self.sbs_batch_samples = int(sbs_batch_seconds * SR)
        self.n_jobs = n_jobs
        self._asr = None
        self._asr_processor = None
        self._asr_cache: dict[bytes, str] = {}
        self._sbs = None

    # ------------------------------------------------------------------ public API
    def __call__(self, ref: Sequence, deg: Sequence) -> pd.DataFrame:
        """Scores time-aligned (reference, degraded) pairs.

        Args:
            ref: Reference waveforms (1-D float arrays/tensors, 16 kHz mono).
            deg: Degraded waveforms; ``deg[i]`` is aligned with ``ref[i]``.
                Each pair is trimmed to the shorter length.

        Returns:
            One row per pair with the requested columns among ``estoi``,
            ``sisdr`` (dB), ``wer`` (Whisper(deg) vs Whisper(ref)) and ``sbs``
            (SpeechBERTScore F1).
        """
        ref_np, deg_np = trim_pairs(to_numpy_list(ref), to_numpy_list(deg))
        out: dict[str, np.ndarray] = {}
        if "estoi" in self.measures:
            out["estoi"] = np.asarray(parallel_map(estoi, list(zip(ref_np, deg_np)), self.n_jobs), dtype=np.float64)
        if "sisdr" in self.measures:
            out["sisdr"] = np.array([si_sdr(r, d) for r, d in zip(ref_np, deg_np)], dtype=np.float64)
        if "wer" in self.measures:
            texts = self._transcribe_np(ref_np + deg_np)
            n = len(ref_np)
            out["wer"] = np.array([word_error_rate(r, h) for r, h in zip(texts[:n], texts[n:])], dtype=np.float64)
        if "sbs" in self.measures:
            out["sbs"] = self._speechbertscore(ref_np, deg_np)
        return pd.DataFrame(out, columns=list(self.measures))

    def transcribe(self, wavs: Sequence) -> list[str]:
        """Transcribes English speech with Whisper and applies Whisper's English text normaliser.

        Args:
            wavs: 1-D waveforms at 16 kHz; audio beyond 30 s is ignored.

        Returns:
            Normalised transcripts (lower-case, no punctuation), in input order.
        """
        return self._transcribe_np(to_numpy_list(wavs))

    def wer_against(self, texts: Sequence[str], deg: Sequence) -> np.ndarray:
        """WER of Whisper(deg) against given reference transcripts.

        Args:
            texts: Raw reference transcripts (normalised here with the same normaliser).
            deg: Waveforms to transcribe.

        Returns:
            [N] float64 WERs.
        """
        if len(texts) != len(deg):
            raise ValueError(f"texts and deg must have the same length, got {len(texts)} vs {len(deg)}")
        hyps = self.transcribe(deg)
        self._load_asr()
        refs = [self.normalize(t) for t in texts]
        return np.array([word_error_rate(r, h) for r, h in zip(refs, hyps)], dtype=np.float64)

    def normalize(self, text: str) -> str:
        """Applies Whisper's ``EnglishTextNormalizer`` (the standard Whisper WER normaliser)."""
        self._load_asr()
        return " ".join(self._asr_processor.tokenizer.normalize(text).split())

    # ------------------------------------------------------------------ ASR
    def _load_asr(self) -> None:
        if self._asr is not None:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._asr_processor = WhisperProcessor.from_pretrained(self.asr_model_name)
        self._asr = WhisperForConditionalGeneration.from_pretrained(self.asr_model_name, dtype=dtype)
        self._asr.to(self.device).eval()

    @torch.no_grad()
    def _transcribe_np(self, wavs: list[np.ndarray]) -> list[str]:
        """Batched, de-duplicated, cached Whisper transcription of numpy waveforms."""
        self._load_asr()
        keys = [wav_hash(w) for w in wavs]
        todo: dict[bytes, np.ndarray] = {}
        for k, w in zip(keys, wavs):
            if k not in self._asr_cache and k not in todo:
                todo[k] = w
        if todo:
            if len(self._asr_cache) > 200_000:
                self._asr_cache.clear()
            items = sorted(todo.items(), key=lambda kv: -len(kv[1]))
            fe = self._asr_processor.feature_extractor
            dtype = next(self._asr.parameters()).dtype
            max_len = int(ASR_MAX_SECONDS * SR)
            for s in range(0, len(items), self.asr_batch_size):
                chunk = items[s : s + self.asr_batch_size]
                audio = [w[:max_len] if len(w) else np.zeros(SR // 10, np.float32) for _, w in chunk]
                feats = fe(audio, sampling_rate=SR, return_tensors="pt", device=str(self.device)).input_features
                # Bound decoding so a hallucination loop cannot stall a batch:
                # ~8 tokens per second of audio is far above normal speech rates.
                longest = max(len(a) for a in audio) / SR
                max_new = int(min(440, 8 * longest + 24))
                with _quiet_transformers():
                    ids = self._asr.generate(feats.to(self.device, dtype), language="en", task="transcribe",
                                             max_new_tokens=max_new)
                    texts = self._asr_processor.batch_decode(ids, skip_special_tokens=True)
                for (k, _), t in zip(chunk, texts):
                    self._asr_cache[k] = " ".join(self._asr_processor.tokenizer.normalize(t).split())
        return [self._asr_cache[k] for k in keys]

    # ------------------------------------------------------------------ SpeechBERTScore
    def _load_sbs(self) -> None:
        if self._sbs is not None:
            return
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.sbs_model_name)
        n_layers = model.config.num_hidden_layers
        if not 0 <= self.sbs_layer <= n_layers:
            raise ValueError(f"sbs_layer={self.sbs_layer} outside [0, {n_layers}] for {self.sbs_model_name}")
        # Layers above the requested one are never used. With post-norm encoders
        # (do_stable_layer_norm=False) dropping them leaves hidden_states[layer] unchanged.
        if not getattr(model.config, "do_stable_layer_norm", False):
            model.encoder.layers = model.encoder.layers[: max(self.sbs_layer, 1)]
        self._sbs = model.to(self.device).eval()

    @torch.no_grad()
    def _speechbertscore(self, ref: list[np.ndarray], deg: list[np.ndarray]) -> np.ndarray:
        """SpeechBERTScore F1 per pair, following DiscreteSpeechMetrics' ``bert_score``.

        Pairs are processed in unpadded equal-length batches because the
        GroupNorm front-end of base SSL models is not padding-invariant.
        """
        self._load_sbs()
        f1 = np.full(len(ref), np.nan, dtype=np.float64)
        lengths = [len(r) for r in ref]
        valid = [i for i, n in enumerate(lengths) if n >= 400]  # conv front-end receptive field
        for batch in equal_length_batches([lengths[i] for i in valid], self.sbs_batch_samples // 2):
            idx = [valid[j] for j in batch]
            x = torch.from_numpy(np.stack([ref[i] for i in idx] + [deg[i] for i in idx])).to(self.device)
            hs = self._sbs(x, output_hidden_states=True).hidden_states[self.sbs_layer].float()
            hs = torch.nn.functional.normalize(hs, dim=-1)
            v_ref, v_gen = hs[: len(idx)], hs[len(idx) :]
            sim = torch.bmm(v_gen, v_ref.transpose(1, 2))  # [B, T_gen, T_ref]
            precision = sim.max(dim=2).values.mean(dim=1)
            recall = sim.max(dim=1).values.mean(dim=1)
            f1[idx] = (2 * precision * recall / (precision + recall)).double().cpu().numpy()
        return f1
