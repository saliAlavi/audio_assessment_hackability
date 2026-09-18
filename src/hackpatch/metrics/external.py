"""External quality measures for E1/E3 reporting (never used as rewards or detectors).

Columns (only for requested measures):

* ``dnsmos_sig``, ``dnsmos_bak``, ``dnsmos_ovrl`` - DNSMOS P.835 (Reddy et al., 2022),
  ``sig_bak_ovr.onnx`` from microsoft/DNS-Challenge with the official
  non-personalised polynomial calibration; ``dnsmos_p808`` - DNSMOS P.808
  (``model_v8.onnx``). Same segmentation as ``DNSMOS/dnsmos_local.py``.
* ``nisqa_mos`` - NISQA v2.0 overall MOS (official ``nisqa.tar``).
* ``utmos`` - UTMOS22 strong learner (tarepan/SpeechMOS v1.2.0 port).
* ``spk_sim`` - cosine similarity of ``microsoft/wavlm-base-plus-sv`` x-vectors
  between deg and ref (needs ref).
* ``pesq`` - wide-band PESQ (ITU-T P.862.2) at 16 kHz via ``pesq`` (needs ref).

Typical usage::

    ext = ExternalMetrics(device="cuda")
    df = ext(enhanced_wavs, ref=clean_wavs)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from hackpatch.metrics import weights
from hackpatch.metrics._utils import (
    SR,
    equal_length_batches,
    librosa_mel_filterbank,
    parallel_map,
    resolve_device,
    to_numpy_list,
    wav_hash,
)

ALL_MEASURES = ("dnsmos", "nisqa", "utmos", "spk_sim", "pesq")
COLUMNS = {
    "dnsmos": ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"),
    "nisqa": ("nisqa_mos",),
    "utmos": ("utmos",),
    "spk_sim": ("spk_sim",),
    "pesq": ("pesq",),
}
SPK_MODEL = "microsoft/wavlm-base-plus-sv"

# DNSMOS constants from DNSMOS/dnsmos_local.py
_DNSMOS_INPUT_LENGTH = 9.01
_P835_POLY = {  # non-personalised calibration (np.poly1d coefficients, highest power first)
    "sig": np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
    "bak": np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
    "ovr": np.poly1d([-0.06766283, 1.11546468, 0.04602535]),
}


def pesq_wb(ref: np.ndarray | None, deg: np.ndarray) -> float:
    """Wide-band PESQ at 16 kHz; NaN without a reference or when no utterance is detected.

    Args:
        ref: Reference waveform (or None).
        deg: Degraded waveform; the pair is trimmed to the shorter length.

    Returns:
        PESQ MOS-LQO in [1.04, 4.64], or NaN.
    """
    from pesq import BufferTooShortError, NoUtterancesError, pesq

    if ref is None:
        return float("nan")
    import warnings

    n = min(len(ref), len(deg))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # pesq divides by max|x|, 0 for silence
            return float(pesq(SR, ref[:n].astype(np.float32), deg[:n].astype(np.float32), "wb"))
    except (NoUtterancesError, BufferTooShortError):
        return float("nan")


class DNSMOS:
    """Faithful batched port of ``dnsmos_local.py`` (P.835 + P.808).

    The official ONNX graphs are converted to PyTorch with ``onnx2torch`` so they
    run batched on GPU (the installed onnxruntime is CPU-only and needs ~0.2 s
    per 9-s segment); outputs match ONNX Runtime to <1e-6.

    Args:
        model_dir: Directory containing ``sig_bak_ovr.onnx`` and ``model_v8.onnx``.
        device: Device for the models and the P.808 mel-spectrogram front-end.
        batch_size: 9.01-s segments per forward pass.
    """

    def __init__(self, model_dir, device: torch.device, batch_size: int = 64):
        from onnx2torch import convert

        self.p835 = convert(str(model_dir / "sig_bak_ovr.onnx")).to(device).eval()
        self.p808 = convert(str(model_dir / "model_v8.onnx")).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        # P.808 front-end: librosa.feature.melspectrogram(n_fft=321, hop=160, n_mels=120, power=2)
        self.mel_fb = torch.from_numpy(librosa_mel_filterbank(SR, 321, 120).astype(np.float64)).to(device)
        self.window = torch.hann_window(321, periodic=True, dtype=torch.float64, device=device)

    @staticmethod
    def segments(audio: np.ndarray) -> list[np.ndarray]:
        """Replicates the official tiling and 1-s-hop segmentation (including its float rounding)."""
        len_samples = int(_DNSMOS_INPUT_LENGTH * SR)
        audio = audio.astype(np.float64)
        if len(audio) == 0:
            return []
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio) / SR) - _DNSMOS_INPUT_LENGTH) + 1
        segs = []
        for idx in range(num_hops):
            seg = audio[int(idx * SR) : int((idx + _DNSMOS_INPUT_LENGTH) * SR)]
            if len(seg) >= len_samples:
                segs.append(seg)
        return segs

    def _p808_features(self, segs: np.ndarray) -> np.ndarray:
        """(power_to_db(mel, ref=max) + 40) / 40 for [N, 144160] float64 segments -> [N, 900, 120]."""
        x = torch.from_numpy(segs[:, :-160]).to(self.device)
        # Matches librosa >= 0.10 as called by the official script (zero padding, rfftfreq mel bins).
        spec = torch.stft(x, n_fft=321, hop_length=160, win_length=321, window=self.window, center=True,
                          pad_mode="constant", return_complex=True).abs() ** 2
        mel = torch.matmul(self.mel_fb, spec)  # [N, 120, T]
        db = 10.0 * torch.log10(mel.clamp_min(1e-10))
        db = db - db.amax(dim=(1, 2), keepdim=True)  # ref=np.max
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80.0)  # top_db=80
        return ((db + 40.0) / 40.0).transpose(1, 2).float()

    @torch.no_grad()
    def __call__(self, wavs: list[np.ndarray]) -> np.ndarray:
        """Scores clips.

        Args:
            wavs: 16 kHz float32 waveforms.

        Returns:
            [N, 4] array of (SIG, BAK, OVRL, P808) clip means (NaN for empty clips).
        """
        owner, segs = [], []
        for i, w in enumerate(wavs):
            s = self.segments(w)
            segs.extend(s)
            owner.extend([i] * len(s))
        out = np.full((len(wavs), 4), np.nan)
        if not segs:
            return out
        seg_arr = np.stack(segs)
        raw = np.empty((len(segs), 3), dtype=np.float64)
        p808 = np.empty(len(segs), dtype=np.float64)
        for s in range(0, len(segs), self.batch_size):
            chunk = seg_arr[s : s + self.batch_size]
            x = torch.from_numpy(chunk.astype(np.float32)).to(self.device)
            raw[s : s + len(chunk)] = self.p835(x).double().cpu().numpy()
            p808[s : s + len(chunk)] = self.p808(self._p808_features(chunk))[:, 0].double().cpu().numpy()
        cal = np.stack([_P835_POLY["sig"](raw[:, 0]), _P835_POLY["bak"](raw[:, 1]),
                        _P835_POLY["ovr"](raw[:, 2]), p808], axis=1)
        owner_arr = np.asarray(owner)
        for i in np.unique(owner_arr):
            out[i] = cal[owner_arr == i].mean(axis=0)
        return out


class ExternalMetrics:
    """Non-intrusive + intrusive external measures for E3/E1 reporting.

    Models load lazily, only for requested measures.

    Args:
        device: Torch device for NISQA, UTMOS, the speaker model and the DNSMOS front-end.
        measures: Subset of ``("dnsmos", "nisqa", "utmos", "spk_sim", "pesq")``.
        batch_seconds: Audio seconds per neural-model forward pass.
        n_jobs: CPU workers for PESQ (default: available CPUs, max 16).
    """

    def __init__(self, device: str = "cuda",
                 measures: tuple[str, ...] = ALL_MEASURES,
                 batch_seconds: float = 320.0, n_jobs: int | None = None):
        unknown = set(measures) - set(ALL_MEASURES)
        if unknown:
            raise ValueError(f"unknown measures {sorted(unknown)}; choose from {ALL_MEASURES}")
        self.device = resolve_device(device)
        self.measures = tuple(m for m in ALL_MEASURES if m in measures)
        self.batch_samples = int(batch_seconds * SR)
        self.n_jobs = n_jobs
        self._dnsmos: DNSMOS | None = None
        self._nisqa = None
        self._utmos = None
        self._spk = None

    def __call__(self, deg: Sequence, ref: Sequence | None = None) -> pd.DataFrame:
        """Scores degraded/enhanced clips.

        Args:
            deg: 1-D waveforms (numpy arrays or tensors), 16 kHz mono.
            ref: Optional clean references aligned with ``deg`` (whole list or
                individual entries may be None). Needed for ``spk_sim`` and ``pesq``.

        Returns:
            One row per clip with the requested columns among ``dnsmos_sig``,
            ``dnsmos_bak``, ``dnsmos_ovrl``, ``dnsmos_p808``, ``nisqa_mos``,
            ``utmos``, ``spk_sim`` and ``pesq``; NaN where a reference is missing
            or a clip is too short for a model.
        """
        deg_np = to_numpy_list(deg)
        if ref is None:
            ref_np: list[np.ndarray | None] = [None] * len(deg_np)
        else:
            if len(ref) != len(deg_np):
                raise ValueError(f"ref and deg must have the same length, got {len(ref)} vs {len(deg_np)}")
            ref_np = [None if r is None else to_numpy_list([r])[0] for r in ref]
        out: dict[str, np.ndarray] = {}
        if "dnsmos" in self.measures:
            if self._dnsmos is None:
                self._dnsmos = DNSMOS(weights.dnsmos_dir(), self.device)
            scores = self._dnsmos(deg_np)
            for j, col in enumerate(COLUMNS["dnsmos"]):
                out[col] = scores[:, j]
        if "nisqa" in self.measures:
            out["nisqa_mos"] = self._nisqa_scores(deg_np)
        if "utmos" in self.measures:
            out["utmos"] = self._utmos_scores(deg_np)
        if "spk_sim" in self.measures:
            out["spk_sim"] = self._spk_sim(deg_np, ref_np)
        if "pesq" in self.measures:
            out["pesq"] = np.asarray(parallel_map(pesq_wb, list(zip(ref_np, deg_np)), self.n_jobs), dtype=np.float64)
        columns = [c for m in self.measures for c in COLUMNS[m]]
        return pd.DataFrame(out, columns=columns)

    # ------------------------------------------------------------------ NISQA
    @torch.no_grad()
    def _nisqa_scores(self, wavs: list[np.ndarray]) -> np.ndarray:
        if self._nisqa is None:
            from hackpatch.metrics._nisqa import NISQAPredictor

            self._nisqa = NISQAPredictor(weights.nisqa_checkpoint(), sr=SR).to(self.device).eval()
        res = np.full(len(wavs), np.nan)
        valid = [i for i, w in enumerate(wavs) if len(w) >= self._nisqa.min_samples()]
        valid.sort(key=lambda i: len(wavs[i]))
        batch: list[int] = []
        total = 0
        for i in valid + [None]:  # sentinel flushes the last batch
            if i is not None and (not batch or total + len(wavs[i]) <= self.batch_samples):
                batch.append(i)
                total += len(wavs[i])
                continue
            if batch:
                x = [torch.from_numpy(wavs[j]).to(self.device) for j in batch]
                res[batch] = self._nisqa(x)[:, 0].double().cpu().numpy()
            batch, total = ([i], len(wavs[i])) if i is not None else ([], 0)
        return res

    # ------------------------------------------------------------------ UTMOS
    def _load_utmos(self) -> None:
        repo = weights.speechmos_repo_dir()
        if (repo / "hubconf.py").exists():
            model = torch.hub.load(str(repo), "utmos22_strong", source="local", pretrained=False)
        else:  # pragma: no cover - network fallback, pinned tag
            model = torch.hub.load(f"tarepan/SpeechMOS:{weights.SPEECHMOS_TAG}", "utmos22_strong",
                                   trust_repo=True, pretrained=False)
        state = torch.load(str(weights.utmos_checkpoint()), map_location="cpu")
        model.load_state_dict(state)
        self._utmos = model.to(self.device).eval()

    @torch.no_grad()
    def _utmos_scores(self, wavs: list[np.ndarray]) -> np.ndarray:
        """UTMOS22-strong per clip; unpadded equal-length batches (the model has no padding mask)."""
        if self._utmos is None:
            self._load_utmos()
        res = np.full(len(wavs), np.nan)
        valid = [i for i, w in enumerate(wavs) if len(w) >= 400]
        for b in equal_length_batches([len(wavs[i]) for i in valid], self.batch_samples):
            idx = [valid[j] for j in b]
            x = torch.from_numpy(np.stack([wavs[i] for i in idx])).to(self.device)
            res[idx] = self._utmos(x, SR).double().cpu().numpy()
        return res

    # ------------------------------------------------------------------ speaker similarity
    @torch.no_grad()
    def _spk_sim(self, deg: list[np.ndarray], ref: list[np.ndarray | None]) -> np.ndarray:
        """Cosine similarity of WavLM-base-plus-SV x-vectors (unpadded equal-length batches)."""
        res = np.full(len(deg), np.nan)
        pairs = [i for i, r in enumerate(ref) if r is not None and len(r) >= 400 and len(deg[i]) >= 400]
        if not pairs:
            return res
        if self._spk is None:
            from transformers import WavLMForXVector

            self._spk = WavLMForXVector.from_pretrained(SPK_MODEL).to(self.device).eval()
        uniq: dict[bytes, np.ndarray] = {}
        for i in pairs:
            for w in (deg[i], ref[i]):
                uniq.setdefault(wav_hash(w), w)
        keys = list(uniq)
        emb: dict[bytes, torch.Tensor] = {}
        for b in equal_length_batches([len(uniq[k]) for k in keys], self.batch_samples):
            x = torch.from_numpy(np.stack([uniq[keys[j]] for j in b])).to(self.device)
            e = torch.nn.functional.normalize(self._spk(input_values=x).embeddings.float(), dim=-1)
            for j, v in zip(b, e):
                emb[keys[j]] = v
        for i in pairs:
            res[i] = float((emb[wav_hash(deg[i])] * emb[wav_hash(ref[i])]).sum())
        return res
