"""Small helpers shared by the metric modules (input coercion, batching, CPU pools)."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch

#: Sample rate every metric operates at.
SR = 16000


def to_numpy_list(wavs: Sequence[np.ndarray | torch.Tensor]) -> list[np.ndarray]:
    """Converts a list of 1-D waveforms to contiguous float32 numpy arrays.

    Args:
        wavs: 1-D numpy arrays or torch tensors (any device/dtype), 16 kHz mono.

    Returns:
        List of 1-D float32 arrays (squeezed; NaN/Inf replaced by 0).

    Raises:
        ValueError: If an element is not 1-D after squeezing.
    """
    out = []
    for w in wavs:
        if isinstance(w, torch.Tensor):
            w = w.detach().to("cpu", torch.float32).numpy()
        w = np.asarray(w, dtype=np.float32).squeeze()
        if w.ndim == 0:
            w = w.reshape(1)
        if w.ndim != 1:
            raise ValueError(f"expected 1-D waveform, got shape {w.shape}")
        if not np.isfinite(w).all():
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        out.append(np.ascontiguousarray(w))
    return out


def trim_pairs(ref: list[np.ndarray], deg: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Trims each (ref, deg) pair to the shorter of the two lengths.

    Args:
        ref: Reference waveforms.
        deg: Degraded waveforms, same count as ``ref``.

    Returns:
        The trimmed (ref, deg) lists.

    Raises:
        ValueError: If the lists differ in length.
    """
    if len(ref) != len(deg):
        raise ValueError(f"ref and deg must have the same length, got {len(ref)} vs {len(deg)}")
    n = [min(len(r), len(d)) for r, d in zip(ref, deg)]
    return [r[:k] for r, k in zip(ref, n)], [d[:k] for d, k in zip(deg, n)]


def wav_hash(w: np.ndarray) -> bytes:
    """Returns a 16-byte content hash of a float32 waveform."""
    return hashlib.blake2b(w.tobytes(), digest_size=16).digest()


def equal_length_batches(lengths: Sequence[int], max_batch_samples: int,
                         max_batch_size: int = 256) -> Iterator[list[int]]:
    """Groups indices into batches whose members all have identical length.

    SSL encoders with GroupNorm front-ends (HuBERT/WavLM base, wav2vec 2.0 base)
    and models without padding masks (UTMOS22) give different outputs when
    zero-padded, so exact results require unpadded, equal-length batches.

    Args:
        lengths: Sample count of each item.
        max_batch_samples: Upper bound on ``batch_size * length``.
        max_batch_size: Upper bound on the batch size.

    Yields:
        Lists of indices into ``lengths``.
    """
    groups: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        groups.setdefault(int(n), []).append(i)
    for n, idx in sorted(groups.items()):
        bs = max(1, min(max_batch_size, max_batch_samples // max(n, 1)))
        for s in range(0, len(idx), bs):
            yield idx[s : s + bs]


def n_cpus() -> int:
    """Returns the number of CPUs this process may use (Slurm-affinity aware)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def parallel_map(fn: Callable[..., Any], args: list[tuple], n_jobs: int | None = None,
                 min_items_parallel: int = 16) -> list[Any]:
    """Maps a picklable CPU-bound function over argument tuples with joblib/loky.

    Small inputs run serially to avoid worker start-up cost. Loky workers are
    separate interpreter processes, so this is safe after CUDA initialisation.

    Args:
        fn: Top-level (picklable) function.
        args: One argument tuple per call.
        n_jobs: Worker count; defaults to ``min(n_cpus(), 16)``.
        min_items_parallel: Below this many items run in-process.

    Returns:
        Results in input order.
    """
    n_jobs = n_jobs or min(n_cpus(), 16)
    if n_jobs <= 1 or len(args) < min_items_parallel:
        return [fn(*a) for a in args]
    from joblib import Parallel, delayed

    batch = max(1, len(args) // (4 * n_jobs))
    return Parallel(n_jobs=n_jobs, backend="loky", batch_size=batch)(delayed(fn)(*a) for a in args)


def _hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    mels = f / (200.0 / 3)
    logstep = np.log(6.4) / 27.0
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / logstep, mels)


def _mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    logstep = np.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(logstep * (m - 15.0)), (200.0 / 3) * m)


def librosa_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0,
                           fmax: float | None = None) -> np.ndarray:
    """Re-implements ``librosa.filters.mel(htk=False, norm="slaney")`` (librosa >= 0.10).

    Avoids a runtime librosa/numba dependency (numba is broken in this env) while
    reproducing the filterbanks the NISQA and DNSMOS feature pipelines were built on.

    Args:
        sr: Sample rate.
        n_fft: FFT size (odd sizes use ``rfftfreq`` bin centres, as librosa does).
        n_mels: Number of mel bands.
        fmin: Lowest band edge in Hz.
        fmax: Highest band edge in Hz; defaults to ``sr / 2``.

    Returns:
        [n_mels, 1 + n_fft // 2] float32 weights.
    """
    fmax = sr / 2.0 if fmax is None else fmax
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz_slaney(np.linspace(_hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, len(fftfreqs)), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


def resolve_device(device: str) -> torch.device:
    """Returns ``device``, falling back to CPU (with a warning) when CUDA is absent."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        import warnings

        warnings.warn(f"device={device!r} requested but CUDA is unavailable; using CPU", stacklevel=3)
        return torch.device("cpu")
    return torch.device(device)
