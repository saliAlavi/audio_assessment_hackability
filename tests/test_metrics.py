"""Sanity tests for ``hackpatch.metrics.panel.Panel`` and ``hackpatch.metrics.external.ExternalMetrics``.

CPU-only unit tests always run. The data tests use 50 evenly spaced
VoiceBank-DEMAND 16 kHz test pairs and need CUDA plus the dataset; run them in
a GPU job with::

    python -m pytest tests/test_metrics.py -s -v

Set ``HACKPATCH_VBD16K`` to override the dataset location and
``HACKPATCH_THROUGHPUT=0`` to skip the 1000-clip timing test.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from hackpatch.metrics._utils import equal_length_batches, librosa_mel_filterbank, trim_pairs
from hackpatch.metrics.external import DNSMOS, pesq_wb
from hackpatch.metrics.panel import si_sdr, word_error_rate

VBD = Path(os.environ.get("HACKPATCH_VBD16K", "/fs/ess/PAS2301/Data/Speech/datasets_voicebank/noisy-vctk-16k"))
N_PAIRS = 50
#: Wide-band PESQ of the unprocessed noisy VB-DMD test set as reported in the SE literature.
PUBLISHED_NOISY_PESQ_WB = 1.97

needs_gpu_data = pytest.mark.skipif(
    not (torch.cuda.is_available() and (VBD / "clean_testset_wav_16k").is_dir()),
    reason="needs CUDA and the VoiceBank-DEMAND 16 kHz test set")


def _summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] n={len(df)}\n{df.describe().loc[['mean', 'std', 'min', 'max']].round(4).to_string()}")


# ----------------------------------------------------------------------------- CPU unit tests
def test_si_sdr_properties():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(16000).astype(np.float32)
    n = rng.standard_normal(16000).astype(np.float32)
    assert si_sdr(s, s) > 60
    assert si_sdr(s, 0.3 * s + 0.01) > 60  # scale- and offset-invariant
    n_scaled = n * np.sqrt((s ** 2).sum() / (n ** 2).sum()) * 10 ** (-10 / 20)  # 10 dB SNR, near-orthogonal
    assert abs(si_sdr(s, s + n_scaled) - 10.0) < 0.3
    assert np.isnan(si_sdr(np.zeros(0, np.float32), np.zeros(0, np.float32)))


def test_word_error_rate_edge_cases():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "hello") == 1.0
    assert word_error_rate("a b c d", "a b c d") == 0.0
    assert word_error_rate("a b c d", "a x c") == 0.5
    assert word_error_rate("a b", "") == 1.0


def test_mel_filterbank_matches_torchaudio_for_even_fft():
    import torchaudio.functional as AF

    ours = librosa_mel_filterbank(16000, 4096, 48, 0.0, 20000.0)
    ta = AF.melscale_fbanks(2049, 0.0, 20000.0, 48, 16000, norm="slaney", mel_scale="slaney").T.numpy()
    assert ours.shape == (48, 2049)
    assert np.abs(ours - ta).max() < 1e-6


def test_dnsmos_segmentation_matches_official_rules():
    assert len(DNSMOS.segments(np.zeros(48000, np.float32))) == 3  # tiled to 12 s -> 3 hops
    assert len(DNSMOS.segments(np.zeros(160160, np.float32))) == 1  # 10.01 s -> 1 hop
    assert len(DNSMOS.segments(np.zeros(17 * 16000, np.float32))) == 7  # 8 hops, idx 7 lost to float rounding
    assert DNSMOS.segments(np.zeros(0, np.float32)) == []


def test_helpers():
    batches = list(equal_length_batches([5, 3, 5, 5, 3], max_batch_samples=10))
    assert sorted(sum(batches, [])) == [0, 1, 2, 3, 4]
    assert all(len({[5, 3, 5, 5, 3][i] for i in b}) == 1 and len(b) <= 3 for b in batches)
    r, d = trim_pairs([np.ones(5), np.ones(3)], [np.ones(4), np.ones(6)])
    assert [len(x) for x in r] == [4, 3] and [len(x) for x in d] == [4, 3]
    assert np.isnan(pesq_wb(None, np.zeros(16000, np.float32)))
    assert np.isnan(pesq_wb(np.zeros(16000, np.float32), np.zeros(16000, np.float32)))  # no utterance


# ----------------------------------------------------------------------------- data fixtures
def _names(n: int | None = N_PAIRS) -> list[str]:
    names = sorted(p.stem for p in (VBD / "clean_testset_wav_16k").glob("*.wav"))
    if n is None:
        return names
    idx = np.linspace(0, len(names) - 1, n).round().astype(int)
    return [names[i] for i in idx]


def _load(split: str, names: list[str]) -> list[np.ndarray]:
    return [sf.read(VBD / f"{split}_testset_wav_16k" / f"{n}.wav", dtype="float32")[0] for n in names]


@pytest.fixture(scope="module")
def vbd():
    names = _names()
    texts = [(VBD / "testset_txt" / f"{n}.txt").read_text().strip() for n in names]
    return {"names": names, "clean": _load("clean", names), "noisy": _load("noisy", names), "texts": texts}


@pytest.fixture(scope="module")
def panel():
    from hackpatch.metrics.panel import Panel

    return Panel(device="cuda")


@pytest.fixture(scope="module")
def external():
    from hackpatch.metrics.external import ExternalMetrics

    return ExternalMetrics(device="cuda")


# ----------------------------------------------------------------------------- Panel
@needs_gpu_data
def test_panel_clean_vs_noisy(vbd, panel):
    clean, noisy = vbd["clean"], vbd["noisy"]
    t0 = time.perf_counter()
    df = panel(clean + clean, clean + [torch.from_numpy(x) for x in noisy])  # tensors accepted too
    print(f"\npanel: {2 * N_PAIRS} pairs in {time.perf_counter() - t0:.1f}s (incl. model load)")
    assert list(df.columns) == ["estoi", "sisdr", "wer", "sbs"] and len(df) == 2 * N_PAIRS
    cc, nc = df.iloc[:N_PAIRS], df.iloc[N_PAIRS:]
    _summary("panel clean-vs-clean", cc)
    _summary("panel noisy-vs-clean", nc)
    assert cc["estoi"].min() > 0.999
    assert cc["sisdr"].min() > 60
    assert cc["wer"].max() == 0.0
    assert cc["sbs"].min() > 0.999
    assert nc["estoi"].mean() < cc["estoi"].mean() - 0.1
    assert nc["sisdr"].mean() < 15
    assert nc["sbs"].mean() < cc["sbs"].mean() - 0.05
    assert nc["wer"].mean() >= cc["wer"].mean()
    assert (nc["estoi"].to_numpy() < cc["estoi"].to_numpy()).all()
    assert (nc["sbs"].to_numpy() < cc["sbs"].to_numpy()).all()


@needs_gpu_data
def test_panel_wer_against_transcripts(vbd, panel):
    clean_wer = panel.wer_against(vbd["texts"], vbd["clean"])
    noisy_wer = panel.wer_against(vbd["texts"], vbd["noisy"])
    print(f"\nWhisper-small WER vs VCTK text: clean {clean_wer.mean():.4f}, noisy {noisy_wer.mean():.4f}")
    print("example:", repr(vbd["texts"][0]), "->", repr(panel.transcribe([vbd["clean"][0]])[0]))
    assert clean_wer.shape == (N_PAIRS,)
    assert clean_wer.mean() < 0.15
    assert noisy_wer.mean() >= clean_wer.mean()


@needs_gpu_data
def test_panel_batching_is_exact(vbd, panel):
    """Scores must not depend on batch composition, order or duplication."""
    ref, deg = vbd["clean"][:6], vbd["noisy"][:6]
    full = panel(ref, deg)
    single = pd.concat([panel([r], [d]) for r, d in zip(ref, deg)], ignore_index=True)
    rev = panel(ref[::-1], deg[::-1]).iloc[::-1].reset_index(drop=True)
    for other in (single, rev):
        np.testing.assert_allclose(full[["estoi", "sisdr", "wer"]], other[["estoi", "sisdr", "wer"]], atol=1e-9)
        np.testing.assert_allclose(full["sbs"], other["sbs"], atol=2e-4)


@needs_gpu_data
def test_speechbertscore_matches_reference_implementation(vbd, panel):
    """Compare against DiscreteSpeechMetrics' algorithm with the full (untruncated) HuBERT."""
    from transformers import HubertModel

    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda().eval()
    ours = panel(vbd["clean"][:4], vbd["noisy"][:4])["sbs"].to_numpy()
    ref_f1 = []
    with torch.no_grad():
        for r, d in zip(vbd["clean"][:4], vbd["noisy"][:4]):
            n = min(len(r), len(d))
            v_ref = model(torch.from_numpy(r[:n])[None].cuda(), output_hidden_states=True).hidden_states[8][0]
            v_gen = model(torch.from_numpy(d[:n])[None].cuda(), output_hidden_states=True).hidden_states[8][0]
            sim = (v_gen @ v_ref.T) / (v_gen.norm(dim=1, keepdim=True) * v_ref.norm(dim=1)[None])
            p, rc = sim.max(dim=1)[0].mean().item(), sim.max(dim=0)[0].mean().item()
            ref_f1.append(2 * p * rc / (p + rc))
    print("\nSBS ours", ours.round(5), "reference", np.round(ref_f1, 5))
    np.testing.assert_allclose(ours, ref_f1, atol=2e-4)


@needs_gpu_data
@pytest.mark.skipif(os.environ.get("HACKPATCH_THROUGHPUT", "1") == "0", reason="throughput test disabled")
def test_panel_throughput_1000_clips(vbd, panel):
    rng = np.random.default_rng(0)
    L = 3 * 16000
    ref, deg = [], []
    for i in range(1000):
        c = np.resize(vbd["clean"][i % N_PAIRS], L)
        n = np.resize(vbd["noisy"][i % N_PAIRS], L)
        # tiny per-copy dither so the ASR cache cannot shortcut repeated clips
        ref.append(c + 1e-4 * rng.standard_normal(L).astype(np.float32))
        deg.append(n + 1e-4 * rng.standard_normal(L).astype(np.float32))
    panel(ref[:8], deg[:8])  # warm-up (CPU workers, CUDA kernels)
    timings = {}
    for m in ("estoi", "sisdr", "wer", "sbs"):
        panel.measures = (m,)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        panel(ref, deg)
        torch.cuda.synchronize()
        timings[m] = time.perf_counter() - t0
    panel.measures = ("estoi", "sisdr", "wer", "sbs")
    total = sum(timings.values())
    print(f"\npanel throughput, 1000 x 3 s pairs: total {total:.1f}s; " +
          ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    assert total < 300


# ----------------------------------------------------------------------------- ExternalMetrics
@needs_gpu_data
def test_external_clean_vs_noisy(vbd, external):
    clean, noisy = vbd["clean"], vbd["noisy"]
    t0 = time.perf_counter()
    df = external(clean + noisy, ref=clean + clean)
    print(f"\nexternal: {2 * N_PAIRS} clips in {time.perf_counter() - t0:.1f}s (incl. model load)")
    expected = ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808", "nisqa_mos", "utmos", "spk_sim", "pesq"]
    assert list(df.columns) == expected and len(df) == 2 * N_PAIRS
    c, n = df.iloc[:N_PAIRS], df.iloc[N_PAIRS:]
    _summary("external clean (ref=clean)", c)
    _summary("external noisy (ref=clean)", n)
    for col in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808", "nisqa_mos", "utmos"):
        assert c[col].mean() > n[col].mean(), col
    assert c["spk_sim"].min() > 0.999
    assert n["spk_sim"].mean() > 0.7
    assert c["pesq"].min() > 4.5
    assert abs(n["pesq"].mean() - PUBLISHED_NOISY_PESQ_WB) < 0.35
    assert df.notna().all().all()

    t0 = time.perf_counter()
    df2 = external(noisy)
    print(f"external (no ref, warm): {N_PAIRS} clips in {time.perf_counter() - t0:.1f}s")
    assert df2[["spk_sim", "pesq"]].isna().all().all()
    # Different batch composition -> different cuDNN kernels (TF32 convs); differences are ~1e-4 MOS.
    for col in ("dnsmos_ovrl", "nisqa_mos", "utmos"):
        np.testing.assert_allclose(df2[col].to_numpy(), n[col].to_numpy(), atol=2e-3, err_msg=col)


@needs_gpu_data
@pytest.mark.skipif(os.environ.get("HACKPATCH_THROUGHPUT", "1") == "0", reason="throughput test disabled")
def test_external_throughput_1000_clips(vbd, external):
    L = 3 * 16000
    deg = [np.resize(vbd["noisy"][i % N_PAIRS], L) for i in range(1000)]
    ref = [np.resize(vbd["clean"][i % N_PAIRS], L) for i in range(1000)]
    external(deg[:8], ref=ref[:8])  # warm-up
    timings = {}
    for m in ("dnsmos", "nisqa", "utmos", "spk_sim", "pesq"):
        external.measures = (m,)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        external(deg, ref=ref)
        torch.cuda.synchronize()
        timings[m] = time.perf_counter() - t0
    external.measures = ("dnsmos", "nisqa", "utmos", "spk_sim", "pesq")
    print(f"\nexternal throughput, 1000 x 3 s clips: total {sum(timings.values()):.1f}s; " +
          ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))


@needs_gpu_data
def test_pesq_full_noisy_testset_matches_published():
    """Wide-band PESQ of all 824 noisy test files vs the published 1.97."""
    from hackpatch.metrics.external import ExternalMetrics

    names = _names(None)
    ext = ExternalMetrics(device="cuda", measures=("pesq",))
    t0 = time.perf_counter()
    df = ext(_load("noisy", names), ref=_load("clean", names))
    mean = df["pesq"].mean()
    print(f"\nPESQ-wb noisy VB-DMD test set (n={len(df)}): {mean:.4f} (published {PUBLISHED_NOISY_PESQ_WB}); "
          f"{time.perf_counter() - t0:.1f}s")
    assert abs(mean - PUBLISHED_NOISY_PESQ_WB) < 0.05


@needs_gpu_data
def test_external_short_and_missing_inputs(external):
    rng = np.random.default_rng(1)
    deg = [rng.standard_normal(200).astype(np.float32) * 0.1, rng.standard_normal(32000).astype(np.float32) * 0.1]
    df = external(deg, ref=[None, deg[1]])
    assert len(df) == 2
    assert df.loc[0, ["nisqa_mos", "utmos", "spk_sim", "pesq"]].isna().all()
    assert np.isfinite(df.loc[1, ["dnsmos_ovrl", "nisqa_mos", "utmos", "spk_sim"]].astype(float)).all()
