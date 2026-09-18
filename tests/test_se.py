"""Tests for the E3 enhancement model, its Gaussian policy and the supervised loss.

Run (GPU node preferred, CPU works)::

    source scripts/env.sh && python -m pytest tests/test_se.py -q

The overfit-one-batch check writes ``checks/se_overfit_one_batch.log``.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from hackpatch.se.model import BandResidualPolicy, MaskNet, erb_interpolation, load_se_model, num_parameters
from hackpatch.se.scoring import bootstrap_summary
from hackpatch.se.train import si_sdr, supervised_loss

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000


def _model(**kw) -> MaskNet:
    torch.manual_seed(0)
    return MaskNet(**kw).to(DEVICE)


def test_shapes_and_param_count():
    model = _model()
    wav = torch.randn(3, SR * 2 + 37, device=DEVICE) * 0.1
    spec = model.stft(wav)
    assert spec.shape == (3, 257, (SR * 2 + 37) // 128 + 1)
    mu = model.mean_logits(spec)
    assert mu.shape == spec.shape
    out = model(wav)
    assert out.shape == wav.shape
    assert 1e6 <= num_parameters(model) <= 3e6


def test_identity_at_zero_logits_and_mask_range():
    model = _model(mask_max=2.0)
    wav = torch.randn(2, SR, device=DEVICE) * 0.1
    spec = model.stft(wav)
    y = model.synthesize(spec, torch.zeros_like(spec.real), wav.shape[1])
    assert torch.allclose(y, wav, atol=1e-4)
    logits = torch.linspace(-50, 50, 1000, device=DEVICE)
    for mmax in (1.0, 2.0, 3.5):
        m = MaskNet(mask_max=mmax).to(DEVICE).mask_from_logits(logits)
        assert m.min() >= 0 and m.max() <= mmax
        assert math.isclose(m.max().item(), mmax, rel_tol=1e-4)


def test_log_prob_matches_torch_distributions():
    model = _model(init_log_std=-0.7)
    with torch.no_grad():
        model.log_std.copy_(torch.linspace(-2, 0.5, model.n_freq))
    mu = torch.randn(4, model.n_freq, 20, device=DEVICE)
    a = model.sample(mu)
    ref = torch.distributions.Normal(mu, model.std().expand_as(mu)).log_prob(a)
    assert torch.allclose(model.log_prob(a, mu), ref, atol=1e-5)
    # clamping of log σ is respected
    with torch.no_grad():
        model.log_std.fill_(10.0)
    assert torch.allclose(model.std(), torch.tensor(math.exp(model.max_log_std), device=DEVICE))
    # sample statistics
    big_mu = torch.zeros(1, model.n_freq, 4000, device=DEVICE)
    with torch.no_grad():
        model.log_std.fill_(-1.0)
    s = model.sample(big_mu)
    assert abs(s.std().item() - math.exp(-1.0)) < 0.01


def test_kl_matches_torch_distributions():
    mu_p, mu_q = torch.randn(2, 8, 5), torch.randn(2, 8, 5)
    sp, sq = torch.rand(1, 8, 1) + 0.1, torch.rand(1, 8, 1) + 0.1
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu_p, sp.expand_as(mu_p)),
                                            torch.distributions.Normal(mu_q, sq.expand_as(mu_q)))
    assert torch.allclose(MaskNet.gaussian_kl(mu_p, sp, mu_q, sq), ref, atol=1e-5)


def test_log_prob_gradient_direction():
    """Policy-gradient sanity: ∇_μ log π(a) points from μ towards a."""
    model = _model()
    mu = torch.zeros(1, model.n_freq, 3, device=DEVICE, requires_grad=True)
    a = torch.full_like(mu, 0.5)
    model.log_prob(a, mu).sum().backward()
    assert (mu.grad > 0).all()


def test_si_sdr_properties():
    x = torch.randn(2, 8000)
    assert (si_sdr(3.0 * x, x) > 60).all()
    n = torch.randn(2, 8000)
    val = si_sdr(x + n, x)
    assert (val.abs() < 3).all()


def test_save_load_roundtrip(tmp_path):
    model = _model(rnn_layers=1, hidden=128)
    model.save(tmp_path / "m.pt", {"step": 1})
    loaded = load_se_model(tmp_path / "m.pt", {"init_log_std": -2.0}, device=DEVICE)
    wav = torch.randn(1, SR, device=DEVICE) * 0.1
    with torch.no_grad():
        assert torch.allclose(model.eval()(wav), loaded(wav), atol=1e-5)
    assert loaded.config["rnn_layers"] == 1


def test_erb_interpolation():
    for k in (8, 32, 64):
        U = erb_interpolation(257, k, SR)
        assert U.shape == (257, k) and (U >= 0).all()
        assert torch.allclose(U.sum(1), torch.ones(257), atol=1e-6)
        assert (U.sum(0) > 0).all()  # every band reaches at least one bin


def test_band_residual_policy(tmp_path):
    base = _model(rnn_layers=1, hidden=128).eval()
    policy = BandResidualPolicy(base, n_bands=16, init_log_std=-0.5).to(DEVICE)
    wav = torch.randn(2, SR, device=DEVICE) * 0.1
    spec = policy.stft(wav)
    mu = policy.mean_logits(spec)
    assert mu.shape == (2, 16, spec.shape[-1]) and torch.count_nonzero(mu) == 0
    with torch.no_grad():  # zero-initialized head: identical to the pre-trained model
        assert torch.allclose(policy(wav), base(wav), atol=1e-5)
        a = policy.sample(mu)
        assert policy.mask(spec, a).max() <= base.mask_max
    ref = torch.distributions.Normal(mu, policy.std().expand_as(mu)).log_prob(a)
    assert torch.allclose(policy.log_prob(a, mu), ref, atol=1e-5)
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("actor.") or n == "log_std" for n in trainable)
    assert "actor.log_std" not in trainable
    with torch.no_grad():
        policy.actor.out.bias.fill_(0.3)
    policy.save(tmp_path / "p.pt")
    loaded = load_se_model(tmp_path / "p.pt", device=DEVICE)
    assert isinstance(loaded, BandResidualPolicy)
    with torch.no_grad():
        assert torch.allclose(policy.eval()(wav), loaded(wav), atol=1e-5)
        assert not torch.allclose(loaded(wav), base(wav), atol=1e-4)


def test_bootstrap_summary():
    import pandas as pd

    df = pd.DataFrame({"sys": ["a"] * 200 + ["b"] * 200,
                       "m": np.r_[np.random.default_rng(0).normal(1, 1, 200), np.full(200, 2.0)]})
    s = bootstrap_summary(df, "sys", ["m"], n_boot=200)
    a, b = s[s.sys == "a"].iloc[0], s[s.sys == "b"].iloc[0]
    assert a.ci_lo < a["mean"] < a.ci_hi and b.ci_lo == b.ci_hi == 2.0


def _one_batch(n: int = 4, seconds: float = 2.0) -> tuple[torch.Tensor, torch.Tensor, str]:
    """A real VBD batch if the manifest exists, else a synthetic harmonic-plus-noise batch."""
    try:
        from hackpatch.se.data import AudioStore, vbd_manifest

        store, crop = AudioStore(), int(seconds * SR)
        df = vbd_manifest("train").sort_values("utt_id").head(n)
        out = {"noisy": [], "clean": []}
        for _, row in df.iterrows():  # centre crops: VBD files start with ~0.5-1 s of silence
            for kind in out:
                w = np.asarray(store.get(row, kind), dtype=np.float32)
                s = max(0, (len(w) - crop) // 2)
                out[kind].append(torch.from_numpy(np.pad(w[s:s + crop], (0, max(0, crop - len(w))))))
        return torch.stack(out["noisy"]), torch.stack(out["clean"]), f"vbd_train (first {n} utts, centre crops)"
    except (FileNotFoundError, OSError):
        g = torch.Generator().manual_seed(0)
        t = torch.arange(int(seconds * SR)) / SR
        f0 = 120 + 80 * torch.rand(n, 1, generator=g)
        clean = sum(torch.sin(2 * math.pi * k * f0 * t) / k for k in range(1, 6)) * 0.1
        clean = clean * (0.5 + 0.5 * torch.sin(2 * math.pi * 3 * t))
        return clean + 0.05 * torch.randn(clean.shape, generator=g), clean, "synthetic"


def test_overfit_one_batch():
    noisy, clean, source = _one_batch()
    noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
    lengths = torch.full((noisy.shape[0],), noisy.shape[1], device=DEVICE)
    model = _model()
    params = [p for n, p in model.named_parameters() if n != "log_std"]
    opt = torch.optim.Adam(params, lr=2e-3)
    with torch.no_grad():
        sdr_in = si_sdr(noisy, clean).mean().item()
        sdr_0 = si_sdr(model(noisy), clean).mean().item()
    lines, t0 = [f"source={source} device={DEVICE} params={num_parameters(model)} input_sisdr={sdr_in:.2f}"], time.time()
    first = None
    for step in range(1, 401):
        loss, logs = supervised_loss(model, noisy, clean, lengths, w_sisdr=1.0, w_mag=10.0, compress=0.3)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        first = first if first is not None else logs
        if step % 50 == 0 or step == 1:
            lines.append(f"step {step} loss={logs['loss/total']:.3f} sisdr={logs['loss/sisdr']:.2f} "
                         f"mag_l1={logs['loss/mag_l1']:.4f}")
    with torch.no_grad():
        sdr_fit = si_sdr(model(noisy), clean).mean().item()
    lines.append(f"init_model_sisdr={sdr_0:.2f} fitted_sisdr={sdr_fit:.2f} gain_over_input={sdr_fit - sdr_in:.2f} dB "
                 f"time={time.time() - t0:.1f}s")
    ok = sdr_fit > sdr_in + 5.0 and logs["loss/mag_l1"] < 0.6 * first["loss/mag_l1"]
    lines.append("PASS" if ok else "FAIL")
    out = ROOT / "checks" / "se_overfit_one_batch.log"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    assert ok, "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
