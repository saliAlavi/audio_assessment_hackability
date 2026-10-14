"""Listening examples for the project page (not a paper result).

The paper's released attackers are applied, unchanged, to audio that may be redistributed: the
VoiceBank-DEMAND clean test set (CC BY 4.0). The paper's own attack set, BVCC test, is under the Blizzard
Challenge terms and may not be redistributed. Every clip is quantized to 16-bit PCM before it is scored, so
each score is the score of the exact file the page plays. Each clip is scored by all four predictors and by
the detector panel, and each attack is judged with the paper's corrected hack rule:

* ceiling: ``R(y) - R(ref) > tau``, where ``ref`` is the zero-perturbation round trip through the attack's own
  codec (the input itself for waveform PGD), and
* panel: at least one panel measure (against the clean input ``x``) is worse for ``y`` than for ``ref`` by
  more than its margin.

Outputs under ``results/``:

* ``scores.csv``    one row per (utterance, clip): every predictor's score and the panel measures.
* ``verdicts.csv``  one row per (utterance, attack): target scores, gains under both references, rules.
* ``summary.csv``   per attack over the whole set: hack, ceiling and panel rates and mean gains.
* ``audio_examples/`` the selected utterances (``{utt}/{clip}.wav`` and ``{utt}/{clip}.webp`` spectrograms)
  and ``examples.json``. ``python website/build.py --audio-from <that dir>`` puts them on the page.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from hackpatch.attack.grpo_attacker import Attacker
from hackpatch.data.datasets import ManifestDataset, collate
from hackpatch.detect.detector import MEASURE_SIGN
from hackpatch.utils.audio import SR, load_wav

#: Clips every utterance has besides its attacks: the input and the two zero-perturbation round trips.
BASE_CLIPS = ("clean", "rt_encodec", "rt_dac")
#: Round-trip clip of each attack codec.
ROUND_TRIP = {"encodec_24khz": "rt_encodec", "dac_16khz": "rt_dac"}


class _NoScore:
    """Stands in for the target predictor while attacks are applied (scores are taken later, on 16-bit audio)."""

    def eval(self) -> "_NoScore":
        return self

    def __call__(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(wav.shape[0], device=wav.device)


def to_pcm16(wav: np.ndarray | torch.Tensor) -> np.ndarray:
    """Quantizes a waveform in [-1, 1] to the int16 samples of a 16-bit PCM WAV file."""
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()
    return np.clip(np.round(np.asarray(wav, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def from_pcm16(pcm: np.ndarray) -> np.ndarray:
    """Decodes int16 samples the way audio readers do (``x / 32768``)."""
    return pcm.astype(np.float32) / 32768.0


def _asset(name: str) -> Path:
    from hackpatch import repro

    return repro.ARTIFACTS / "checkpoints" / name


# ------------------------------------------------------------------------------------------- attacks
def _apply_latent(spec: DictConfig, df: pd.DataFrame, seed: int, batch_size: int,
                  max_seconds: float) -> tuple[dict, dict, DictConfig]:
    """Applies a stored latent attacker (or random directions with its codec and latent statistics).

    Returns:
        ({utt_id: input x}, {utt_id: (round trip, attacked)}, the attacker's own attack config).
    """
    sd = torch.load(_asset(spec.attacker), map_location="cuda", weights_only=False)
    attacker = Attacker(OmegaConf.create(sd["cfg"]))
    attacker.load_state_dict(sd)
    torch.manual_seed(seed * 7919 + zlib.crc32(spec.name.encode()) % 1000)
    attacker.apply(_NoScore(), df, None, n_samples=0, batch_size=batch_size, max_seconds=max_seconds,
                   random_policy=spec.type == "random")
    inputs = {uid: x.numpy() for (uid, _), (x, _, _) in attacker.last_audio.items()}
    out = {uid: (xh.numpy(), y.numpy()) for (uid, _), (_, xh, y) in attacker.last_audio.items()}
    acfg = attacker.cfg
    del attacker
    torch.cuda.empty_cache()
    return inputs, out, acfg


def _apply_pgd(spec: DictConfig, df: pd.DataFrame, predictor, seed: int, max_seconds: float) -> dict:
    """White-box L-inf waveform PGD against ``predictor`` (the reference is the input itself)."""
    from hackpatch.patch.trainer import pgd_attack

    torch.manual_seed(seed * 7919 + zlib.crc32(spec.name.encode()) % 1000)
    ds = ManifestDataset(df.reset_index(drop=True), max_seconds=max_seconds, cache_in_ram=False)
    out = {}
    for b in DataLoader(ds, batch_size=spec.pgd.batch_size, collate_fn=collate, num_workers=8):
        wav, lengths = b["wav"].cuda(), b["lengths"].cuda()
        adv = pgd_attack(predictor, wav, lengths, spec.pgd.eps, spec.pgd.steps, spec.pgd.step_size)
        for i, idx in enumerate(b["idx"].tolist()):
            out[ds.df.at[idx, "utt_id"]] = adv[i, : int(lengths[i])].cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_sslmos(model, wavs: list[np.ndarray]) -> np.ndarray:
    """Scores clips one at a time (no padding, so a clip's score does not depend on its batch)."""
    out = np.zeros(len(wavs))
    for i, w in enumerate(wavs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[i] = model(torch.from_numpy(w)[None].cuda()).float().item()
    return out


def verdicts(scores: pd.DataFrame, attacks: list[dict], tau: float, margins: dict[str, float]) -> pd.DataFrame:
    """Applies the corrected hack rule to every (utterance, attack).

    Args:
        scores: One row per (``utt_id``, ``clip``) with a column per predictor and per panel measure
            (measured against the clean input).
        attacks: ``{"name", "target", "reference"}`` per attack; ``reference`` is the clip the gain is
            measured from (a round trip, or ``clean`` for waveform attacks).
        tau: Ceiling-rule threshold in MOS.
        margins: Panel margin per measure.

    Returns:
        One row per (utterance, attack) with ``score_input``, ``score_ref``, ``score_attacked``,
        ``gain_ref`` (corrected), ``gain_input`` (naive), ``ceiling``, ``panel_{measure}``, ``panel``, ``hack``.
    """
    by = scores.set_index(["utt_id", "clip"])
    rows = []
    for a in attacks:
        t = a["target"]
        y = by.xs(a["name"], level="clip")
        ref = by.xs(a["reference"], level="clip").loc[y.index]
        x = by.xs("clean", level="clip").loc[y.index]
        d = pd.DataFrame({"utt_id": y.index, "attack": a["name"], "target": t,
                          "score_input": x[t].to_numpy(), "score_ref": ref[t].to_numpy(),
                          "score_attacked": y[t].to_numpy()})
        d["gain_ref"] = d.score_attacked - d.score_ref
        d["gain_input"] = d.score_attacked - d.score_input
        d["ceiling"] = d.gain_ref > tau
        fires = np.zeros(len(d), dtype=bool)
        for m, margin in margins.items():
            col = (MEASURE_SIGN[m] * (ref[m].to_numpy() - y[m].to_numpy())) > margin
            d[f"panel_{m}"] = col
            fires |= col
        d["panel"] = fires
        d["hack"] = d.ceiling & d.panel
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------- selection
def select_examples(v: pd.DataFrame, meta: pd.DataFrame, ecfg: DictConfig, seed: int) -> list[dict]:
    """Picks the utterances shown on the page, by a stated rule.

    Among utterances between ``min_seconds`` and ``max_seconds`` long: one per speaker drawn at random
    (``seed``), then the ``n_strongest`` others with the largest corrected gain among those ``strongest_attack``
    hacks.

    Returns:
        ``[{"utt_id", "pick"}]``, random picks first.
    """
    ok = meta[(meta.seconds >= ecfg.min_seconds) & (meta.seconds <= ecfg.max_seconds)]
    rng = np.random.default_rng(seed)
    picks = []
    for spk in ecfg.speakers:
        pool = ok[ok.speaker == spk]
        picks.append({"utt_id": pool.utt_id.iloc[rng.integers(len(pool))], "pick": "random"})
    taken = {p["utt_id"] for p in picks}
    hacks = v[(v.attack == ecfg.strongest_attack) & v.hack & v.utt_id.isin(ok.utt_id) & ~v.utt_id.isin(taken)]
    for uid in hacks.sort_values("gain_ref", ascending=False, kind="stable").utt_id.head(ecfg.n_strongest):
        picks.append({"utt_id": uid, "pick": "strongest"})
    return picks


def spectrogram(wav: np.ndarray, top_db: float, scfg: DictConfig) -> "Image.Image":
    """Linear-frequency log-magnitude spectrogram as an image (time left to right, 0 Hz at the bottom).

    Args:
        wav: 16 kHz waveform.
        top_db: dB value mapped to the brightest colour (the clean clip's maximum, shared by all clips of
            an utterance so their brightness is comparable).
        scfg: ``n_fft``, ``hop``, ``db_range``, ``cmap``, ``width``, ``height``.
    """
    import matplotlib
    from PIL import Image

    n_fft, hop = int(scfg.n_fft), int(scfg.hop)
    pad = np.pad(wav.astype(np.float64), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop] * np.hanning(n_fft)
    db = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)).T + 1e-10)
    level = np.clip((db - (top_db - scfg.db_range)) / scfg.db_range, 0, 1)[::-1]
    rgb = (matplotlib.colormaps[scfg.cmap](level)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((int(scfg.width), int(scfg.height)), Image.LANCZOS)


def top_db_of(wav: np.ndarray, scfg: DictConfig) -> float:
    """Maximum dB of :func:`spectrogram`'s magnitude, used as the shared top of an utterance's colour scale."""
    n_fft, hop = int(scfg.n_fft), int(scfg.hop)
    pad = np.pad(wav.astype(np.float64), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop] * np.hanning(n_fft)
    return float(20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)).max() + 1e-10))


# ------------------------------------------------------------------------------------------- task
def run_audio_examples(cfg: DictConfig, rp, run) -> None:
    """Applies every configured attack to ``cfg.data`` and writes scores, verdicts and the page examples."""
    import wandb

    from hackpatch import repro
    from hackpatch.data.manifests import manifest_path
    from hackpatch.metrics.external import ExternalMetrics
    from hackpatch.metrics.panel import Panel
    from hackpatch.models.metric_predictors import METRIC_COLUMNS, build_predictor

    repro.fetch(sorted(set(cfg.fetch)))
    man = pd.read_csv(manifest_path(*cfg.data.manifest.split("/")))
    if cfg.data.get("max_utts"):
        man = man.sample(min(cfg.data.max_utts, len(man)), random_state=cfg.seed).reset_index(drop=True)
    texts = dict(zip(man.utt_id, man.text.str.strip()))
    sslmos = build_predictor(str(_asset(cfg.predictors.sslmos)), cfg.model.pretrained)
    sslmos.eval().requires_grad_(False)

    # 0. Every attack starts from the 16-bit clean file the page plays.
    clean_dir = rp.artifacts / "clean_pcm16"
    clean_dir.mkdir(exist_ok=True)
    pcm: dict[str, dict[str, np.ndarray]] = {}
    for uid, path in zip(man.utt_id, man[cfg.data.audio_column]):
        pcm[uid] = {"clean": to_pcm16(load_wav(path))}
        sf.write(clean_dir / f"{uid}.wav", pcm[uid]["clean"], SR, subtype="PCM_16")
    df = pd.DataFrame({"utt_id": man.utt_id, "path": [str(clean_dir / f"{u}.wav") for u in man.utt_id],
                       "mos": np.nan, "system": "", "corpus": cfg.data.manifest, "split": ""})

    # 1. Apply every attack; keep each clip as the int16 samples of the file the page would play.
    attacks = []
    for spec in cfg.attacks:
        print(f"[examples] applying {spec.name}", flush=True)
        if spec.type in ("grpo", "random"):
            inputs, out, acfg = _apply_latent(spec, df, cfg.seed, cfg.apply.batch_size, cfg.apply.max_seconds)
            codec, size = acfg.codec.name, {"eps": float(acfg.eps)}
            rt = ROUND_TRIP[codec]
            for uid, (xh, y) in out.items():
                clips = pcm[uid]
                assert np.array_equal(to_pcm16(inputs[uid]), clips["clean"]), uid
                xh16 = to_pcm16(xh)
                if rt in clips and np.abs(clips[rt].astype(int) - xh16).max() > 2:
                    print(f"[examples] warning: {uid} {rt} differs between attacks", flush=True)
                clips.setdefault(rt, xh16)
                clips[spec.name] = to_pcm16(y)
            ref = rt
        elif spec.type == "pgd":
            for uid, y in _apply_pgd(spec, df, sslmos, cfg.seed, cfg.apply.max_seconds).items():
                pcm[uid][spec.name] = to_pcm16(y)
            codec, ref, size = "waveform", "clean", {"pgd": OmegaConf.to_container(spec.pgd)}
        else:
            raise ValueError(spec.type)
        attacks.append({"name": spec.name, "type": spec.type, "target": spec.target, "codec": codec,
                        "reference": ref, "checkpoint": spec.get("attacker"), **size})

    # 2. Score every clip with every predictor and the panel (against the clean input).
    keys = [(uid, c) for uid in df.utt_id for c in pcm[uid]]
    wavs = [from_pcm16(pcm[uid][c]) for uid, c in keys]
    clean = [from_pcm16(pcm[uid]["clean"]) for uid, _ in keys]
    print(f"[examples] scoring {len(keys)} clips", flush=True)
    scores = pd.DataFrame(keys, columns=["utt_id", "clip"])
    scores["sslmos"] = score_sslmos(sslmos, wavs)
    published = [n for n in cfg.predictors if n != "sslmos"]
    ext = ExternalMetrics(measures=tuple(METRIC_COLUMNS[n][0] for n in published))(wavs)
    for name in published:
        scores[name] = ext[METRIC_COLUMNS[name][1]].to_numpy()
    del ext
    panel = Panel(**cfg.panel)
    pan = pd.concat([panel(clean[i:i + cfg.panel_chunk], wavs[i:i + cfg.panel_chunk])
                     for i in range(0, len(keys), cfg.panel_chunk)], ignore_index=True)
    for m in pan.columns:
        scores[m] = pan[m].to_numpy()
    scores.to_csv(rp.results / "scores.csv", index=False)

    # 3. Verdicts per utterance and the rate of each outcome over the whole set.
    v = verdicts(scores, attacks, cfg.detect.tau, dict(cfg.detect.margins))
    v.to_csv(rp.results / "verdicts.csv", index=False)
    summary = v.groupby("attack", sort=False).agg(
        n=("hack", "size"), hack_rate=("hack", "mean"), ceiling_rate=("ceiling", "mean"),
        panel_rate=("panel", "mean"), mean_gain_ref=("gain_ref", "mean"), mean_gain_input=("gain_input", "mean"),
    ).reset_index()
    summary.to_csv(rp.results / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    run.log({"examples/summary": wandb.Table(dataframe=summary)})

    # 4. The page examples: audio, spectrograms and examples.json.
    meta = pd.DataFrame({"utt_id": df.utt_id, "speaker": df.utt_id.str.split("_").str[0],
                         "seconds": [len(pcm[u]["clean"]) / SR for u in df.utt_id]})
    picks = select_examples(v, meta, cfg.examples, cfg.seed)
    out_dir = rp.results / "audio_examples"
    by = scores.set_index(["utt_id", "clip"])
    measures = list(cfg.panel.measures)
    examples = []
    for p in picks:
        uid = p["utt_id"]
        top = top_db_of(from_pcm16(pcm[uid]["clean"]), cfg.examples.spectrogram)
        clips = {}
        for c, samples in pcm[uid].items():
            (out_dir / uid).mkdir(parents=True, exist_ok=True)
            sf.write(out_dir / uid / f"{c}.wav", samples, SR, subtype="PCM_16")
            spectrogram(from_pcm16(samples), top, cfg.examples.spectrogram).save(
                out_dir / uid / f"{c}.webp", quality=int(cfg.examples.spectrogram.quality))
            row = by.loc[(uid, c)]
            clips[c] = {"wav": f"audio/{uid}/{c}.wav", "img": f"audio/{uid}/{c}.webp",
                        "scores": {n: round(float(row[n]), 3) for n in cfg.predictors},
                        "panel": {m: round(float(row[m]), 4) for m in measures}}
        vv = v[v.utt_id == uid].set_index("attack")
        examples.append({
            "utt_id": uid, "speaker": uid.split("_")[0], "text": texts[uid], "pick": p["pick"],
            "seconds": round(len(pcm[uid]["clean"]) / SR, 2), "clips": clips,
            "verdicts": {a: {"gain_ref": round(float(r.gain_ref), 3), "gain_input": round(float(r.gain_input), 3),
                             "ceiling": bool(r.ceiling), "hack": bool(r.hack),
                             "panel": {m: bool(r[f"panel_{m}"]) for m in measures}}
                         for a, r in vv.iterrows()}})
    doc = {
        "generated_by": {"command": f"python main.py --config-name {cfg.config_name}", "run": rp.root.name},
        "corpus": OmegaConf.to_container(cfg.data.attribution),
        "n_utts": int(len(df)), "tau": float(cfg.detect.tau), "margins": dict(cfg.detect.margins),
        "predictors": list(cfg.predictors), "measures": measures,
        "selection": OmegaConf.to_container(cfg.examples, resolve=True),
        "attacks": [{**a, "summary": {k: (int(x) if k == "n" else round(float(x), 4))
                                      for k, x in summary.set_index("attack").loc[a["name"]].items()}}
                    for a in attacks],
        "examples": examples,
    }
    (out_dir / "examples.json").write_text(json.dumps(doc, indent=1))
    print(f"[examples] wrote {len(examples)} examples to {out_dir}", flush=True)
