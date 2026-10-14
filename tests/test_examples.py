"""CPU tests for the project page's listening examples (``hackpatch.attack.examples``)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import OmegaConf

from hackpatch.attack.examples import from_pcm16, select_examples, to_pcm16, verdicts

MARGINS = {"estoi": 0.02, "sisdr": 1.0, "wer": 0.0, "sbs": 0.02}


def test_pcm16_matches_the_written_file():
    rng = np.random.default_rng(0)
    wav = np.clip(rng.standard_normal(16000) * 0.3, -1.2, 1.2).astype(np.float32)
    pcm = to_pcm16(wav)
    buf = io.BytesIO()
    sf.write(buf, pcm, 16000, subtype="PCM_16", format="WAV")
    buf.seek(0)
    read, _ = sf.read(buf, dtype="float32")
    assert np.array_equal(read, from_pcm16(pcm))          # what is scored is exactly what a reader decodes
    assert np.array_equal(to_pcm16(from_pcm16(pcm)), pcm)  # quantizing twice changes nothing
    assert np.abs(from_pcm16(pcm) - np.clip(wav, -1, 32767 / 32768)).max() <= 0.5 / 32768 + 1e-9


def _scores() -> pd.DataFrame:
    """Two utterances; ``att`` targets ``r`` and is referenced to round trip ``rt``."""
    good = {"estoi": 0.95, "sisdr": 10.0, "wer": 0.0, "sbs": 0.9}
    rows = [
        # u1: score +0.5 over the round trip, ESTOI drops by 0.05 -> hack
        ("u1", "clean", 3.0, good), ("u1", "rt", 2.8, good),
        ("u1", "att", 3.3, {**good, "estoi": 0.90}),
        # u2: +0.4 over the input but only +0.2 over the round trip; audio worse -> no hack (naive rule would fire)
        ("u2", "clean", 3.0, good), ("u2", "rt", 3.2, good),
        ("u2", "att", 3.4, {**good, "sisdr": 5.0}),
    ]
    return pd.DataFrame([{"utt_id": u, "clip": c, "r": s, **m} for u, c, s, m in rows])


def test_corrected_rule():
    v = verdicts(_scores(), [{"name": "att", "target": "r", "reference": "rt"}], 0.3, MARGINS).set_index("utt_id")
    assert v.loc["u1", "hack"] and v.loc["u1", "panel_estoi"] and not v.loc["u1", "panel_sisdr"]
    assert np.isclose(v.loc["u1", "gain_ref"], 0.5) and np.isclose(v.loc["u1", "gain_input"], 0.3)
    assert v.loc["u2", "panel"] and not v.loc["u2", "ceiling"] and not v.loc["u2", "hack"]
    assert v.loc["u2", "gain_input"] > 0.3


def test_panel_margin():
    s = _scores()
    s.loc[(s["utt_id"] == "u1") & (s["clip"] == "att"), "estoi"] = 0.935  # within the margin: not worse enough
    v = verdicts(s, [{"name": "att", "target": "r", "reference": "rt"}], 0.3, MARGINS).set_index("utt_id")
    assert not v.loc["u1", "panel"] and not v.loc["u1", "hack"]


def test_selection_rule():
    ids = [f"p232_{i:03d}" for i in range(6)] + [f"p257_{i:03d}" for i in range(6)]
    meta = pd.DataFrame({"utt_id": ids, "speaker": [u[:4] for u in ids], "seconds": [3.0] * 12})
    meta.loc[meta.utt_id == "p232_005", "seconds"] = 9.0  # too long: never shown
    gains = {"p232_001": 0.4, "p232_005": 2.0, "p257_002": 0.6, "p257_003": 0.9}
    v = pd.DataFrame({"utt_id": ids, "attack": "a", "gain_ref": [gains.get(u, 0.0) for u in ids],
                      "hack": [u in gains for u in ids]})
    cfg = OmegaConf.create({"speakers": ["p232", "p257"], "min_seconds": 2.5, "max_seconds": 5.0,
                            "strongest_attack": "a", "n_strongest": 2})
    picks = select_examples(v, meta, cfg, seed=0)
    assert [p["pick"] for p in picks] == ["random", "random", "strongest", "strongest"]
    assert [p["utt_id"][:4] for p in picks[:2]] == ["p232", "p257"]
    random_ids = {p["utt_id"] for p in picks[:2]}
    eligible = sorted((g, u) for u, g in gains.items() if u != "p232_005" and u not in random_ids)[::-1]
    assert [p["utt_id"] for p in picks[2:]] == [u for _, u in eligible[:2]]
    assert "p232_005" not in {p["utt_id"] for p in picks}
    assert picks == select_examples(v, meta, cfg, seed=0)
