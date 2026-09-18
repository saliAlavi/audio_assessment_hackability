"""Locations and one-time download of non-Hugging-Face metric weights.

All files live under ``paths.cache_dir("metric_weights")`` so compute jobs never
need internet access. Run once on a login node::

    python -m hackpatch.metrics.weights

Hugging Face models (Whisper, HuBERT, WavLM-SV) are cached in ``HF_HOME`` by
``transformers`` and are pre-fetched by the same command.
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

from hackpatch.utils import paths

#: DNSMOS P.835 (SIG/BAK/OVRL) and P.808 ONNX models (microsoft/DNS-Challenge, master).
DNSMOS_URLS = {
    "sig_bak_ovr.onnx": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
    "model_v8.onnx": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx",
}
#: Official NISQA v2.0 weights (full model with quality dimensions) and weight licence.
NISQA_URLS = {
    "nisqa.tar": "https://github.com/gabrielmittag/NISQA/raw/master/weights/nisqa.tar",
    "LICENSE_model_weights": "https://github.com/gabrielmittag/NISQA/raw/master/weights/LICENSE_model_weights",
}
#: UTMOS22 strong learner, torch.hub port by tarepan/SpeechMOS (code pinned to tag v1.2.0).
SPEECHMOS_REPO = "https://github.com/tarepan/SpeechMOS.git"
SPEECHMOS_TAG = "v1.2.0"
UTMOS_URL = "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt"

#: Hugging Face checkpoints used by the metric modules.
HF_MODELS = ("openai/whisper-small", "facebook/hubert-base-ls960", "microsoft/wavlm-base-plus-sv")


def weights_root() -> Path:
    """Returns the root directory holding all non-HF metric weights."""
    return paths.cache_dir("metric_weights")


def dnsmos_dir() -> Path:
    """Returns the directory with ``sig_bak_ovr.onnx`` and ``model_v8.onnx``."""
    return weights_root() / "dnsmos"


def nisqa_checkpoint() -> Path:
    """Returns the path of the official ``nisqa.tar`` checkpoint."""
    return weights_root() / "nisqa" / "nisqa.tar"


def speechmos_repo_dir() -> Path:
    """Returns the local checkout of tarepan/SpeechMOS at :data:`SPEECHMOS_TAG`."""
    return weights_root() / "utmos" / f"SpeechMOS-{SPEECHMOS_TAG}"


def utmos_checkpoint() -> Path:
    """Returns the path of the UTMOS22-strong state dict."""
    return weights_root() / "utmos" / Path(UTMOS_URL).name


def _fetch(url: str, dst: Path) -> None:
    """Downloads ``url`` to ``dst`` atomically unless ``dst`` already exists."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dst)
    print(f"downloaded {dst}")


def download_all(hf: bool = True) -> None:
    """Fetches every metric weight file (idempotent).

    Args:
        hf: Also pre-fetch the Hugging Face checkpoints into ``HF_HOME``.
    """
    for name, url in DNSMOS_URLS.items():
        _fetch(url, dnsmos_dir() / name)
    for name, url in NISQA_URLS.items():
        _fetch(url, nisqa_checkpoint().parent / name)
    repo = speechmos_repo_dir()
    if not (repo / "hubconf.py").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", SPEECHMOS_TAG,
                        SPEECHMOS_REPO, str(repo)], check=True)
    _fetch(UTMOS_URL, utmos_checkpoint())
    if hf:
        from huggingface_hub import snapshot_download

        for repo_id in HF_MODELS:
            snapshot_download(repo_id, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"])
            print(f"cached {repo_id}")


if __name__ == "__main__":
    download_all()
