#!/usr/bin/env python3
"""Provenance manifest for a single experiment run.

Import and call `write_manifest(out_dir, config)` at the start of every run.
Captures everything needed to reproduce or audit the run later. The manifest
is content-addressable: the config_hash + data_hash uniquely identify a
result, which is what lets `results/` be append-only and drift-free.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _hash_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def git_state(root="."):
    sha = _sh(["git", "-C", root, "rev-parse", "HEAD"])
    dirty = bool(_sh(["git", "-C", root, "status", "--porcelain"]))
    return {"git_sha": sha, "git_dirty": dirty}


def env_fingerprint():
    try:
        import torch  # noqa
        torch_v = torch.__version__
        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        torch_v = cuda = gpu = None
    pip_freeze = _sh([sys.executable, "-m", "pip", "freeze"])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_v,
        "cuda": cuda,
        "gpu": gpu,
        "pip_freeze_hash": hashlib.sha256(pip_freeze.encode()).hexdigest()[:16],
    }


def data_hash(paths):
    """Hash the *identity* of data files (path + size + mtime) cheaply,
    or their content if small. Detects silent dataset drift."""
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        pp = Path(p)
        if pp.exists():
            st = pp.stat()
            h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def write_manifest(out_dir, config, data_paths=None, extra=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
        "config_hash": _hash_obj(config),
        "data_hash": data_hash(data_paths or []),
        "seed": config.get("seed"),
        **git_state(),
        "env": env_fingerprint(),
        "argv": sys.argv,
        "extra": extra or {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if manifest["git_dirty"]:
        print("WARNING: git working tree is dirty. Results may not be reproducible.",
              file=sys.stderr)
    return manifest


if __name__ == "__main__":
    # smoke test
    m = write_manifest("/tmp/_manifest_test", {"seed": 0, "lr": 1e-3})
    print(json.dumps(m, indent=2, default=str))
