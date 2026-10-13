"""Tests for the reproduction helpers (``hackpatch.repro``) and the project page; CPU only, no network.

    python -m pytest tests/test_repro.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hackpatch import repro

ROOT = Path(__file__).resolve().parents[1]


def _fake_curl(calls: list, public_ok: bool):
    """A ``subprocess.run`` stand-in for curl: the public release URL fails unless ``public_ok``."""
    def run(cmd, **kwargs):
        calls.append(cmd)
        if "api.github.com" not in cmd[-1] and not public_ok:
            return subprocess.CompletedProcess(cmd, 22, "", "curl: (22) The requested URL returned error: 404")
        Path(cmd[cmd.index("-o") + 1]).write_text("checkpoint")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run


def test_download_public_release_needs_no_token(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(repro.subprocess, "run", _fake_curl(calls, public_ok=True))
    monkeypatch.setattr(repro, "github_token", lambda: (_ for _ in ()).throw(AssertionError("token requested")))
    repro._download("r0_s0.pt", tmp_path / "r0_s0.pt")
    assert len(calls) == 1
    assert calls[0][-1] == f"https://github.com/{repro.REPO}/releases/download/{repro.TAG}/r0_s0.pt"
    assert (tmp_path / "r0_s0.pt").read_text() == "checkpoint"


def test_download_private_release_falls_back_to_token(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(repro.subprocess, "run", _fake_curl(calls, public_ok=False))
    monkeypatch.setattr(repro, "github_token", lambda: "tok")
    monkeypatch.setattr(repro, "_release_assets", lambda: {"r0_s0.pt": 7})
    repro._download("r0_s0.pt", tmp_path / "r0_s0.pt")
    assert calls[1][-1] == f"https://api.github.com/repos/{repro.REPO}/releases/assets/7"
    assert "Authorization: token tok" in calls[1]
    assert (tmp_path / "r0_s0.pt").exists() and not (tmp_path / "r0_s0.pt.part").exists()


def test_submit_takes_slurm_settings_from_env(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "RUNS", tmp_path)
    monkeypatch.setenv("HACKPATCH_SLURM_ACCOUNT", "ACC123")
    monkeypatch.delenv("HACKPATCH_SLURM_PARTITION", raising=False)
    assert repro.submit("r0_s0", "retrain", dry_run=True) == "dry-run"
    script = (tmp_path / "retrain" / "_logs" / "r0_s0.sbatch").read_text()
    assert "#SBATCH --account=ACC123\n" in script
    assert "--partition" not in script
    assert "python scripts/repro_run.py --mode retrain --key r0_s0" in script


def test_project_page_matches_paper_tables():
    r = subprocess.run([sys.executable, str(ROOT / "website" / "build.py"), "--check"], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
