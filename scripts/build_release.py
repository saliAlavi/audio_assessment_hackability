#!/usr/bin/env python3
"""Packages every checkpoint and run output the paper depends on as a GitHub release.

Predictor checkpoints are 378 MB each, above GitHub's 100 MB per-file git
limit, so checkpoints are published as release assets rather than committed.
This script is the only writer of ``repro/checkpoints.json`` (asset manifest
with sha256) and ``repro/configs/`` (the resolved config of every registered
run), which the notebooks use to re-run any result from the checkpoints.

What is collected, for every run in the campaign registry:

* every checkpoint its resolved config points to (predictors, attackers,
  enhancers), with ``selected_checkpoint.txt`` / ``done.json`` pointers
  resolved and recorded so they can be remapped on another machine;
* every loop round's predictor and attacker (needed to recompute in-loop ASR);
* the attackers trained inside E1, the attacker-seed control and the benchmark;
* a results bundle (``run_results.tar.gz``) with the small per-run outputs the
  analysis scripts read, so tables can also be rebuilt without a GPU.

    python scripts/build_release.py                 # manifest + configs + bundle
    python scripts/build_release.py --upload        # ... and upload to the release
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hackpatch import repro  # noqa: E402
from hackpatch.utils import paths  # noqa: E402

PROJ = paths.SCRATCH_DIR / "projects" / paths.project_name()
BUILD = paths.SCRATCH_DIR / "cache" / f"{paths.project_name()}_release_build"
# Per-run outputs the analysis scripts read (relative to the run dir).
RESULT_GLOBS = ["resolved_config.yaml", "manifest.json", "results/*.csv", "results/*.json", "results/*.txt",
                "round*/done.json", "round*/attack_train*.csv"]
# Files outside registered run dirs that analysis/analysis.py reads.
EXTRA_FILES = ["_campaign.json", "_compute_gpu_hours.txt", "pilot_eps3/h0_e0.5.json", "pilot_eps3/h0_e1.0.json",
               "pilot_eps3/h0_e2.0.json", "pilot_eps3/h25_e0.5.json", "pilot_eps3/h25_e1.0.json",
               "pilot_eps3/h25_e2.0.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def resolve_pointer(value: str) -> str | None:
    """Returns the checkpoint a config value points to, or None if it is not a checkpoint path."""
    p = Path(value)
    if not str(p).startswith(str(PROJ)) or not p.exists():
        return None
    if p.suffix == ".txt":
        return p.read_text().strip()
    if p.suffix == ".json":
        return json.loads(p.read_text())["ckpt"]
    return str(p) if p.suffix == ".pt" else None


def config_strings(node) -> list[str]:
    """All string leaves of a parsed YAML config."""
    if isinstance(node, dict):
        return [s for v in node.values() for s in config_strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in config_strings(v)]
    return [node] if isinstance(node, str) else []


def asset_name(path: str, owners: dict[str, str]) -> str:
    """Flat, GitHub-safe asset name: ``{registry key}.{path inside the run dir}``."""
    run_dir, rel = path[len(str(PROJ)) + 1:].split("/", 1)
    owner = owners.get(run_dir, run_dir)
    return re.sub(r"[^A-Za-z0-9._-]", ".", f"{owner}.{rel.replace('/', '.')}")


def collect(reg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Returns ({checkpoint path: role}, {pointer path: checkpoint path})."""
    import yaml

    ckpts: dict[str, str] = {}
    pointers: dict[str, str] = {}
    for key, v in reg.items():
        rd = Path(v["run_dir"])
        cfg = yaml.safe_load((rd / "resolved_config.yaml").read_text())
        for s in config_strings(cfg):
            target = resolve_pointer(s)
            if target is None:
                continue
            if s != target:
                pointers[s] = target
            ckpts.setdefault(target, "referenced")
        for done in sorted(rd.glob("round*/done.json")):
            info = json.loads(done.read_text())
            pointers[str(done)] = info["ckpt"]
            for k in ("ckpt", "prev_ckpt"):
                ckpts.setdefault(info[k], "loop_predictor")
        for f in sorted(rd.glob("round*/attacker*.pt")) + sorted(rd.glob("checkpoints/attacker*.pt")):
            ckpts.setdefault(str(f), "attacker")
        sel = rd / "results" / "selected_checkpoint.txt"
        if sel.exists():
            pointers[str(sel)] = sel.read_text().strip()
            ckpts.setdefault(sel.read_text().strip(), "selected")
    missing = [c for c in ckpts if not Path(c).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced checkpoints are missing, e.g. {missing[:3]}")
    return ckpts, pointers


def build_bundle(reg: dict, out: Path) -> None:
    """Writes the results bundle with paths relative to the project runs dir."""
    with tarfile.open(out, "w:gz") as tar:
        for rel in EXTRA_FILES:
            if (PROJ / rel).exists():
                tar.add(PROJ / rel, arcname=rel)
        for v in reg.values():
            rd = Path(v["run_dir"])
            for pat in RESULT_GLOBS:
                for f in sorted(rd.glob(pat)):
                    tar.add(f, arcname=str(f.relative_to(PROJ)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", action="store_true", help="Upload assets to the GitHub release.")
    ap.add_argument("--skip-bundle", action="store_true")
    args = ap.parse_args()
    reg = json.loads((PROJ / "_campaign.json").read_text())
    owners = {Path(v["run_dir"]).name: k for k, v in reg.items()}

    cfg_dir = ROOT / "repro" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for key, v in reg.items():
        (cfg_dir / f"{key}.yaml").write_text((Path(v["run_dir"]) / "resolved_config.yaml").read_text())
    # The enhancer pre-training runs predate the registry; their configs are kept for retraining.
    for s in (0, 1, 2):
        run = max(p for p in PROJ.glob(f"se_pretrain_s{s}__*") if "smoke" not in p.name)
        (cfg_dir / f"se_pretrain_s{s}.yaml").write_text((run / "resolved_config.yaml").read_text())
    (ROOT / "repro" / "campaign.json").write_text(json.dumps(reg, indent=1) + "\n")

    ckpts, pointers = collect(reg)
    old = {}
    if repro.MANIFEST.exists():
        old = {a["orig_path"]: a for a in json.loads(repro.MANIFEST.read_text())["assets"]}
    assets = []
    for path, role in sorted(ckpts.items()):
        size = os.path.getsize(path)
        prev = old.get(path)
        digest = prev["sha256"] if prev and prev["size"] == size else sha256(Path(path))
        assets.append({"asset": asset_name(path, owners), "orig_path": path, "role": role, "size": size,
                       "sha256": digest})
        print(f"{assets[-1]['asset']:<70} {size / 1e6:8.1f} MB  {role}", flush=True)
    names = [a["asset"] for a in assets]
    assert len(names) == len(set(names)), "asset name collision"

    BUILD.mkdir(parents=True, exist_ok=True)
    bundle = BUILD / repro.RESULTS_BUNDLE
    if not args.skip_bundle:
        build_bundle(reg, bundle)
    manifest = {
        "repo": repro.REPO, "tag": repro.TAG, "orig_runs_dir": str(PROJ),
        "results_bundle": {"asset": repro.RESULTS_BUNDLE, "size": bundle.stat().st_size, "sha256": sha256(bundle)},
        "pointers": dict(sorted(pointers.items())),
        "assets": assets,
    }
    repro.MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(assets)} checkpoints, {sum(a['size'] for a in assets) / 1e9:.2f} GB; "
          f"bundle {bundle.stat().st_size / 1e6:.1f} MB -> {repro.MANIFEST}")
    if args.upload:
        upload([(a["asset"], Path(a["orig_path"])) for a in assets] + [(repro.RESULTS_BUNDLE, bundle)])


def upload(files: list[tuple[str, Path]]) -> None:
    """Creates the release if needed and uploads every missing asset (idempotent)."""
    token = repro.github_token()
    api = f"https://api.github.com/repos/{repro.REPO}"
    hdr = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    def call(url: str, method: str = "GET", data: bytes | None = None, extra: dict | None = None):
        req = urllib.request.Request(url, data=data, method=method, headers={**hdr, **(extra or {})})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or b"null")

    try:
        rel = call(f"{api}/releases/tags/{repro.TAG}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        body = {"tag_name": repro.TAG, "name": "Checkpoints and run outputs",
                "body": "Every checkpoint behind the paper's results, plus the per-run outputs the analysis reads. "
                        "Manifest with sha256: repro/checkpoints.json. Download with "
                        "`python -c 'from hackpatch import repro; repro.fetch()'` or notebooks/00_setup.ipynb."}
        rel = call(f"{api}/releases", "POST", json.dumps(body).encode())
    have = {a["name"]: a for a in call(f"{api}/releases/{rel['id']}/assets?per_page=100")}
    page = 2
    while len(have) == 100 * (page - 1):
        more = call(f"{api}/releases/{rel['id']}/assets?per_page=100&page={page}")
        if not more:
            break
        have |= {a["name"]: a for a in more}
        page += 1
    upload_url = rel["upload_url"].split("{")[0]
    for name, path in files:
        size = path.stat().st_size
        if name in have and have[name]["size"] == size and have[name]["state"] == "uploaded":
            continue
        if name in have:
            call(f"{api}/releases/assets/{have[name]['id']}", "DELETE")
        print(f"uploading {name} ({size / 1e6:.0f} MB)", flush=True)
        # curl streams from disk; urllib would hold a 378 MB body in memory.
        subprocess.run(["curl", "-sS", "--fail", "--retry", "3", "-o", "/dev/null", "-X", "POST",
                        "-H", f"Authorization: token {token}", "-H", "Content-Type: application/octet-stream",
                        "--data-binary", f"@{path}", f"{upload_url}?name={urllib.parse.quote(name)}"], check=True)


if __name__ == "__main__":
    main()
