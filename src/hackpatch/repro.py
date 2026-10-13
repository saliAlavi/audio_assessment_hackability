"""Reproduction helpers: released checkpoints, stored run configs, and re-running any result.

Every run behind the paper is a key in the campaign registry (``r0_s0``,
``e1_R3_s1``, ``bench_nisqa``, ...). For each key the repo stores the exact
resolved config the run used (``repro/configs/{key}.yaml``), and the GitHub
release ``TAG`` holds every checkpoint those configs point to
(``repro/checkpoints.json`` lists them with sha256).

* :func:`fetch` downloads (or, on the original cluster, symlinks) checkpoints.
* :func:`load_config` returns a stored config with every checkpoint path
  remapped to the local copy.
* :func:`recompute` re-runs a key's *evaluation* from checkpoints (no training):
  E1 and benchmark attacks with the stored attackers, E2 correlations, E3
  enhancement scoring, and the in-loop attacks of the patch loops.
* :func:`retrain` re-runs a key's *training* from its stored config, starting
  from released upstream checkpoints (e.g. a patch loop starts from released R0).
* :func:`submit` runs either as a Slurm job via ``scripts/repro_run.py`` (account and
  partition from ``HACKPATCH_SLURM_ACCOUNT`` / ``HACKPATCH_SLURM_PARTITION``).
* :func:`results_registry` returns a registry whose run dirs hold released or
  recomputed outputs, for the analysis scripts.

Outputs go under ``{SCRATCH}/projects/{PROJECT}/{recompute,retrain}/{key}/``.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from hackpatch.utils import paths

ROOT = Path(__file__).resolve().parents[2]
REPO = "saliAlavi/audio_assessment_hackability"
TAG = "checkpoints-v1"
#: The paper (arXiv 2610.10899) whose results this package reproduces.
ARXIV_ID = "2610.10899"
MANIFEST = ROOT / "repro" / "checkpoints.json"
CONFIGS = ROOT / "repro" / "configs"
RESULTS_BUNDLE = "run_results.tar.gz"
#: Where downloaded checkpoints and the extracted results bundle live.
ARTIFACTS = Path(os.environ.get("HACKPATCH_ARTIFACTS",
                                paths.SCRATCH_DIR / "cache" / f"{paths.project_name()}_release"))
RUNS = paths.SCRATCH_DIR / "projects" / paths.project_name()

#: Registry kinds that are evaluations of fixed checkpoints (recomputable without training).
EVAL_KINDS = {"e1", "e1_attacker_seeds", "bench", "e2", "e3_eval", "e3_eval_r2", "hnp", "multispace", "ablation"}
#: Registry kinds that train something (re-runnable with :func:`retrain`).
TRAIN_KINDS = {"r0", "baseline", "hnp", "ablation", "multispace", "e1", "e1_attacker_seeds", "bench", "e3_grpo"}


# ------------------------------------------------------------------ manifest
def manifest() -> dict:
    """The release manifest (``repro/checkpoints.json``)."""
    return json.loads(MANIFEST.read_text())


def registry() -> dict:
    """The campaign registry as stored in the repo (``repro/campaign.json``)."""
    return json.loads((ROOT / "repro" / "campaign.json").read_text())


def _orig_run_dir(key: str) -> str:
    return registry()[key]["run_dir"]


def local_path(orig_path: str) -> Path:
    """Local location of a released checkpoint, given its original path."""
    by_orig = {a["orig_path"]: a for a in manifest()["assets"]}
    return ARTIFACTS / "checkpoints" / by_orig[orig_path]["asset"]


def github_token() -> str | None:
    """Token for a private release: ``GITHUB_TOKEN``/``GH_TOKEN`` or the git credential store."""
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    for line in out.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@functools.cache
def _release_assets() -> dict[str, int]:
    """{asset name: asset id} of the release (API; only needed while the repository is private)."""
    import urllib.request

    token = github_token()
    hdr = {"Accept": "application/vnd.github+json", **({"Authorization": f"token {token}"} if token else {})}
    api = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
    with urllib.request.urlopen(urllib.request.Request(api, headers=hdr)) as r:
        rel = json.loads(r.read())
    out, page = {}, 1
    while True:
        url = f"https://api.github.com/repos/{REPO}/releases/{rel['id']}/assets?per_page=100&page={page}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr)) as r:
            batch = json.loads(r.read())
        out |= {a["name"]: a["id"] for a in batch}
        if len(batch) < 100:
            return out
        page += 1


def _download(asset: str, dest: Path) -> None:
    """Downloads one release asset: anonymously from the public release URL, or, if that fails
    (private repository), through the API with :func:`github_token`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = ["curl", "-sSL", "--fail", "--retry", "3", "-o", str(tmp)]
    public = subprocess.run(cmd + [f"https://github.com/{REPO}/releases/download/{TAG}/{asset}"],
                            capture_output=True, text=True)
    if public.returncode:
        token = github_token()
        if not token:
            raise RuntimeError(f"cannot download {asset}: {public.stderr.strip()} (a private release needs "
                               "GITHUB_TOKEN)")
        subprocess.run(cmd + ["-H", "Accept: application/octet-stream", "-H", f"Authorization: token {token}",
                              f"https://api.github.com/repos/{REPO}/releases/assets/{_release_assets()[asset]}"],
                       check=True)
    tmp.rename(dest)


def assets_for(keys: list[str] | None = None, include_own: bool = True) -> list[dict]:
    """Manifest entries needed by the given registry keys (all when None).

    Args:
        keys: Registry keys (or ``se_pretrain_s{seed}``); None selects every asset.
        include_own: Also include checkpoints the key's own run produced (needed to
            re-evaluate it, not to retrain it).
    """
    assets = manifest()["assets"]
    if keys is None:
        return assets
    need = set()
    for key in keys:
        run_dir = _orig_run_dir(key) if key in registry() else None
        text = (CONFIGS / f"{key}.yaml").read_text()
        for a in assets:
            own = run_dir is not None and a["orig_path"].startswith(run_dir + "/")
            if (include_own and own) or a["orig_path"] in text:
                need.add(a["asset"])
        for ptr, target in manifest()["pointers"].items():
            if ptr in text:
                need.add(next(a["asset"] for a in assets if a["orig_path"] == target))
    return [a for a in assets if a["asset"] in need]


def fetch(keys: list[str] | None = None, prefer_local: bool = True, verify: bool = True,
          include_own: bool = True) -> list[Path]:
    """Makes the checkpoints of ``keys`` (default: all) available under ``ARTIFACTS/checkpoints``.

    Args:
        keys: Registry keys whose checkpoints are needed; None fetches everything (~19 GB).
        prefer_local: Symlink the original file when it still exists (the original cluster)
            instead of downloading it.
        verify: Check sha256 of downloaded files.
        include_own: See :func:`assets_for`.

    Returns:
        Local paths of the fetched checkpoints.
    """
    out = []
    for a in assets_for(keys, include_own):
        dest = ARTIFACTS / "checkpoints" / a["asset"]
        if not (dest.exists() and dest.stat().st_size == a["size"]):
            if dest.is_symlink():
                dest.unlink()
            if prefer_local and Path(a["orig_path"]).exists() and os.path.getsize(a["orig_path"]) == a["size"]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(a["orig_path"])
            else:
                print(f"[fetch] {a['asset']} ({a['size'] / 1e6:.0f} MB)", flush=True)
                _download(a["asset"], dest)
                if verify and _sha256(dest) != a["sha256"]:
                    dest.unlink()
                    raise RuntimeError(f"sha256 mismatch for {a['asset']}")
        out.append(dest)
    print(f"[fetch] {len(out)} checkpoints ready in {ARTIFACTS / 'checkpoints'}", flush=True)
    return out


# ------------------------------------------------------------------- configs
def remap(value: str) -> str:
    """Maps an original checkpoint path or pointer (``.txt``/``done.json``) to its local copy."""
    m = manifest()
    if not isinstance(value, str) or not value.startswith(m["orig_runs_dir"]):
        return value
    target = m["pointers"].get(value, value)
    by_orig = {a["orig_path"]: a for a in m["assets"]}
    return str(ARTIFACTS / "checkpoints" / by_orig[target]["asset"]) if target in by_orig else value


def load_config(key: str, remap_paths: bool = True):
    """The exact resolved config of a registered run, with checkpoint paths made local.

    Args:
        key: Registry key (``repro/configs/{key}.yaml``), e.g. ``"e1_R3_s0"``.
        remap_paths: Replace original checkpoint paths/pointers by released local copies.

    Returns:
        An OmegaConf ``DictConfig``.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(CONFIGS / f"{key}.yaml"))

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return remap(node) if remap_paths else node

    return OmegaConf.create(walk(cfg))


def _start(cfg, run_name: str, wandb_mode: str):
    """Starts a logged run (scratch dir + manifest + W&B) with a fixed name.

    The W&B entity is ``WANDB_ENTITY`` (unset: your default entity), not the one stored in the config.
    """
    from omegaconf import OmegaConf

    from hackpatch.utils.run import init_run

    cfg = OmegaConf.merge(cfg, {"run_name": run_name,
                                "wandb": {"mode": wandb_mode, "entity": os.environ.get("WANDB_ENTITY")}})
    return (cfg, *init_run(cfg))


# ------------------------------------------------------------------ recompute
def recompute(key: str, wandb_mode: str = "disabled", force: bool = False) -> Path:
    """Re-runs the evaluation behind one registry key from released checkpoints (no training).

    * ``e1`` / ``e1_attacker_seeds`` / ``bench``: the stored attackers are placed
      in the new run's checkpoint dir, so the task skips attacker training and
      only attacks and scores the evaluation set.
    * ``e2``: correlation of each predictor on every eval corpus.
    * ``e3_eval`` / ``e3_eval_r2``: scoring of the enhancement checkpoints.
    * ``hnp`` / ``multispace`` / ``ablation``: each round's stored attacker
      re-attacks ``R_{k-1}`` on the loop's train subset (mean action only),
      i.e. the in-loop ASR of the round figure.

    Args:
        key: Registry key.
        wandb_mode: ``disabled`` (default), ``offline`` or ``online``.
        force: Recompute even if the output run already finished.

    Returns:
        The output run dir ``{RUNS}/recompute/{key}``.
    """
    from hackpatch import tasks

    kind = registry()[key]["kind"]
    if kind not in EVAL_KINDS:
        raise ValueError(f"{key} ({kind}) has no evaluation to recompute; use retrain()")
    out = RUNS / "recompute" / key
    if (out / "DONE").exists() and not force:
        print(f"[recompute] {key} already done -> {out}", flush=True)
        return out
    fetch([key])
    cfg = load_config(key)
    cfg, rp, run = _start(cfg, f"recompute/{key}", wandb_mode)
    try:
        if kind in ("e1", "e1_attacker_seeds", "bench"):
            orig = Path(_orig_run_dir(key))
            for a in manifest()["assets"]:
                p = Path(a["orig_path"])
                if p.parent == orig / "checkpoints" and p.name.startswith("attacker"):
                    shutil.copyfile(local_path(a["orig_path"]), rp.checkpoints / p.name)
            getattr(tasks, cfg.task)(cfg, rp, run)
        elif kind in ("e2", "e3_eval", "e3_eval_r2"):
            getattr(tasks, cfg.task)(cfg, rp, run)
        else:
            _recompute_loop(key, cfg, rp)
        (rp.root / "DONE").touch()
    finally:
        run.finish()
    return rp.root


def _recompute_loop(key: str, cfg, rp) -> None:
    """Re-attacks ``R_{k-1}`` with round k's stored attacker(s); writes ``round{k}/attack_train*.csv``."""
    import gc

    import torch
    from omegaconf import OmegaConf

    from hackpatch.attack.grpo_attacker import Attacker
    from hackpatch.data.manifests import read_manifest
    from hackpatch.metrics.panel import Panel
    from hackpatch.models.ssl_mos import load_predictor
    from hackpatch.patch.loop import attack_and_detect, pgd_attack_and_detect

    orig = Path(_orig_run_dir(key))
    train_df = read_manifest(cfg.data.corpus, "train")
    attack_train_df = train_df.sample(cfg.loop.max_train_utts, random_state=cfg.seed) \
        if cfg.loop.get("max_train_utts") else train_df
    # Loops run before multi-space support had one EnCodec space and unsuffixed file names.
    spaces = cfg.loop.get("attack_spaces")
    legacy = spaces is None
    spaces = spaces or [OmegaConf.create({"name": "encodec", "type": "grpo"})]
    for k in range(1, cfg.loop.rounds + 1):
        rd = Path(rp.root) / f"round{k}"
        rd.mkdir(exist_ok=True)
        done = json.loads((orig / f"round{k}" / "done.json").read_text())
        # The training-time record (pair counts, patch stats) is copied, not recomputed.
        (rd / "done.json").write_text(json.dumps(done, indent=1))
        predictor = load_predictor(remap(done["prev_ckpt"]), cfg.model.pretrained)
        panel = Panel(**cfg.panel)
        for si, space in enumerate(spaces):
            torch.manual_seed(cfg.seed * 1000 + k * 10 + si)
            if space.type == "grpo":
                acfg = OmegaConf.merge(cfg.attack, space.get("attack_overrides") or {})
                attacker = Attacker(acfg)
                src = orig / f"round{k}" / ("attacker.pt" if legacy else f"attacker_{space.name}.pt")
                attacker.load_state_dict(torch.load(local_path(str(src)), map_location="cuda", weights_only=False))
                res = attack_and_detect(attacker, predictor, attack_train_df, panel,
                                        OmegaConf.merge(cfg, {"attack": acfg}), None, 0)
                del attacker
            else:
                res = pgd_attack_and_detect(predictor, attack_train_df, panel, cfg, space.pgd, None)
            res.to_csv(rd / ("attack_train.csv" if legacy else f"attack_train_{space.name}.csv"), index=False)
            torch.cuda.empty_cache()
        del predictor, panel
        gc.collect()
        torch.cuda.empty_cache()


# -------------------------------------------------------------------- retrain
def retrain(key: str, wandb_mode: str | None = None) -> Path:
    """Re-runs the training behind one registry key from its stored config.

    Upstream checkpoints (the R0 a loop starts from, the predictor used as a
    GRPO reward, ...) are the released ones, so any stage can be retrained on
    its own. E1 / attacker-seed / benchmark keys train fresh attackers and then
    evaluate them, exactly like the original job.

    Args:
        key: Registry key of a training run (see ``TRAIN_KINDS``) or ``se_pretrain_s{seed}``.
        wandb_mode: W&B mode for the new run; default ``HACKPATCH_WANDB_MODE``, else ``offline``.

    Returns:
        The output run dir ``{RUNS}/retrain/{key}``.
    """
    from omegaconf import OmegaConf

    from hackpatch import tasks

    if not key.startswith("se_pretrain") and registry()[key]["kind"] not in TRAIN_KINDS:
        raise ValueError(f"{key} does not train anything; use recompute()")
    wandb_mode = wandb_mode or os.environ.get("HACKPATCH_WANDB_MODE", "offline")
    fetch([key], include_own=False)  # upstream inputs only; this key's own outputs get retrained
    cfg = load_config(key)
    if cfg.task == "hack_loop" and cfg.loop.get("attack_spaces") is None:
        cfg = OmegaConf.merge(cfg, {"loop": {"attack_spaces": [{"name": "encodec", "type": "grpo"}]}})
    cfg, rp, run = _start(cfg, f"retrain/{key}", wandb_mode)
    try:
        getattr(tasks, cfg.task)(cfg, rp, run)
        (rp.root / "DONE").touch()
    finally:
        run.finish()
    return rp.root


# --------------------------------------------------------------------- slurm
def submit(key: str, mode: str = "recompute", time: str = "02:00:00", after: str | None = None,
           dry_run: bool = False) -> str:
    """Submits ``scripts/repro_run.py --mode {mode} --key {key}`` as a one-GPU Slurm job.

    The account and partition come from ``HACKPATCH_SLURM_ACCOUNT`` and
    ``HACKPATCH_SLURM_PARTITION`` (set per machine in ``scripts/env.local.sh``);
    either is left to the cluster default when unset.

    Returns:
        The Slurm job id (or ``"dry-run"``).
    """
    log_dir = RUNS / mode / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    slurm = "".join(f"#SBATCH --{opt}={os.environ[var]}\n" for opt, var in
                    (("account", "HACKPATCH_SLURM_ACCOUNT"), ("partition", "HACKPATCH_SLURM_PARTITION"))
                    if os.environ.get(var))
    script = f"""#!/bin/bash
{slurm}#SBATCH --job-name={mode}_{key}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
{f'#SBATCH --dependency=afterok:{after}' if after else ''}
set -euo pipefail
cd {ROOT}
source scripts/env.sh
python scripts/repro_run.py --mode {mode} --key {key}
"""
    path = log_dir / f"{key}.sbatch"
    path.write_text(script)
    if dry_run:
        print(script)
        return "dry-run"
    job = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True, check=True).stdout
    job = job.strip().split(";")[0]
    print(f"[submit] {mode} {key} -> job {job}, log {log_dir}/{mode}_{key}_{job}.out", flush=True)
    return job


# ------------------------------------------------------------------- results
def fetch_results() -> Path:
    """Downloads and extracts the released per-run outputs; returns the runs dir that holds them."""
    m = manifest()["results_bundle"]
    dest = ARTIFACTS / "runs"
    tgz = ARTIFACTS / RESULTS_BUNDLE
    stamp = dest / ".bundle_sha256"
    if not (stamp.exists() and stamp.read_text() == m["sha256"]):
        if dest.exists():
            shutil.rmtree(dest)
        if not (tgz.exists() and _sha256(tgz) == m["sha256"]):
            tgz.unlink(missing_ok=True)
            _download(RESULTS_BUNDLE, tgz)
            if _sha256(tgz) != m["sha256"]:
                raise RuntimeError("sha256 mismatch for the results bundle")
        with tarfile.open(tgz) as tar:
            tar.extractall(dest, filter="data")
        # Point the registry at the extracted copies, not the original cluster paths.
        reg = json.loads((dest / "_campaign.json").read_text())
        for v in reg.values():
            v["run_dir"] = str(dest / Path(v["run_dir"]).name)
        (dest / "_campaign.json").write_text(json.dumps(reg, indent=1))
        stamp.write_text(m["sha256"])
    return dest


def fetch_arxiv_source(version: str = "") -> Path:
    """Downloads and extracts the LaTeX source of the paper as posted on arXiv.

    Args:
        version: An arXiv version such as ``"v1"``; empty for the latest.

    Returns:
        The directory holding the posted ``main.tex``, ``tables/`` and ``figs/``.
    """
    dest = ARTIFACTS / f"arxiv_{ARXIV_ID}{version or '_latest'}"
    tgz = dest.parent / f"{dest.name}.tar.gz"
    tgz.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-A", "hackpatch-repro", "-o", str(tgz),
                    f"https://arxiv.org/src/{ARXIV_ID}{version}"], check=True)
    if dest.exists():
        shutil.rmtree(dest)
    with tarfile.open(tgz) as tar:
        tar.extractall(dest, filter="data")
    return dest


def results_registry(source: str) -> Path:
    """Builds a runs dir whose ``_campaign.json`` points at released or recomputed outputs.

    Args:
        source: ``"released"`` — the outputs of the original runs (release bundle);
            ``"recompute"`` — outputs of :func:`recompute` where they exist, else
            the released ones (training kinds have nothing to recompute).

    Returns:
        A runs dir for the analysis scripts (``HACKPATCH_RUNS_DIR``).
    """
    released = fetch_results()
    if source == "released":
        return released
    if source != "recompute":
        raise ValueError(source)
    out = RUNS / "recompute"
    out.mkdir(parents=True, exist_ok=True)
    reg = {}
    for key, v in registry().items():
        new = out / key
        use_new = v["kind"] in EVAL_KINDS and (new / "DONE").exists()
        if v["kind"] in EVAL_KINDS and not use_new:
            print(f"[registry] {key}: not recomputed yet, using released outputs", flush=True)
        reg[key] = {**v, "run_dir": str(new if use_new else released / Path(v["run_dir"]).name),
                    "source": "recompute" if use_new else "released"}
    (out / "_campaign.json").write_text(json.dumps(reg, indent=1))
    for extra in ("_compute_gpu_hours.txt", "pilot_eps3"):
        if not (out / extra).exists():
            (out / extra).symlink_to(released / extra)
    return out


def load_registry(runs_dir: Path) -> dict:
    """Reads ``{runs_dir}/_campaign.json``; run dirs that do not exist are relocated under ``runs_dir``."""
    reg = json.loads((Path(runs_dir) / "_campaign.json").read_text())
    for v in reg.values():
        if not Path(v["run_dir"]).exists():
            v["run_dir"] = str(Path(runs_dir) / Path(v["run_dir"]).name)
    return reg


# ------------------------------------------------------------------ notebooks
def wait(jobs: list[str], poll: int = 60) -> None:
    """Blocks until the given Slurm jobs have left the queue; raises if any did not complete."""
    import time

    jobs = [j for j in jobs if j and j != "dry-run"]
    while jobs:
        q = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(jobs)], capture_output=True, text=True)
        live = set(q.stdout.split())
        if not live:
            break
        print(f"[wait] {len(live)} job(s) still queued/running", flush=True)
        time.sleep(poll)
    bad = []
    for j in jobs:
        st = subprocess.run(["sacct", "-n", "-X", "-j", j, "--format=State"], capture_output=True, text=True)
        if st.stdout.split() and st.stdout.split()[0] != "COMPLETED":
            bad.append((j, st.stdout.split()[0]))
    if bad:
        raise RuntimeError(f"jobs did not complete: {bad}; logs in {RUNS}/<mode>/_logs")


def run_keys(keys: list[str], mode: str = "recompute", run: str = "slurm", time: str | None = None) -> None:
    """Recomputes or retrains ``keys`` in this kernel (``run="local"``) or as Slurm jobs, then waits."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from repro_run import RETRAIN_TIME, TIME

    reg = registry()
    if run == "local":
        for k in keys:
            (recompute if mode == "recompute" else retrain)(k)
        return
    jobs = []
    for k in keys:
        if mode == "recompute" and (RUNS / "recompute" / k / "DONE").exists():
            continue
        kind = reg[k]["kind"] if k in reg else "se_pretrain"
        t = time or (RETRAIN_TIME if mode == "retrain" else {}).get(kind, TIME.get(kind, "02:00:00"))
        jobs.append(submit(k, mode, time=t))
    wait(jobs)


def output_dirs(source: str) -> dict[str, Path]:
    """Where the analysis writes: the repo itself for released outputs, ``notebooks/out/`` otherwise."""
    if source == "released":
        return {"out": ROOT / "analysis" / "out", "paper": ROOT / "paper", "results_md": ROOT / "docs" / "results.md"}
    base = ROOT / "notebooks" / "out" / source
    return {"out": base / "analysis", "paper": base / "paper", "results_md": base / "results.md"}


def run_analysis(source: str) -> dict[str, Path]:
    """Runs the whole analysis chain (recompute_e1 → collect_bench → analysis → paper_assets).

    Args:
        source: ``released`` (outputs of the original runs; regenerates the committed
            ``analysis/out`` and ``paper/`` files) or ``recompute`` (outputs of
            :func:`recompute` where available; writes to ``notebooks/out/recompute``).

    Returns:
        The output dirs (keys ``out``, ``paper``, ``results_md``).
    """
    import sys

    runs_dir = results_registry(source)
    dirs = output_dirs(source)
    for d in (dirs["out"], dirs["paper"] / "tables", dirs["paper"] / "figs"):
        d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HACKPATCH_RUNS_DIR": str(runs_dir), "HACKPATCH_ANALYSIS_OUT": str(dirs["out"]),
           "HACKPATCH_PAPER_DIR": str(dirs["paper"]), "HACKPATCH_RESULTS_MD": str(dirs["results_md"]),
           "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}"}
    for script in ("recompute_e1.py", "collect_bench.py", "analysis.py", "paper_assets.py"):
        r = subprocess.run([sys.executable, str(ROOT / "analysis" / script)], env=env, cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    print(f"[analysis] runs={runs_dir}\n[analysis] out={dirs['out']}\n[analysis] paper={dirs['paper']}", flush=True)
    return dirs


def compare_numbers(new_json: Path, ref_json: Path = ROOT / "analysis" / "out" / "paper_numbers.json",
                    prefixes: tuple[str, ...] = ()):
    """Side-by-side of every paper number (``paper_numbers.json``) against the committed values.

    Args:
        new_json: A ``paper_numbers.json`` produced by :func:`run_analysis`.
        ref_json: The committed one (what the paper quotes).
        prefixes: Keep only keys starting with one of these (all when empty).

    Returns:
        A DataFrame with ``paper``, ``reproduced``, ``abs_diff`` per number, largest differences first.
    """
    import pandas as pd

    new, ref = json.loads(Path(new_json).read_text()), json.loads(Path(ref_json).read_text())
    keys = [k for k in ref if not prefixes or k.startswith(prefixes)]
    df = pd.DataFrame({"paper": [ref[k] for k in keys], "reproduced": [new.get(k) for k in keys]}, index=keys)
    df["abs_diff"] = (df.reproduced.astype(float) - df.paper.astype(float)).abs()
    return df.sort_values("abs_diff", ascending=False)


def show_pdf(path: Path, dpi: int = 200):
    """Renders the first page of a PDF figure inline (needs ``pdftoppm``)."""
    import tempfile

    from IPython.display import Image

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(path), f"{tmp}/fig"], check=True)
        return Image(filename=f"{tmp}/fig.png")


def diff_text(new: Path, ref: Path) -> str:
    """Unified diff of a regenerated text artifact (e.g. a ``.tex`` table) against the committed one."""
    import difflib

    a, b = Path(ref).read_text().splitlines(), Path(new).read_text().splitlines()
    d = list(difflib.unified_diff(a, b, "committed", "regenerated", lineterm=""))
    return "identical to the committed file" if not d else "\n".join(d)


def compare_csv(new: Path, ref: Path):
    """Largest absolute difference per numeric column of a regenerated CSV vs the committed one.

    Rows are aligned by the non-numeric columns (e.g. target, attacker) plus row order within them.

    Returns:
        A Series ``column -> max |diff|`` (NaN where a column is missing in one of the files).
    """
    import pandas as pd

    a, b = pd.read_csv(ref), pd.read_csv(new)
    keys = [c for c in a.columns if a[c].dtype == object and c in b]
    if keys:
        a = a.sort_values(keys, kind="stable").reset_index(drop=True)
        b = b.sort_values(keys, kind="stable").reset_index(drop=True)
    num = [c for c in a.columns if c not in keys and pd.api.types.is_numeric_dtype(a[c])]
    out = {}
    for c in num:
        out[c] = float((pd.to_numeric(b[c], errors="coerce").astype(float) - a[c].astype(float)).abs().max()) \
            if c in b and len(a) == len(b) else float("nan")
    return pd.Series(out, name="max_abs_diff")
