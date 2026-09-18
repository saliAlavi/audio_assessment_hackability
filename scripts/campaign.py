#!/usr/bin/env python3
"""Submits the preregistered campaign as a Slurm dependency graph.

Stages (each can be submitted before its inputs exist; checkpoints are passed
as run-time pointers resolved by ``hackpatch.models.ssl_mos.resolve_ckpt``):

    main       Hack-and-Patch loops + 3 baselines, per seed (needs R0 run dirs)
    ablations  seed-0 loop variants (τ, margin, head-only patch, pair side)
    e1         hackability of every predictor (after main)
    e2         correlation on all eval corpora (after main + ablations)

The registry of what was submitted is appended to
``{SCRATCH}/projects/{PROJECT}/_campaign.json``.

    python scripts/campaign.py --stage main --r0 s0=/run/dir s1=... --eps 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import submit_one  # noqa: E402

from hackpatch.utils import paths  # noqa: E402

REGISTRY = paths.SCRATCH_DIR / "projects" / paths.project_name() / "_campaign.json"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def save_registry(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=1))


def record(reg: dict, key: str, job: str, run_dir: Path, **extra) -> None:
    reg[key] = {"job": job, "run_dir": str(run_dir), **extra}
    save_registry(reg)


def stage_main(reg: dict, seeds: list[int], eps: float, dry: bool) -> None:
    """Per seed: one H&P loop job, and one GPU lane running noise → random → PGD baselines."""
    common = [f"attack.eps={eps}"]
    for s in seeds:
        r0_ckpt = f"{reg[f'r0_s{s}']['run_dir']}/results/selected_checkpoint.txt"
        job, rd = submit_one("hack_loop", s, time="10:00:00", overrides=common + [f"init_ckpt={r0_ckpt}"],
                             dry_run=dry)
        record(reg, f"hnp_s{s}", job, rd, kind="hnp")
        lane = None
        for name, t in [("baseline_noise", "02:00:00"), ("baseline_random", "03:00:00"),
                        ("baseline_pgd", "05:00:00")]:
            ov = [f"init_ckpt={r0_ckpt}"] + (common if name == "baseline_random" else [])
            lane, rd = submit_one(name, s, time=t, overrides=ov, dry_run=dry, lane_after=lane)
            record(reg, f"{name}_s{s}", lane, rd, kind="baseline")


def stage_ablations(reg: dict, eps: float, n_lanes: int, dry: bool) -> None:
    """Seed-0 loop variants, serialized into ``n_lanes`` GPU lanes."""
    r0_ckpt = f"{reg['r0_s0']['run_dir']}/results/selected_checkpoint.txt"
    variants = {
        "tau0.1": ["detect.tau=0.1"],
        "margin0.1": ["patch.margin=0.1"],
        "headonly": ["model.freeze_encoder=true", "patch.lr_head=1.0e-3"],
        "tau0.5": ["detect.tau=0.5"],
        "margin1.0": ["patch.margin=1.0"],
        "prefinput": ["loop.pair_preferred=input"],
    }
    lanes: list[str | None] = [None] * n_lanes
    for i, (tag, ov) in enumerate(variants.items()):
        lane = i % n_lanes
        job, rd = submit_one("hack_loop", 0, tag=tag, time="10:00:00", lane_after=lanes[lane],
                             overrides=[f"attack.eps={eps}", f"init_ckpt={r0_ckpt}", "phase=ablation"] + ov,
                             dry_run=dry)
        lanes[lane] = job
        record(reg, f"abl_{tag}_s0", job, rd, kind="ablation")


def predictor_pointers(reg: dict, seed: int) -> dict[str, tuple[str, str | None]]:
    """Name -> (checkpoint pointer, job id that produces it) for one seed's main predictors."""
    out = {"R0": (f"{reg[f'r0_s{seed}']['run_dir']}/results/selected_checkpoint.txt", None)}
    hnp = reg[f"hnp_s{seed}"]
    for k in (1, 2, 3):
        out[f"R{k}"] = (f"{hnp['run_dir']}/round{k}/done.json", hnp["job"])
    for name in ("baseline_noise", "baseline_pgd", "baseline_random"):
        b = reg[f"{name}_s{seed}"]
        out[name] = (f"{b['run_dir']}/results/selected_checkpoint.txt", b["job"])
    return out


def stage_e1(reg: dict, seeds: list[int], eps: float, targets: list[str], n_lanes: int, dry: bool) -> None:
    """E1 jobs ordered by priority (R0/R3 first, then baselines, then R1/R2) in ``n_lanes`` GPU lanes."""
    priority = {"R0": 0, "R3": 0, "baseline_noise": 1, "baseline_pgd": 1, "baseline_random": 1, "R1": 2, "R2": 2}
    jobs = sorted(((priority.get(t, 3), s, t) for s in seeds for t in targets))
    lanes: list[str | None] = [None] * n_lanes
    for i, (_, s, name) in enumerate(jobs):
        if f"e1_{name}_s{s}" in reg:
            continue
        preds = predictor_pointers(reg, s)
        hnp = reg[f"hnp_s{s}"]
        ckpt, dep = preds[name]
        deps = ":".join(sorted({j for j in (dep, hnp["job"]) if j}))
        ov = [f"attack.eps={eps}", f"target_name={name}", f"target_ckpt={ckpt}",
              f"transfer_attacker_ckpt={hnp['run_dir']}/round1/attacker.pt"]
        lane = i % n_lanes
        job, rd = submit_one("e1_attack", s, tag=name, time="04:00:00", after=deps or None,
                             lane_after=lanes[lane], overrides=ov, dry_run=dry)
        lanes[lane] = job
        record(reg, f"e1_{name}_s{s}", job, rd, kind="e1", target=name)


def stage_e2(reg: dict, seeds: list[int], only: list[str] | None, dry: bool) -> None:
    """One E2 job per (predictor, seed) so predictors evaluate in parallel as soon as they exist."""
    for s in seeds:
        preds = {"R0": (f"{reg[f'r0_s{s}']['run_dir']}/results/selected_checkpoint.txt", None)}
        if f"hnp_s{s}" in reg:
            preds = predictor_pointers(reg, s)
        if s == 0:
            for key, v in reg.items():
                if v.get("kind") == "ablation":
                    preds[key.replace(".", "p")] = (f"{v['run_dir']}/round3/done.json", v["job"])
        for name, (ptr, dep) in preds.items():
            if (only and name not in only) or f"e2_{name}_s{s}" in reg:
                continue
            job, rd = submit_one("e2_eval", s, tag=name, time="04:00:00", after=dep,
                                 overrides=[f"+predictors.{name}='{ptr}'"], dry_run=dry)
            record(reg, f"e2_{name}_s{s}", job, rd, kind="e2", predictor=name)


def stage_e3(reg: dict, seeds: list[int], rewards: list[str], n_lanes: int, dry: bool) -> None:
    """E3: GRPO enhancement post-training with each predictor as reward, then test-set eval per seed."""
    import glob

    lanes: list[str | None] = [None] * n_lanes
    i = 0
    for s in seeds:
        pre = glob.glob(str(paths.SCRATCH_DIR / "projects" / paths.project_name() / f"se_pretrain_s{s}__*"))
        pre = [p for p in pre if "smoke" not in p]
        pre_ckpt = f"{sorted(pre)[-1]}/checkpoints/best.pt"
        preds = predictor_pointers(reg, s)
        grpo_runs, grpo_jobs = {}, []
        for name in rewards:
            key = f"e3_grpo_{name}_s{s}"
            ptr, dep = preds[name]
            cross = [f"+reward.cross_ckpts.{n}='{p}'" for n, (p, _) in preds.items()
                     if n != name and n in ("R0", "R3")]
            if key not in reg:
                lane = i % n_lanes
                job, rd = submit_one("se_grpo", s, tag=name, time="01:00:00", after=dep, lane_after=lanes[lane],
                                     overrides=[f"init_ckpt={pre_ckpt}", f"reward.ckpt='{ptr}'"] + cross,
                                     dry_run=dry)
                lanes[lane] = job
                i += 1
                record(reg, key, job, rd, kind="e3_grpo", reward=name)
            grpo_runs[name] = reg[key]["run_dir"]
            grpo_jobs.append(reg[key]["job"])
        if f"e3_eval_s{s}" in reg:
            continue
        ov = [f"+checkpoints.pretrained={pre_ckpt}"]
        ov += [f"+checkpoints.grpo_{n}={rd}/checkpoints/last.pt" for n, rd in grpo_runs.items()]
        ov += [f"+predictors.{n}='{p}'" for n, (p, _) in preds.items() if n in rewards]
        job, rd = submit_one("se_eval", s, time="02:00:00", after=":".join(grpo_jobs), overrides=ov, dry_run=dry)
        record(reg, f"e3_eval_s{s}", job, rd, kind="e3_eval")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["register_r0", "main", "ablations", "e1", "e2", "e3"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--r0", nargs="*", default=[], help="seed=run_dir for register_r0")
    ap.add_argument("--targets", default="R0,R1,R2,R3,baseline_noise,baseline_pgd,baseline_random")
    ap.add_argument("--lanes", type=int, default=2, help="Parallel GPU lanes for ablations.")
    ap.add_argument("--rewards", default="R0,R3,baseline_noise,baseline_pgd,baseline_random")
    ap.add_argument("--only", default=None, help="Comma list of predictor names for e2.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    reg = load_registry()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.stage == "register_r0":
        for item in args.r0:
            s, rd = item.split("=")
            record(reg, f"r0_s{int(s)}", "done", Path(rd), kind="r0")
    elif args.stage == "main":
        stage_main(reg, seeds, args.eps, args.dry_run)
    elif args.stage == "ablations":
        stage_ablations(reg, args.eps, args.lanes, args.dry_run)
    elif args.stage == "e1":
        stage_e1(reg, seeds, args.eps, args.targets.split(","), args.lanes, args.dry_run)
    elif args.stage == "e3":
        stage_e3(reg, seeds, args.rewards.split(","), args.lanes, args.dry_run)
    elif args.stage == "e2":
        stage_e2(reg, seeds, args.only.split(",") if args.only else None, args.dry_run)


if __name__ == "__main__":
    main()
