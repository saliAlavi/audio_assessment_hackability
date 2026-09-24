#!/usr/bin/env python3
"""Re-runs one registered run from released checkpoints (see ``hackpatch.repro``).

    python scripts/repro_run.py --mode recompute --key e1_R3_s0     # evaluation only, on this GPU
    python scripts/repro_run.py --mode retrain --key hnp_s0         # training, on this GPU
    python scripts/repro_run.py --mode recompute --kind e2 --submit # one Slurm job per E2 key
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hackpatch import repro  # noqa: E402

#: Wall time per kind for --submit (original job durations with headroom).
TIME = {"e2": "00:30:00", "e1": "02:00:00", "e1_attacker_seeds": "02:00:00", "bench": "02:00:00",
        "e3_eval": "01:00:00", "e3_eval_r2": "01:00:00", "hnp": "03:00:00", "ablation": "03:00:00",
        "multispace": "04:00:00", "r0": "02:00:00", "baseline": "05:00:00", "e3_grpo": "01:00:00"}
RETRAIN_TIME = {"hnp": "10:00:00", "ablation": "10:00:00", "multispace": "10:00:00", "e1": "04:00:00",
                "e1_attacker_seeds": "04:00:00", "bench": "04:00:00"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["recompute", "retrain"], required=True)
    ap.add_argument("--key", nargs="*", default=[], help="Registry keys.")
    ap.add_argument("--kind", nargs="*", default=[], help="Select every key of these registry kinds.")
    ap.add_argument("--submit", action="store_true", help="One Slurm job per key instead of running here.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    reg = repro.registry()
    keys = list(args.key) + [k for k, v in reg.items() if v["kind"] in args.kind]
    for key in keys:
        kind = reg[key]["kind"] if key in reg else "se_pretrain"
        if args.submit:
            t = (RETRAIN_TIME if args.mode == "retrain" else {}).get(kind, TIME.get(kind, "02:00:00"))
            repro.submit(key, args.mode, time=t, dry_run=args.dry_run)
        elif args.mode == "recompute":
            repro.recompute(key)
        else:
            repro.retrain(key)


if __name__ == "__main__":
    main()
