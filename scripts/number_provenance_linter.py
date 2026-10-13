#!/usr/bin/env python3
"""Number-provenance linter for research papers.

Extracts every numeric literal from a .tex file's prose and checks that each
one appears in a generated artifact under an analysis directory (CSV/JSON/TXT).
A number in the paper that can't be traced to a generated artifact is either a
typo or a fabrication — both must be fixed before submission.

Usage:
    python number_provenance_linter.py paper/main.tex --artifacts analysis/

Numbers you intend to be untraceable (years, equation constants, hyperparameters
stated in the method) can be whitelisted with a trailing LaTeX comment:
    84.2\\%  % lint:ok
or globally in a .lintignore file (one regex per line).

Exit 0 if all numbers trace; 1 if any don't.
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Matches numbers like 84.2, 0.031, 1,234, 3e-4, 12  (not inside \ref, labels)
NUM_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w])")

# Lines/spans to skip entirely
SKIP_ENV = re.compile(r"\\begin\{(equation|align|tabular|figure|table)\*?\}"
                      r".*?\\end\{\1\*?\}", re.DOTALL)
COMMENT_OK = re.compile(r"%\s*lint:ok")
CITE_YEAR = re.compile(r"\\cite[a-z]*\{[^}]*\}")


def collect_artifact_numbers(artifact_dir: Path):
    """Gather every number that appears in generated artifacts."""
    nums = set()
    for p in artifact_dir.rglob("*"):
        if p.suffix.lower() in {".csv", ".json", ".txt", ".tsv", ".md"} and p.is_file():
            text = p.read_text(errors="ignore")
            for m in NUM_RE.finditer(text):
                nums.add(_norm(m.group(1)))
    return nums


def _norm(s: str) -> str:
    s = s.replace(",", "")
    try:
        f = float(s)
        # store a few rounded forms so 84.2 in tex matches 84.20 in csv
        return f"{f:.6g}"
    except ValueError:
        return s


def load_ignore(tex_path: Path):
    ignore = [re.compile(r"^(19|20)\d{2}$")]  # years by default
    f = tex_path.parent / ".lintignore"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignore.append(re.compile(line))
    return ignore


def lint(tex_path: Path, artifact_dir: Path):
    tex = tex_path.read_text()
    tex = SKIP_ENV.sub(" ", tex)       # drop math/table environments
    tex = CITE_YEAR.sub(" ", tex)      # drop citation keys
    artifact_nums = collect_artifact_numbers(artifact_dir)
    ignore = load_ignore(tex_path)

    problems = []
    for i, line in enumerate(tex.splitlines(), 1):
        if COMMENT_OK.search(line):
            continue
        code = line.split("%")[0]      # strip comments
        for m in NUM_RE.finditer(code):
            raw = m.group(1)
            if any(rx.match(raw.replace(",", "")) for rx in ignore):
                continue
            if _norm(raw) not in artifact_nums:
                problems.append((i, raw, line.strip()[:80]))
    return problems, len(artifact_nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--artifacts", default="analysis/")
    args = ap.parse_args()

    tex_path, art_dir = Path(args.tex), Path(args.artifacts)
    if not tex_path.exists():
        print(f"error: {tex_path} not found"); return 2
    if not art_dir.exists():
        print(f"error: artifact dir {art_dir} not found"); return 2

    problems, n_art = lint(tex_path, art_dir)
    print(f"Checked {tex_path} against {n_art} numbers in {art_dir}")
    if not problems:
        print("PASS: every paper number traces to a generated artifact.")
        return 0
    print(f"FAIL: {len(problems)} untraceable number(s):")
    for ln, raw, ctx in problems:
        print(f"  line {ln}: {raw!r}  in: {ctx}")
    print("\nFix each: regenerate from analysis.py, or whitelist with "
          "'% lint:ok' / .lintignore if legitimately non-derived.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
