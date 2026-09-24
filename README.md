# How Hackable Is Your Speech Quality Metric?

Code, checkpoints and reproduction notebooks for the paper
**"How Hackable Is Your Speech Quality Metric? A Corrected Protocol, a Benchmark, and What Patching Buys"**
(Ali Alavi, Donald S. Williamson — The Ohio State University; ICASSP 2027 submission, `paper/main.tex`).

Speech quality predictors (SSL-MOS, UTMOS, NISQA, DNSMOS, ...) are increasingly used as training rewards. This
project measures how easily they are *hacked* — pushed to a higher score by audio that is actually worse — and
whether an attack → detect → patch loop fixes it. In short:

* **The usual measurement is flawed.** Scoring an attack against the raw input charges the codec round trip's own
  score shift to the attack; referencing the unperturbed round trip changes measured hackability up to 4×.
  One trained attacker is a sample, not a measurement: five attacker seeds span success rates 0.00–0.38.
* **Benchmark.** Under the corrected protocol NISQA is hacked on 90% of utterances, SSL-MOS 21%, DNSMOS 14%,
  UTMOS 6%.
* **Patching** hardens a predictor only in the attack space it was trained on, by less than the spread between
  attackers, costs up to 0.30 out-of-domain system SRCC, and is matched by a random-perturbation baseline.
  Enhancers trained against patched predictors do hack them far less.

## Repository layout

| path | contents |
|---|---|
| `src/hackpatch/` | library: predictors, codec-latent GRPO attacker, detector panel, patch loop, enhancer, metrics |
| `src/hackpatch/repro.py` | reproduction helpers: fetch released checkpoints, remap stored configs, re-run any run |
| `main.py`, `configs/` | Hydra entry point (`python main.py --config-name <cfg>`) and experiment configs |
| `scripts/` | Slurm submission (`submit.py`, `campaign.py`), data manifests, `repro_run.py`, `build_release.py` |
| `analysis/` | scripts that turn run outputs into `analysis/out/` (every number) and `paper/tables`, `paper/figs` |
| `notebooks/` | one notebook per trained model, table, figure and result (below) |
| `repro/` | `campaign.json` (registry of every run), `configs/<key>.yaml` (exact config per run), `checkpoints.json` |
| `paper/` | LaTeX source, generated tables and figure |
| `preregistration.md`, `LOG.md`, `findings/`, `docs/` | preregistered hypotheses, lab notebook, negative findings, notes |

## Setup

```bash
source scripts/env.sh            # project venv, caches on scratch, PYTHONPATH=src
bash scripts/install_kernel.sh   # Jupyter kernel "hackpatch" with the same environment
```

Data: the audio corpora are not redistributed. Build the manifests with `scripts/build_eval_manifests.py` (BVCC,
NISQA, SOMOS, TMHINT-QI, VMC'23, URGENT) and `scripts/build_vbd_manifests.py` (VoiceBank-DEMAND); sources are in
`docs/eval_corpora.md`.

## Checkpoints

Every checkpoint behind the paper is an asset of the GitHub release
[`checkpoints-v1`](https://github.com/saliAlavi/audio_assessment_hackability/releases/tag/checkpoints-v1):
195 files (19.3 GB — predictors, every patch-loop round, all attackers, enhancers) plus `run_results.tar.gz`
(the per-run outputs the analysis reads). They are release assets rather than git/LFS files because each predictor
is 378 MB. `repro/checkpoints.json` lists each asset with its sha256 and the run it came from.

```python
from hackpatch import repro
repro.fetch()                # everything, sha256-verified
repro.fetch(["e1_R3_s0"])    # only what one run needs
cfg = repro.load_config("hnp_s0")   # the exact config of a run, checkpoint paths remapped to local copies
```

The repository is private; downloads use `GITHUB_TOKEN` or the git credential store.

## Reproducing the paper

Start with `notebooks/00_setup.ipynb`. Every other notebook is self-contained.

**Tables, figure and results never train.** They load the released checkpoints and take a `SOURCE` parameter:
`"recompute"` re-runs the underlying evaluation on GPUs (one Slurm job per run, the kernel waits), `"released"`
only aggregates the original run outputs (CPU, seconds) and regenerates the committed `analysis/out/` and `paper/`
files. Each notebook ends by comparing its numbers with the committed ones.

| notebook | reproduces | recompute cost |
|---|---|---|
| `tables/table1_defences` | Table 1: ASR per attack space and human agreement for every defence | 48 jobs |
| `tables/table2_human_agreement` | Table 2: BVCC and out-of-domain agreement with human ratings | 30 × 3 min |
| `tables/table3_downstream` | Table 3: enhancers post-trained with each predictor as reward | 4 × 15 min |
| `tables/table_bench_published` | benchmark table of SSL-MOS, UTMOS, NISQA, DNSMOS | 4 × 15 min |
| `figures/fig1_results` | Fig. 1: (a) benchmark, (b) attack success per patch round | 10 jobs |
| `results/r0_reproduction` | R0 matches the VMC'22 SSL-MOS baseline | 3 × 3 min |
| `results/reference_check` | E1 under the naive and corrected reference | 24 × 15 min |
| `results/tau_sweep` | ranking of defences across detection thresholds | 24 × 15 min |
| `results/attacker_variance` | five attacker seeds against one predictor | 3 × 15 min |
| `results/loop_rounds` | in-loop statistics of every patch loop and ablation | 12 × 0.5–1 h |
| `results/e2_per_corpus` | R3 − R0 agreement per test set (Holm-corrected) | 6 × 3 min |
| `results/decision_rules` | preregistered hypotheses H1–H3 | E1 + E2 + E3 |
| `results/paper_numbers` | all 1902 quoted numbers + provenance check of `main.tex` | 77 jobs, ~17 A100-h |

**Training notebooks** retrain each model from its stored config, starting from released upstream checkpoints (so
any stage can be retrained alone). `LAUNCH=False` (default) only shows the job; `LAUNCH=True` submits it.

| notebook | trains | time per run |
|---|---|---|
| `training/01_r0_predictor` | R0, the undefended SSL-MOS predictor (3 seeds) | 20 min |
| `training/02_baseline_noise` | baseline: patch on additive-noise pairs | 1 h |
| `training/03_baseline_pgd` | baseline: patch on waveform-PGD pairs | 4 h |
| `training/04_baseline_random` | baseline: patch on random codec-latent pairs | 2 h |
| `training/05_patch_loop` | attack → detect → patch loop, R1–R3 (3 seeds) | 1.3 h |
| `training/06_ablation_loops` | six loop ablations (τ, margin, head-only, preferred side) | 1.3 h |
| `training/07_multispace_loop` | multi-space loop (EnCodec + DAC + PGD) | 2.3 h |
| `training/08_e1_attackers` | fresh EnCodec/DAC attackers against every predictor (E1) | 30 min |
| `training/09_attacker_seed_attackers` | five attacker seeds per predictor | 30 min |
| `training/10_benchmark_attackers` | five attackers per published predictor | 30 min |
| `training/11_eps_pilot` | perturbation-budget pilot (validation only) | 10 min |
| `training/12_enhancer_pretrain` | speech enhancer, supervised pre-training | 45 min |
| `training/13_enhancer_grpo` | enhancer GRPO post-training with a predictor reward | 20 min |

The same operations from the command line:

```bash
python scripts/repro_run.py --mode recompute --kind e2 --submit     # re-run all E2 evaluations as Slurm jobs
python scripts/repro_run.py --mode retrain --key hnp_s0 --submit    # retrain one patch loop
python analysis/recompute_e1.py && python analysis/collect_bench.py && \
  python analysis/analysis.py && python analysis/paper_assets.py    # aggregate → analysis/out, paper/
```

### Verification status

* Aggregating the released outputs regenerates every committed table, the figure (byte-identical) and all 1902
  paper numbers exactly.
* Recomputing from released checkpoints (one run of each kind tested) reproduces the E2 correlations, benchmark,
  attacker-seed control, enhancer evaluation and the EnCodec/DAC loop attacks exactly. Waveform-PGD and DAC attacks
  are not bit-deterministic on GPU: E1 success rates were still identical, but the multi-space loop's in-loop PGD
  success rate drifts slightly (round 3: 0.382 → 0.404).
* Retraining an enhancer from its stored config reproduced the original validation record exactly.
