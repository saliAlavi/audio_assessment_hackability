# How Hackable Is Your Speech Quality Metric?

[![arXiv](https://img.shields.io/badge/arXiv-2610.10899-b31b1b.svg)](https://arxiv.org/abs/2610.10899)
[![Project page](https://img.shields.io/badge/project-page-1f6fb5.svg)](https://salialavi.github.io/audio_assessment_hackability/)
[![Checkpoints](https://img.shields.io/badge/checkpoints-19.3%20GB-555.svg)](https://github.com/saliAlavi/audio_assessment_hackability/releases/tag/checkpoints-v1)

Code, checkpoints and reproduction notebooks for the paper

> **How Hackable Is Your Speech Quality Metric? A Corrected Protocol, a Benchmark, and What Patching Buys**<br>
> Ali Alavi, Donald S. Williamson (The Ohio State University)<br>
> arXiv:2610.10899, 2026 · [paper](https://arxiv.org/abs/2610.10899) · [PDF](https://arxiv.org/pdf/2610.10899) ·
> [project page](https://salialavi.github.io/audio_assessment_hackability/)

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

## Where each result of the paper comes from

Every table, figure and quoted number of [arXiv:2610.10899](https://arxiv.org/abs/2610.10899) has a notebook that
regenerates it and compares it with the paper:

| in the paper | notebook | recompute cost |
|---|---|---|
| Table I: hackability of SSL-MOS, UTMOS22, NISQA, DNSMOS | [`tables/table1_benchmark`](notebooks/tables/table1_benchmark.ipynb) | 4 × 15 min |
| Table II: ASR per attack space and human agreement for every defence | [`tables/table2_loop_audit`](notebooks/tables/table2_loop_audit.ipynb) | 48 jobs |
| Table III: enhancers post-trained with each predictor as reward | [`tables/table3_downstream`](notebooks/tables/table3_downstream.ipynb) | 4 × 15 min |
| Fig. 2: (a) benchmark, (b) attack success per patch round | [`figures/fig2_results`](notebooks/figures/fig2_results.ipynb) | 10 jobs |
| Sec. V, "The reference of the hack rule changes the answer"; abstract's 4× | [`results/reference_check`](notebooks/results/reference_check.ipynb) | 24 × 15 min |
| Sec. V, "One attacker is not a measurement"; abstract's 0.00–0.38 | [`results/attacker_variance`](notebooks/results/attacker_variance.ipynb) | 3 × 15 min |
| Sec. V, τ sweep in "Patching hardens narrowly" | [`results/tau_sweep`](notebooks/results/tau_sweep.ipynb) | 24 × 15 min |
| Sec. V, "Fidelity": BVCC and out-of-domain agreement, utterance MSE | [`results/human_agreement`](notebooks/results/human_agreement.ipynb) | 30 × 3 min |
| Sec. V, "Fidelity": change per test set (TMHINT-QI) | [`results/e2_per_corpus`](notebooks/results/e2_per_corpus.ipynb) | 6 × 3 min |
| Fig. 2b, pair-side ablation (Sec. III), replay-buffer sizes (Sec. IV) | [`results/loop_rounds`](notebooks/results/loop_rounds.ipynb) | 12 × 0.5–1 h |
| preregistered hypotheses H1–H3 (Sec. V) | [`results/decision_rules`](notebooks/results/decision_rules.ipynb) | E1 + E2 + E3 |
| supporting check: R0 matches the VMC'22 SSL-MOS baseline | [`results/r0_reproduction`](notebooks/results/r0_reproduction.ipynb) | 3 × 3 min |
| **all 1902 quoted numbers**, plus a provenance check of `paper/main.tex` and of the arXiv source | [`results/paper_numbers`](notebooks/results/paper_numbers.ipynb) | 77 jobs, ~17 A100-h |

Fig. 1 is the method diagram. Every model behind these results has a training notebook (below).

## Repository layout

| path | contents |
|---|---|
| `src/hackpatch/` | library: predictors, codec-latent GRPO attacker, detector panel, patch loop, enhancer, metrics |
| `src/hackpatch/repro.py` | reproduction helpers: fetch released checkpoints, remap stored configs, re-run any run |
| `main.py`, `configs/` | Hydra entry point (`python main.py --config-name <cfg>`) and experiment configs |
| `scripts/` | environment (`env.sh`), data manifests, Slurm submission, `repro_run.py`, `build_release.py`, provenance linter |
| `analysis/` | scripts that turn run outputs into `analysis/out/` (every number) and `paper/tables`, `paper/figs` |
| `notebooks/` | one notebook per trained model, table, figure and result (above and below) |
| `repro/` | `campaign.json` (registry of every run), `configs/<key>.yaml` (exact config per run), `checkpoints.json` |
| `paper/` | LaTeX source of arXiv:2610.10899, generated tables and figure |
| `website/` | the [project page](https://salialavi.github.io/audio_assessment_hackability/) and its build script |
| `preregistration.md`, `findings/`, `docs/` | preregistered hypotheses, negative findings, corpus sources, generated results |

## Setup

**Environment** (Python 3.11; the pins are the environment that produced the paper):

```bash
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt   # DAC: its metadata pins protobuf<3.20, which onnx/wandb cannot share
cp scripts/env.local.example.sh scripts/env.local.sh   # then set SCRATCH_DIR etc. (not tracked)
source scripts/env.sh                              # caches on $SCRATCH_DIR, PYTHONPATH=src
bash scripts/install_kernel.sh                     # Jupyter kernel "hackpatch" with the same environment
```

`scripts/env.sh` reads these variables from the environment or from `scripts/env.local.sh` (git-ignored, one per
machine; `scripts/env.local.example.sh` is the template).

| variable | default | used for |
|---|---|---|
| `SCRATCH_DIR` | `~/hackpatch_scratch` | run outputs, model caches, downloaded checkpoints |
| `HACKPATCH_DATA_ROOT` | the original cluster's corpus storage | root of the speech corpora (`docs/eval_corpora.md`) |
| `HACKPATCH_ARTIFACTS` | `$SCRATCH_DIR/cache/audio_assessment_hackability_release` | where checkpoints and released outputs are downloaded |
| `HACKPATCH_SLURM_ACCOUNT`, `HACKPATCH_SLURM_PARTITION` | unset (cluster default) | `sbatch` account and partition when notebooks run on Slurm |
| `HACKPATCH_WANDB_MODE`, `WANDB_ENTITY` | `offline`, your default entity | Weights & Biases logging of retrained runs |
| `HACKPATCH_VENV` | unset | a virtualenv for `env.sh` to activate |

**Data.** The audio corpora are not redistributed. `docs/eval_corpora.md` lists where to download each one (BVCC,
the VoiceMOS'22 OOD track, NISQA, SOMOS, TMHINT-QI, VoiceMOS'23, URGENT'24/'25, VoiceBank-DEMAND), its terms, and the
directory layout the builders expect. Then:

```bash
python scripts/build_eval_manifests.py --data-root /data/speech   # BVCC (R0 training and test) + 12 out-of-domain sets
python scripts/build_vbd_manifests.py --root /data/speech/datasets_voicebank/noisy-vctk-16k --pack   # E3
```

On the original data both builders reproduce the manifests of the paper's runs byte for byte. Only the *recompute*
and *retrain* modes need the audio; aggregating the released outputs (below) does not.

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

Downloads need no account. If the public URL fails (a private copy of the repository), `repro` retries through the
GitHub API with a token from `GITHUB_TOKEN` or the git credential store. Upstream models (wav2vec 2.0 base, EnCodec
24 kHz, DAC 16 kHz, Whisper-small, HuBERT, NISQA, DNSMOS, UTMOS22) are downloaded from their public sources on first
use.

## Reproducing the paper

Start with `notebooks/00_setup.ipynb`. Every other notebook is self-contained.

**Tables, figure and results never train.** They load the released checkpoints and take a `SOURCE` parameter:
`"recompute"` re-runs the underlying evaluation on GPUs, `"released"` only aggregates the original run outputs (CPU,
seconds, no audio needed) and regenerates the committed `analysis/out/` and `paper/` files. Each notebook ends by
comparing its numbers with the committed ones. `RUN="slurm"` submits one GPU job per run and waits; `RUN="local"`
runs them one after another in the kernel (one GPU, no Slurm). Both are also environment variables:

```bash
NB_SOURCE=released jupyter nbconvert --to notebook --execute notebooks/results/paper_numbers.ipynb       # CPU, minutes
NB_SOURCE=recompute NB_RUN=local jupyter nbconvert --to notebook --execute notebooks/tables/table1_benchmark.ipynb
```

**Training notebooks** retrain each model from its stored config, starting from released upstream checkpoints (so
any stage can be retrained alone). `LAUNCH=False` (default) only shows the job; `LAUNCH=True` (`NB_LAUNCH=1`)
starts it.

| notebook | trains | time per run |
|---|---|---|
| `training/01_r0_predictor` | R0, the undefended SSL-MOS predictor (3 seeds) | 20 min |
| `training/02_baseline_noise` | baseline: patch on additive-noise pairs | 1 h |
| `training/03_baseline_pgd` | baseline: patch on waveform-PGD pairs | 4 h |
| `training/04_baseline_random` | baseline: patch on random codec-latent pairs | 2 h |
| `training/05_patch_loop` | attack → detect → patch loop, R1–R3 (3 seeds) | 1.3 h |
| `training/06_ablation_loops` | six loop ablations (τ, margin, head-only, preferred side) | 1.3 h |
| `training/07_multispace_loop` | multi-space loop (EnCodec + DAC + PGD) | 2.3 h |
| `training/08_e1_attackers` | fresh EnCodec/DAC attackers against every predictor (Table II) | 30 min |
| `training/09_attacker_seed_attackers` | five attacker seeds per predictor | 30 min |
| `training/10_benchmark_attackers` | five attackers per published predictor (Table I) | 30 min |
| `training/11_eps_pilot` | perturbation-budget pilot (validation only) | 10 min |
| `training/12_enhancer_pretrain` | speech enhancer, supervised pre-training | 45 min |
| `training/13_enhancer_grpo` | enhancer GRPO post-training with a predictor reward (Table III) | 20 min |

The same operations from the command line:

```bash
python scripts/repro_run.py --mode recompute --kind e2 --submit     # re-run all E2 evaluations as Slurm jobs
python scripts/repro_run.py --mode retrain --key hnp_s0             # retrain one patch loop on this GPU
python analysis/recompute_e1.py && python analysis/collect_bench.py && \
  python analysis/analysis.py && python analysis/paper_assets.py    # aggregate → analysis/out, paper/
```

### Verification status

* Aggregating the released outputs regenerates every committed table, the figure (byte-identical) and all 1902
  paper numbers exactly. Every number in `paper/main.tex` and in the LaTeX source posted to arXiv traces to a
  generated file.
* Recomputing from released checkpoints (one run of each kind tested) reproduces the E2 correlations, benchmark,
  attacker-seed control, enhancer evaluation and the EnCodec/DAC loop attacks exactly. Waveform-PGD and DAC attacks
  are not bit-deterministic on GPU: E1 success rates were still identical, but the multi-space loop's in-loop PGD
  success rate drifts slightly (round 3: 0.382 → 0.404).
* Retraining an enhancer from its stored config reproduced the original validation record exactly.

## Project page

`website/` is the [project page](https://salialavi.github.io/audio_assessment_hackability/), deployed to GitHub
Pages by `.github/workflows/pages.yml` on every push to `main` that touches it. Its tables and benchmark chart are
generated from `paper/tables/` and `analysis/out/`: `python website/build.py` (add `--figures` to re-render the
figures from `paper/`); `tests/test_repro.py` fails if the page and the paper disagree.

Its listening examples (`website/static/audio/`) are not a paper result. BVCC audio may not be redistributed, so
`configs/audio_examples.yaml` applies the released attackers unchanged to the VoiceBank-DEMAND clean test set
(CC BY 4.0), scores every clip with the four predictors and the detector panel, and picks the utterances shown by a
fixed rule (one drawn at random per speaker, plus the largest SSL-MOS hack):

```bash
python scripts/submit.py --config-name audio_examples --time 01:30:00                # one GPU
python website/build.py --audio-from <run dir>/results/audio_examples
```

## License

The code is released under the [MIT License](LICENSE); `src/hackpatch/metrics/_nisqa/` keeps the NISQA MIT notice.
The released checkpoints were trained on BVCC and VoiceBank-DEMAND, whose terms (`docs/eval_corpora.md`) still apply.

## Citation

```bibtex
@misc{alavi2026hackable,
  title         = {How Hackable Is Your Speech Quality Metric? A Corrected Protocol,
                   a Benchmark, and What Patching Buys},
  author        = {Alavi, Ali and Williamson, Donald S.},
  year          = {2026},
  eprint        = {2610.10899},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.10899}
}
```
