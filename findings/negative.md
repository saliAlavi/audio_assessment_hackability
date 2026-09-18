# Negative and limiting findings (2026-09-15)

All numbers come from `analysis/analysis.py` (see `docs/results.md`); n = 3 seeds unless stated.

## N1. Hardening does not transfer to an unseen attack space — it can invert
A fresh GRPO attacker in the **DAC** latent space reaches a *higher* attack success rate on every patched
predictor than on R0: ASR 0.120 (R0) → 0.301 / 0.304 / 0.293 (R1 / R2 / R3), i.e. patching against EnCodec
made the predictor **more** hackable through a different codec (mean ASR change −0.173, 2 of 3 seeds negative).
Waveform PGD succeeds on everything (ASR ≈ 1.0, gain +1.8 → +2.1 MOS); patching does not touch white-box
gradient attacks at all. Only the attack family used in the loop (EnCodec latents) and its exact replay
attacker are defeated: ASR 0.181 → 0.050 for a fresh EnCodec attacker, 0.251 → 0.000 for the round-1 attacker.

## N2. The no-RL baseline matches Hack-and-Patch on hackability, at a lower fidelity cost
Random latent perturbation pairs (no RL, no detector) give a fresh-EnCodec ASR of 0.053 versus 0.050 for R3,
while keeping a higher mean system-SRCC over the 13 eval sets (0.714 vs 0.659). On this evidence the RL
attacker is *not* what makes the patch work — the predictor mostly learns "codec-domain perturbation ⇒ worse",
which random pairs teach just as well and more cheaply. This directly contradicts the motivating claim that
"the RL attacker matters".

## N3. In-domain fidelity is preserved but out-of-domain agreement degrades
The preregistered primary (BVCC test system-SRCC) is non-inferior: R3 − R0 = −0.002. But the same patch costs
large, seed-consistent drops on noisy/enhanced speech: TMHINT-QI −0.296, VMC'23 track3 −0.218, NISQA FOR −0.184,
NISQA LIVETALK −0.151 (all 3 seeds same sign). Mean system-SRCC over 13 eval sets: R0 0.745 → R3 0.659.
With n = 3 only VMC'23 track2 survives Holm correction (p = 0.036), so the study is underpowered for
significance, but the effect sizes are far larger than the preregistered −0.02 margin.
**The preregistered H2 metric was too narrow**: it declared success on the one corpus where nothing happens.

## N4. E3 primary metric not supported (but the mechanism is visible)
Counting external measures that degrade versus the pretrained enhancer: R0-reward 2.33, R3-reward 2.33 —
no difference, so H3 as preregistered fails. The count is a poor instrument here: NISQA (+0.3) and UTMOS (+0.06)
*rise* under every reward, i.e. they are blind to this hack, and DNSMOS OVRL and speaker similarity fall under
every reward. The intended effect does appear in the intrusive measures: PESQ −0.226 (R0 reward) vs −0.034
(R3 reward), ESTOI −0.057 vs −0.038, and R3 scores the R0-hacked enhancer at 2.77 while the pretrained model
gets 3.72 — the patched predictor detects the hacked generator even though it was never trained on it.

## N5. The attacker is weak, and the detector's margins are small
No ε reached the preregistered 25% VAL ceiling rate; the best black-box attacker gains ≈ +0.1 MOS on average
(mean action) and hacks concentrate on low-quality inputs (mean R0 input score 2.3–2.5 vs 3.2 overall).
Flagged "hacks" degrade ESTOI by ≈ 0.02 and SI-SDR by ≈ 1.5 dB — real but small, and **not verified by human
listeners** (E4 was skipped). A listening test could show these are not perceptually meaningful.

## N6. Attack diversity displaces the vulnerability instead of removing it
Patching on three attack spaces at once (EnCodec latents + DAC latents + waveform PGD, one shared buffer,
same 3 rounds) suppresses the spaces that dominated before — fresh-DAC ASR 0.120 → 0.037 and waveform PGD
0.999 → 0.841, the only defence that dents PGD at all — while robustness in the EnCodec space is not retained: fresh-EnCodec ASR goes from 0.050 ± 0.021 (R3) to
0.306 ± 0.395, per-seed 0.12 / 0.76 / 0.04 — in one of three seeds the multi-space predictor is markedly more
hackable than the undefended R0 (0.181). The defence is therefore also seed-unstable. Inside the loop the same
displacement is visible: by round 3 the DAC attacker is at 0.008–0.092 and PGD at 0.23–0.43, while the EnCodec
attacker climbs to 0.29–0.82. Multi-space patching also costs the most human agreement of any condition
(out-of-domain mean system SRCC 0.552 vs 0.731 for R0). Across every configuration we tried, the total attack
surface did not shrink; it moved.

## N7. The ceiling rule's reference frame determines the conclusion (the most important finding)
The ceiling rule as run compares the attacked score with the **raw input** `x`, while the patch trains on pairs
referenced to the **codec round trip** `x̂`. Patching moves `R(x̂)` relative to `R(x)`, so the metric partly
measures the defence's own shift: the offset `R(x̂) − R(x)` is −0.09 (R0), −0.27 (R3) and **+0.95** (R3 multi-space).
Recomputing the same stored outputs with the ceiling referenced to `x̂`:
- R3ms goes from "catastrophically hackable" (naive ASR 0.306) to among the most robust (0.082).
- H1 no longer holds: ASR(R0) − ASR(R3) = +0.089, per seed [−0.073, −0.039, +0.379], bootstrap CI [−0.073, +0.379].
- The random-latent baseline becomes the *best* in-space defence (0.019 vs R3's 0.137).
Any future work on hardened quality predictors must reference the hack rule to the processing chain's own output;
otherwise a defence that merely shifts how it scores processed audio can manufacture or hide robustness.
Found by the red-team pass, verified directly from the per-utterance CSVs, and now the paper's lead contribution.

## N8. Attacker-training variance is as large as every defence effect, and the worst case reverses the ranking
Five GRPO attackers trained with different attacker seeds against the *same* fixed predictor (seed 0) give
ASR from 0.000 to 0.261 on R0 (sd 0.130) and 0.000 to 0.382 on R3 (sd 0.180). A single attacker per cell — the
design used in Table 1 and, as far as we can tell, in the attack literature — is therefore not a measurement.
Taking the worst of five attackers, which is what a defence claim requires, reverses the ordering: R0 0.261,
R3 0.382 (the EnCodec loop is *worse* than no defence), R3-multi-space 0.086 with the only small spread (0.021).

## N9. The benchmark (positive result, and the reason the paper exists)
Under the corrected protocol at one budget, published predictors differ by more than an order of magnitude:
NISQA is hacked on 89.6% of utterances for +0.954 MOS, SSL-MOS on 20.5%, DNSMOS OVRL on 13.6%, UTMOS on 5.7%.
NISQA and DNSMOS are the ones most often used as RL rewards. Unlike our defence comparisons, this ranking is
robust: the spread over five independently trained attackers is small within each predictor.
