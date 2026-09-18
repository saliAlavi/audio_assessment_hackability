# Preregistration — Hack-and-Patch (v2, 2026-09-15; v1 checksum 82234197… in scratch `_prereg/`)

Written before any test-label contact. BVCC test *audio* (no labels) is the E1 attack set; test labels
are first read by the E2 job. Any change after this point is logged in `LOG.md` as `prereg-v2` with a reason.

## Seeds
`SEEDS = [0, 1, 2]` for every main condition (R0, Hack-and-Patch, three baselines, E1, E3). Ablations: seed 0 only
(labeled exploratory). n=3 is below the protocol's n≥5 because of the one-day compute window; every
number is reported with n and with an utterance-level bootstrap CI in addition to the seed spread.

## Conditions
- **R0**: SSL-MOS (wav2vec2-base, mean pool, linear), MSE on BVCC train, best BVCC-val utt-MSE checkpoint.
- **Hack-and-Patch (R1–R3)**: 3 rounds; fresh GRPO EnCodec-latent attacker vs R_{k-1}; ceiling rule τ=0.3 AND panel rule
  (ESTOI −0.02, SI-SDR −1 dB, WER >0 increase, SpeechBERTScore −0.02, vs the codec reconstruction); pairs x̂ ≻ y;
  patch = MSE + margin ranking (m=0.5, w=1) over the replay buffer, 1500 steps/round, last checkpoint.
- **Baselines** (from R0, same 4500-step patch budget, same losses): Gaussian-noise pairs; waveform PGD pairs;
  random latent pairs at budget ε (no RL, no detector).

## Amendment v2 (before any main run; VAL-only evidence, see LOG.md pilots v1–v3)
- Attacker action space: one time-invariant C-dim latent offset per utterance (`action_hop=0`), projected onto the
  RMS sphere of radius ε (`budget_mode=sphere`), λ=0, lr 1e-3, 2000 GRPO steps. Frame-rate actions (v1) did not learn.
- ε rule changed: the v1 rule assumed attack gain grows with ε; the VAL pilot shows it does not
  (ceiling rate 22.3% / 21.5% / 18.5% at ε 0.5 / 1 / 2). New rule: **ε = argmax VAL ceiling rate over {0.5, 1, 2} → ε = 0.5**.
  No ε reached the 25% threshold; this is reported as a finding (R0 is only moderately hackable by a black-box latent attacker).
- DAC attacker (E1 unseen codec) uses the quantized DAC latent (continuous-latent reconstruction loses 0.52 MOS on VAL).

## Attack budget ε (v1 text, superseded by v2 above)
Pilot: GRPO EnCodec attacker vs R0 seed 0, ε ∈ {0.25, 0.5, 1.0, 2.0}, 1000 steps, 400 BVCC-val utterances.
Rule: **ε = the smallest value whose mean-action ceiling rate (gain > 0.3) is ≥ 25%**. If none reaches it, ε = 2.0 and
this is reported as a finding. The same ε is used for every attacker and every target.

## Hypotheses, primary metrics and decision rules
- **H1 (hackability, E1).** Primary: attack success rate (ASR = fraction of BVCC-test utterances whose mean-action output
  is flagged by ceiling ∧ panel, scored with the *target* predictor) of a **fresh** GRPO-EnCodec attacker.
  Supported iff mean over seeds of ASR(R0) − ASR(R3) ≥ 0.10 and the reduction holds in all 3 seeds.
  Secondary: mean score gain; ASR under unseen attackers (GRPO-DAC, waveform PGD, transferred round-1 attacker);
  external-judge (UTMOS, DNSMOS) deltas of attacked outputs.
  Transfer claim: hardening "transfers" to an unseen attacker iff that attacker's ASR also drops by ≥ 0.10.
- **H2 (fidelity, E2).** Primary: BVCC test system-level SRCC. Non-inferiority: mean over seeds of
  SRCC(R3) − SRCC(R0) ≥ −0.02. Secondary: utterance/system MSE, LCC, SRCC, KTAU on BVCC test, VMC'22 OOD,
  NISQA test sets, SOMOS, TMHINT-QI, VMC'23 tracks, URGENT'24/'25 SQA (whichever are obtainable), 95% bootstrap CIs.
- **H3 (downstream, E3).** GRPO-post-train one small mask-based enhancement model with reward R0 vs R3 (same seeds,
  same steps). Primary: number of external measures among {DNSMOS OVRL, NISQA, UTMOS, WER, speaker similarity}
  that get worse than the pretrained model with a paired-bootstrap 95% CI excluding 0 on VoiceBank-DEMAND test.
  Supported iff that count is lower for R3 than for R0 in the mean over seeds, while the R3 reward still rises.

## Multiple comparisons
E2 secondary corpora are corrected with Holm–Bonferroni over (corpus × metric) for the R3−R0 differences.

## Null-result commitment
If a decision rule is not met, the negative result is reported and analyzed; baselines are not weakened and
the primary metrics are not changed.

## Stopping rule
One campaign: the declared conditions × seeds, run once. Failed runs are re-run only for infrastructure failures
(OOM, node failure, preemption), logged in `results/failures/`.
