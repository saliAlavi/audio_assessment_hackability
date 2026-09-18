H1: ASR(R0)-ASR(R3) per seed = [0.143, 0.005, 0.247], mean 0.131; supported = True (rule: mean ≥ 0.10 and all seeds > 0)

H1-transfer [grpo_dac]: ASR drop per seed [-0.274, 0.072, -0.317], mean -0.173; transfers = False

H1-transfer [pgd_wave]: ASR drop per seed [-0.003, 0.0, 0.0], mean -0.001; transfers = False

H1-transfer [random_encodec]: ASR drop per seed [0.006, 0.018, 0.012], mean 0.012; transfers = False

H1-transfer [transfer_r0_round1]: ASR drop per seed [0.213, 0.233, 0.308], mean 0.251; transfers = True

H2: BVCC-test sys-SRCC R3-R0 per seed = [-0.0012, 0.0018, -0.006], mean -0.0018; non-inferior = True (rule: mean ≥ -0.02)

H2 secondary: mean sys-SRCC over the 13 eval sets R0 0.745 vs R3 0.659; R3 significantly worse (Holm p<0.05) on 1/13 eval sets: vmc23/track2

H3: degraded external measures (mean over seeds) R0-reward 2.33 vs R3-reward 2.33; reward rise +0.263 / +0.269; supported = False