# E1 under both ceiling references

tau=0.3; panel margins {'estoi': 0.02, 'sisdr': 1.0, 'wer': 0.0, 'sbs': 0.02}; mean over seeds; mean-action candidate only.

## Predictor's own reconstruction offset R(x_hat)-R(x)

| target          |   grpo_dac |   grpo_encodec |   pgd_wave |   random_encodec |   transfer_r0_round1 |
|:----------------|-----------:|---------------:|-----------:|-----------------:|---------------------:|
| R0              |      0.072 |         -0.095 |          0 |           -0.095 |               -0.095 |
| R1              |      0.074 |         -0.064 |          0 |           -0.064 |               -0.064 |
| R2              |      0.081 |          0.001 |          0 |            0.001 |                0.001 |
| R3              |      0.081 |         -0.266 |          0 |           -0.266 |               -0.266 |
| R3ms            |      0.04  |          0.949 |          0 |            0.949 |                0.949 |
| baseline_noise  |      0.069 |         -0.085 |          0 |           -0.085 |               -0.085 |
| baseline_pgd    |      0.082 |         -0.077 |          0 |           -0.077 |               -0.077 |
| baseline_random |      0.09  |         -0.03  |          0 |           -0.03  |               -0.03  |

## ASR, ceiling referenced to raw input x (as run)

| target          |   grpo_dac |   grpo_encodec |   pgd_wave |   random_encodec |   transfer_r0_round1 |
|:----------------|-----------:|---------------:|-----------:|-----------------:|---------------------:|
| R0              |      0.12  |          0.181 |      0.999 |            0.012 |                0.251 |
| R1              |      0.301 |          0.067 |      1     |            0.001 |                0     |
| R2              |      0.304 |          0.079 |      1     |            0     |                0     |
| R3              |      0.293 |          0.05  |      1     |            0     |                0     |
| R3ms            |      0.037 |          0.306 |      0.841 |            0.022 |                0     |
| baseline_noise  |      0.356 |          0.235 |      1     |            0.013 |                0.217 |
| baseline_pgd    |      0.352 |          0.21  |      0.969 |            0.01  |                0.181 |
| baseline_random |      0.35  |          0.053 |      1     |            0     |                0.002 |

## ASR, ceiling referenced to codec reconstruction x_hat (corrected)

| target          |   grpo_dac |   grpo_encodec |   pgd_wave |   random_encodec |   transfer_r0_round1 |
|:----------------|-----------:|---------------:|-----------:|-----------------:|---------------------:|
| R0              |      0.065 |          0.226 |      0.999 |            0.001 |                0.318 |
| R1              |      0.161 |          0.05  |      1     |            0     |                0     |
| R2              |      0.141 |          0.017 |      1     |            0     |                0     |
| R3              |      0.127 |          0.137 |      1     |            0.002 |                0     |
| R3ms            |      0.016 |          0.082 |      0.841 |            0     |                0     |
| baseline_noise  |      0.23  |          0.27  |      1     |            0     |                0.246 |
| baseline_pgd    |      0.225 |          0.223 |      0.969 |            0     |                0.18  |
| baseline_random |      0.2   |          0.019 |      1     |            0     |                0     |

## Attacker gain over x_hat

| target          |   grpo_dac |   grpo_encodec |   pgd_wave |   random_encodec |   transfer_r0_round1 |
|:----------------|-----------:|---------------:|-----------:|-----------------:|---------------------:|
| R0              |     -0.051 |          0.136 |      1.786 |           -0.344 |                0.245 |
| R1              |      0.142 |          0.138 |      1.935 |           -0.913 |               -1.56  |
| R2              |      0.134 |          0.03  |      2.003 |           -1.303 |               -1.743 |
| R3              |      0.122 |          0.216 |      2.102 |           -1.614 |               -2.491 |
| R3ms            |      0.048 |         -0.455 |      1.227 |           -2.713 |               -3.51  |
| baseline_noise  |      0.171 |          0.223 |      2.04  |           -0.388 |                0.191 |
| baseline_pgd    |      0.176 |          0.205 |      1.618 |           -0.684 |                0.078 |
| baseline_random |      0.156 |          0.102 |      2.055 |           -1.887 |               -0.959 |
