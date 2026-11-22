# F2 tables

Written by `python -m domainbed.scripts.f2_analysis` from the raw runs under `outputs/f2` (390 runs, ten seeds per setting). Mean ± standard error over seeds; differences are paired by seed; bold = larger than two standard errors.

### Pre-registered outcome (computed)

- `aglen`, memory 50: ER(32) − fine-tuning -4.03 ± 0.59, ER(2) − fine-tuning -4.05 ± 0.64, ER(2) − ER(32) -0.02 ± 0.22; share of the loss that remains 1.01 -> "removes at least half" **failed**.
- `aglen`, memory 200: ER(32) − fine-tuning -2.69 ± 0.46, ER(2) − fine-tuning -2.75 ± 0.52, ER(2) − ER(32) -0.07 ± 0.22; share of the loss that remains 1.03 -> "removes at least half" **failed**.
- `skew:0.9`, memory 50: ER(32) − fine-tuning +4.03 ± 0.32, ER(2) − fine-tuning +3.92 ± 0.33, ER(2) − ER(32) -0.11 ± 0.36; share of the gain kept 0.97 -> "keeps at least half" **held**.
- `skew:0.9`, memory 200: ER(32) − fine-tuning +5.62 ± 0.55, ER(2) − fine-tuning +5.99 ± 0.46, ER(2) − ER(32) +0.37 ± 0.38; share of the gain kept 1.07 -> "keeps at least half" **held**.
- Failure criterion, `aglen`, memory 50: ER − fine-tuning is -4.03, -3.99, -4.05 at replay batch 32, 8, 2; ER(2) − ER(32) = -0.02 ± 0.22 -> the loss **does not shrink** (two standard errors).
- Failure criterion, `aglen`, memory 200: ER − fine-tuning is -2.69, -2.73, -2.75 at replay batch 32, 8, 2; ER(2) − ER(32) = -0.07 ± 0.22 -> the loss **does not shrink** (two standard errors).

Prediction as a whole (all four parts): **failed**. Failure criterion (loss on `aglen` does not shrink): **met at both memory sizes**.

### T1. Final accuracy (%)

| Setting | aglen | rounds | skew:0.9 |
|---|---|---|---|
| Fine-tuning | 68.00 ± 0.44 | 66.54 ± 0.56 | 57.42 ± 0.51 |
| Fine-tuning, weight decay 0.01 | 66.88 ± 0.41 | 66.08 ± 0.42 | 51.86 ± 0.32 |
| Fine-tuning, weight decay 0.1 | 51.66 ± 0.39 | 50.23 ± 0.32 | 49.04 ± 0.16 |
| ER, memory 50, replay batch 32 | 63.97 ± 0.36 | 66.37 ± 0.39 | 61.45 ± 0.63 |
| ER, memory 50, replay batch 8 | 64.01 ± 0.44 | 66.40 ± 0.49 | 61.42 ± 0.53 |
| ER, memory 50, replay batch 2 | 63.95 ± 0.31 | 66.90 ± 0.51 | 61.34 ± 0.66 |
| ER, memory 200, replay batch 32 | 65.31 ± 0.39 | 66.30 ± 0.59 | 63.04 ± 0.46 |
| ER, memory 200, replay batch 8 | 65.27 ± 0.41 | 66.53 ± 0.44 | 63.33 ± 0.50 |
| ER, memory 200, replay batch 2 | 65.25 ± 0.44 | 66.95 ± 0.32 | 63.41 ± 0.49 |
| ER, memory 50, replay batch 32, weight decay 0.01 | 63.86 ± 0.37 | 66.54 ± 0.37 | 60.95 ± 0.60 |
| ER, memory 50, replay batch 32, weight decay 0.1 | 60.43 ± 0.58 | 55.17 ± 2.28 | 58.87 ± 0.98 |
| ER, memory 200, replay batch 32, weight decay 0.01 | 65.59 ± 0.32 | 66.57 ± 0.59 | 63.30 ± 0.41 |
| ER, memory 200, replay batch 32, weight decay 0.1 | 63.78 ± 0.42 | 51.62 ± 1.33 | 62.85 ± 0.59 |

### T2. Forgetting (points)

| Setting | aglen | rounds | skew:0.9 |
|---|---|---|---|
| Fine-tuning | 3.18 ± 0.55 | 3.80 ± 0.67 | 43.97 ± 0.78 |
| Fine-tuning, weight decay 0.01 | 3.81 ± 0.47 | 4.13 ± 0.46 | 51.30 ± 0.62 |
| Fine-tuning, weight decay 0.1 | 9.19 ± 1.26 | 9.61 ± 0.91 | 54.81 ± 0.42 |
| ER, memory 50, replay batch 32 | 7.49 ± 0.48 | 2.94 ± 0.34 | 33.04 ± 1.05 |
| ER, memory 50, replay batch 8 | 7.39 ± 0.57 | 2.98 ± 0.40 | 33.35 ± 0.90 |
| ER, memory 50, replay batch 2 | 7.44 ± 0.51 | 2.53 ± 0.56 | 33.47 ± 0.87 |
| ER, memory 200, replay batch 32 | 4.72 ± 0.33 | 3.02 ± 0.51 | 26.68 ± 0.71 |
| ER, memory 200, replay batch 8 | 5.31 ± 0.39 | 2.72 ± 0.57 | 25.68 ± 0.91 |
| ER, memory 200, replay batch 2 | 5.14 ± 0.54 | 1.90 ± 0.34 | 23.25 ± 0.94 |
| ER, memory 50, replay batch 32, weight decay 0.01 | 7.84 ± 0.46 | 2.64 ± 0.48 | 33.92 ± 0.92 |
| ER, memory 50, replay batch 32, weight decay 0.1 | 9.45 ± 0.70 | 3.82 ± 1.27 | 22.64 ± 2.23 |
| ER, memory 200, replay batch 32, weight decay 0.01 | 4.67 ± 0.33 | 2.64 ± 0.40 | 24.83 ± 0.82 |
| ER, memory 200, replay batch 32, weight decay 0.1 | 4.81 ± 0.47 | 4.75 ± 1.55 | 10.12 ± 2.31 |

### T3. Accuracy of each task at the end of its own phase (%)

| Setting | aglen | rounds | skew:0.9 |
|---|---|---|---|
| Fine-tuning | 70.23 ± 0.32 | 69.16 ± 0.30 | 90.39 ± 0.11 |
| Fine-tuning, weight decay 0.01 | 69.55 ± 0.36 | 69.09 ± 0.22 | 90.32 ± 0.19 |
| Fine-tuning, weight decay 0.1 | 58.55 ± 0.99 | 57.17 ± 0.77 | 90.15 ± 0.19 |
| ER, memory 50, replay batch 32 | 69.59 ± 0.28 | 68.18 ± 0.22 | 86.23 ± 0.25 |
| ER, memory 50, replay batch 8 | 69.55 ± 0.24 | 68.25 ± 0.31 | 86.43 ± 0.25 |
| ER, memory 50, replay batch 2 | 69.53 ± 0.24 | 68.49 ± 0.21 | 86.44 ± 0.24 |
| ER, memory 200, replay batch 32 | 68.82 ± 0.31 | 67.89 ± 0.20 | 83.05 ± 0.27 |
| ER, memory 200, replay batch 8 | 69.15 ± 0.32 | 68.02 ± 0.24 | 82.58 ± 0.38 |
| ER, memory 200, replay batch 2 | 69.04 ± 0.35 | 67.87 ± 0.30 | 80.84 ± 0.67 |
| ER, memory 50, replay batch 32, weight decay 0.01 | 69.72 ± 0.35 | 68.16 ± 0.30 | 86.39 ± 0.24 |
| ER, memory 50, replay batch 32, weight decay 0.1 | 67.42 ± 0.30 | 56.59 ± 1.71 | 73.63 ± 2.56 |
| ER, memory 200, replay batch 32, weight decay 0.01 | 68.99 ± 0.32 | 68.02 ± 0.30 | 81.92 ± 0.49 |
| ER, memory 200, replay batch 32, weight decay 0.1 | 67.17 ± 0.30 | 54.56 ± 1.34 | 66.06 ± 2.55 |

### T4. ER minus fine-tuning with the same weight decay, final accuracy (paired)

| Setting | aglen | rounds | skew:0.9 |
|---|---|---|---|
| ER, memory 50, replay batch 32 | **-4.03 ± 0.59** | -0.18 ± 0.46 | **+4.03 ± 0.32** |
| ER, memory 50, replay batch 8 | **-3.99 ± 0.71** | -0.15 ± 0.41 | **+4.00 ± 0.25** |
| ER, memory 50, replay batch 2 | **-4.05 ± 0.64** | +0.36 ± 0.37 | **+3.92 ± 0.33** |
| ER, memory 200, replay batch 32 | **-2.69 ± 0.46** | -0.25 ± 0.35 | **+5.62 ± 0.55** |
| ER, memory 200, replay batch 8 | **-2.73 ± 0.54** | -0.02 ± 0.37 | **+5.91 ± 0.53** |
| ER, memory 200, replay batch 2 | **-2.75 ± 0.52** | +0.41 ± 0.46 | **+5.99 ± 0.46** |
| ER, memory 50, replay batch 32, weight decay 0.01 | **-3.03 ± 0.59** | +0.45 ± 0.26 | **+9.09 ± 0.45** |
| ER, memory 50, replay batch 32, weight decay 0.1 | **+8.77 ± 0.85** | **+4.93 ± 2.24** | **+9.84 ± 0.88** |
| ER, memory 200, replay batch 32, weight decay 0.01 | **-1.29 ± 0.42** | +0.49 ± 0.42 | **+11.44 ± 0.51** |
| ER, memory 200, replay batch 32, weight decay 0.1 | **+12.12 ± 0.49** | +1.39 ± 1.45 | **+13.81 ± 0.63** |

### T5. ER minus fine-tuning with the same weight decay, forgetting (paired; negative = ER forgets less)

| Setting | aglen | rounds | skew:0.9 |
|---|---|---|---|
| ER, memory 50, replay batch 32 | **+4.31 ± 0.84** | -0.86 ± 0.73 | **-10.93 ± 0.66** |
| ER, memory 50, replay batch 8 | **+4.21 ± 0.99** | -0.81 ± 0.68 | **-10.62 ± 0.55** |
| ER, memory 50, replay batch 2 | **+4.26 ± 0.94** | -1.27 ± 0.72 | **-10.50 ± 0.51** |
| ER, memory 200, replay batch 32 | **+1.54 ± 0.56** | -0.77 ± 0.42 | **-17.29 ± 0.83** |
| ER, memory 200, replay batch 8 | **+2.13 ± 0.63** | **-1.07 ± 0.45** | **-18.29 ± 0.85** |
| ER, memory 200, replay batch 2 | **+1.97 ± 0.82** | **-1.90 ± 0.63** | **-20.72 ± 1.03** |
| ER, memory 50, replay batch 32, weight decay 0.01 | **+4.03 ± 0.80** | **-1.49 ± 0.48** | **-17.37 ± 0.76** |
| ER, memory 50, replay batch 32, weight decay 0.1 | +0.26 ± 1.30 | **-5.79 ± 1.07** | **-32.17 ± 2.12** |
| ER, memory 200, replay batch 32, weight decay 0.01 | +0.86 ± 0.61 | **-1.49 ± 0.32** | **-26.47 ± 0.98** |
| ER, memory 200, replay batch 32, weight decay 0.1 | **-4.39 ± 1.34** | **-4.86 ± 1.15** | **-44.69 ± 2.37** |

### T6. ER minus fine-tuning with the same weight decay, accuracy of each task at the end of its own phase (paired)

| Setting | aglen | rounds | skew:0.9 |
|---|---|---|---|
| ER, memory 50, replay batch 32 | **-0.65 ± 0.14** | **-0.98 ± 0.19** | **-4.16 ± 0.24** |
| ER, memory 50, replay batch 8 | **-0.68 ± 0.21** | **-0.91 ± 0.22** | **-3.95 ± 0.25** |
| ER, memory 50, replay batch 2 | **-0.70 ± 0.19** | **-0.67 ± 0.20** | **-3.94 ± 0.29** |
| ER, memory 200, replay batch 32 | **-1.41 ± 0.20** | **-1.27 ± 0.22** | **-7.34 ± 0.26** |
| ER, memory 200, replay batch 8 | **-1.08 ± 0.23** | **-1.14 ± 0.24** | **-7.80 ± 0.37** |
| ER, memory 200, replay batch 2 | **-1.19 ± 0.24** | **-1.29 ± 0.32** | **-9.54 ± 0.68** |
| ER, memory 50, replay batch 32, weight decay 0.01 | +0.17 ± 0.27 | **-0.93 ± 0.28** | **-3.93 ± 0.33** |
| ER, memory 50, replay batch 32, weight decay 0.1 | **+8.87 ± 1.13** | -0.59 ± 2.05 | **-16.52 ± 2.61** |
| ER, memory 200, replay batch 32, weight decay 0.01 | -0.57 ± 0.30 | **-1.07 ± 0.22** | **-8.40 ± 0.51** |
| ER, memory 200, replay batch 32, weight decay 0.1 | **+8.62 ± 1.08** | **-2.61 ± 1.29** | **-24.08 ± 2.56** |

### T7. The pre-registered test: ER minus fine-tuning at replay batch 32 and at a smaller replay batch

| Stream | Memory | Replay batch | ER(32) − fine-tuning | ER(batch) − fine-tuning | ER(batch) − ER(32) | share of the ER(32) effect that remains |
|---|---|---|---|---|---|---|
| aglen | 50 | 8 | **-4.03 ± 0.59** | **-3.99 ± 0.71** | +0.04 ± 0.22 | 0.99 (0.88 to 1.08) |
| aglen | 50 | 2 | **-4.03 ± 0.59** | **-4.05 ± 0.64** | -0.02 ± 0.22 | 1.01 (0.90 to 1.10) |
| aglen | 200 | 8 | **-2.69 ± 0.46** | **-2.73 ± 0.54** | -0.05 ± 0.28 | 1.02 (0.82 to 1.22) |
| aglen | 200 | 2 | **-2.69 ± 0.46** | **-2.75 ± 0.52** | -0.07 ± 0.22 | 1.03 (0.86 to 1.18) |
| rounds | 50 | 8 | -0.18 ± 0.46 | -0.15 ± 0.41 | +0.03 ± 0.22 | n/a (no effect at 32) |
| rounds | 50 | 2 | -0.18 ± 0.46 | +0.36 ± 0.37 | +0.53 ± 0.31 | n/a (no effect at 32) |
| rounds | 200 | 8 | -0.25 ± 0.35 | -0.02 ± 0.37 | +0.23 ± 0.25 | n/a (no effect at 32) |
| rounds | 200 | 2 | -0.25 ± 0.35 | +0.41 ± 0.46 | +0.66 ± 0.35 | n/a (no effect at 32) |
| skew:0.9 | 50 | 8 | **+4.03 ± 0.32** | **+4.00 ± 0.25** | -0.03 ± 0.19 | 0.99 (0.91 to 1.08) |
| skew:0.9 | 50 | 2 | **+4.03 ± 0.32** | **+3.92 ± 0.33** | -0.11 ± 0.36 | 0.97 (0.82 to 1.15) |
| skew:0.9 | 200 | 8 | **+5.62 ± 0.55** | **+5.91 ± 0.53** | +0.29 ± 0.32 | 1.05 (0.95 to 1.17) |
| skew:0.9 | 200 | 2 | **+5.62 ± 0.55** | **+5.99 ± 0.46** | +0.37 ± 0.38 | 1.07 (0.95 to 1.22) |

Share = ratio of the two mean differences; in brackets a 95% seed-bootstrap interval (exploratory). Prediction: on `aglen` the share at replay batch 2 is at most 0.5 at both memory sizes (at least half of ER's loss removed); on `skew:0.9` it is at least 0.5 (at least half of ER's gain kept).

### T8. Weight decay (replay batch 32), final accuracy, paired

| Stream | Weight decay | fine-tuning: decay − none | ER: decay − none, memory 50 | ER − fine-tuning, no decay, memory 50 | ER − fine-tuning, both with decay, memory 50 | change of ER − fine-tuning with decay, memory 50 | ER: decay − none, memory 200 | ER − fine-tuning, no decay, memory 200 | ER − fine-tuning, both with decay, memory 200 | change of ER − fine-tuning with decay, memory 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| aglen | 0.01 | **-1.12 ± 0.23** | -0.11 ± 0.25 | **-4.03 ± 0.59** | **-3.03 ± 0.59** | **+1.00 ± 0.23** | +0.28 ± 0.18 | **-2.69 ± 0.46** | **-1.29 ± 0.42** | **+1.39 ± 0.30** |
| aglen | 0.1 | **-16.34 ± 0.25** | **-3.54 ± 0.42** | **-4.03 ± 0.59** | **+8.77 ± 0.85** | **+12.80 ± 0.53** | **-1.54 ± 0.36** | **-2.69 ± 0.46** | **+12.12 ± 0.49** | **+14.80 ± 0.43** |
| rounds | 0.01 | -0.46 ± 0.27 | +0.17 ± 0.32 | -0.18 ± 0.46 | +0.45 ± 0.26 | +0.63 ± 0.48 | +0.27 ± 0.21 | -0.25 ± 0.35 | +0.49 ± 0.42 | **+0.74 ± 0.28** |
| rounds | 0.1 | **-16.31 ± 0.67** | **-11.20 ± 2.32** | -0.18 ± 0.46 | **+4.93 ± 2.24** | **+5.11 ± 2.33** | **-14.67 ± 1.41** | -0.25 ± 0.35 | +1.39 ± 1.45 | +1.64 ± 1.59 |
| skew:0.9 | 0.01 | **-5.56 ± 0.39** | **-0.50 ± 0.19** | **+4.03 ± 0.32** | **+9.09 ± 0.45** | **+5.06 ± 0.42** | +0.26 ± 0.35 | **+5.62 ± 0.55** | **+11.44 ± 0.51** | **+5.82 ± 0.63** |
| skew:0.9 | 0.1 | **-8.38 ± 0.40** | **-2.58 ± 0.79** | **+4.03 ± 0.32** | **+9.84 ± 0.88** | **+5.80 ± 0.91** | -0.19 ± 0.68 | **+5.62 ± 0.55** | **+13.81 ± 0.63** | **+8.19 ± 0.85** |

### T9. Loss on the stored examples against validation loss of the three earlier tasks, at the last step

| Stream | Setting | memory loss | validation loss | gap (validation − memory) | memory accuracy (%) | loss on all training examples of the earlier tasks |
|---|---|---|---|---|---|---|
| aglen | Fine-tuning | – | 0.598 ± 0.004 | – | – | 0.547 ± 0.002 |
| aglen | Fine-tuning, weight decay 0.01 | – | 0.595 ± 0.003 | – | – | 0.576 ± 0.003 |
| aglen | Fine-tuning, weight decay 0.1 | – | 0.693 ± 0.001 | – | – | 0.693 ± 0.001 |
| aglen | ER, memory 50, replay batch 32 | 0.003 ± 0.000 | 1.256 ± 0.051 | 1.253 ± 0.051 | 100.0 ± 0.0 | 1.146 ± 0.048 |
| aglen | ER, memory 50, replay batch 8 | 0.003 ± 0.000 | 1.256 ± 0.052 | 1.253 ± 0.052 | 100.0 ± 0.0 | 1.154 ± 0.050 |
| aglen | ER, memory 50, replay batch 2 | 0.005 ± 0.001 | 1.229 ± 0.047 | 1.224 ± 0.048 | 100.0 ± 0.0 | 1.140 ± 0.045 |
| aglen | ER, memory 200, replay batch 32 | 0.012 ± 0.001 | 1.120 ± 0.023 | 1.108 ± 0.023 | 100.0 ± 0.0 | 0.972 ± 0.017 |
| aglen | ER, memory 200, replay batch 8 | 0.016 ± 0.001 | 1.111 ± 0.024 | 1.095 ± 0.024 | 100.0 ± 0.0 | 0.977 ± 0.015 |
| aglen | ER, memory 200, replay batch 2 | 0.042 ± 0.003 | 0.999 ± 0.024 | 0.957 ± 0.025 | 99.4 ± 0.2 | 0.911 ± 0.016 |
| aglen | ER, memory 50, replay batch 32, weight decay 0.01 | 0.005 ± 0.000 | 1.143 ± 0.044 | 1.138 ± 0.043 | 100.0 ± 0.0 | 1.070 ± 0.042 |
| aglen | ER, memory 50, replay batch 32, weight decay 0.1 | 0.034 ± 0.005 | 1.156 ± 0.051 | 1.123 ± 0.053 | 99.8 ± 0.2 | 1.132 ± 0.046 |
| aglen | ER, memory 200, replay batch 32, weight decay 0.01 | 0.018 ± 0.001 | 1.035 ± 0.022 | 1.017 ± 0.022 | 100.0 ± 0.0 | 0.933 ± 0.017 |
| aglen | ER, memory 200, replay batch 32, weight decay 0.1 | 0.114 ± 0.008 | 1.043 ± 0.023 | 0.929 ± 0.024 | 97.4 ± 0.5 | 1.017 ± 0.015 |
| rounds | Fine-tuning | – | 0.633 ± 0.008 | – | – | 0.579 ± 0.009 |
| rounds | Fine-tuning, weight decay 0.01 | – | 0.599 ± 0.005 | – | – | 0.581 ± 0.006 |
| rounds | Fine-tuning, weight decay 0.1 | – | 0.693 ± 0.000 | – | – | 0.693 ± 0.000 |
| rounds | ER, memory 50, replay batch 32 | 0.005 ± 0.000 | 0.775 ± 0.021 | 0.770 ± 0.021 | 100.0 ± 0.0 | 0.658 ± 0.006 |
| rounds | ER, memory 50, replay batch 8 | 0.006 ± 0.000 | 0.775 ± 0.020 | 0.769 ± 0.020 | 100.0 ± 0.0 | 0.666 ± 0.007 |
| rounds | ER, memory 50, replay batch 2 | 0.008 ± 0.001 | 0.755 ± 0.019 | 0.746 ± 0.018 | 100.0 ± 0.0 | 0.675 ± 0.008 |
| rounds | ER, memory 200, replay batch 32 | 0.017 ± 0.002 | 0.833 ± 0.019 | 0.816 ± 0.019 | 99.9 ± 0.1 | 0.654 ± 0.010 |
| rounds | ER, memory 200, replay batch 8 | 0.021 ± 0.001 | 0.812 ± 0.017 | 0.791 ± 0.016 | 99.9 ± 0.1 | 0.662 ± 0.009 |
| rounds | ER, memory 200, replay batch 2 | 0.053 ± 0.002 | 0.766 ± 0.013 | 0.714 ± 0.013 | 99.8 ± 0.2 | 0.668 ± 0.006 |
| rounds | ER, memory 50, replay batch 32, weight decay 0.01 | 0.009 ± 0.000 | 0.737 ± 0.018 | 0.728 ± 0.018 | 100.0 ± 0.0 | 0.659 ± 0.007 |
| rounds | ER, memory 50, replay batch 32, weight decay 0.1 | 0.442 ± 0.101 | 0.745 ± 0.021 | 0.303 ± 0.122 | 72.1 ± 7.4 | 0.730 ± 0.016 |
| rounds | ER, memory 200, replay batch 32, weight decay 0.01 | 0.025 ± 0.002 | 0.796 ± 0.020 | 0.771 ± 0.019 | 100.0 ± 0.0 | 0.662 ± 0.009 |
| rounds | ER, memory 200, replay batch 32, weight decay 0.1 | 0.644 ± 0.049 | 0.707 ± 0.013 | 0.063 ± 0.063 | 55.9 ± 4.2 | 0.700 ± 0.007 |
| skew:0.9 | Fine-tuning | – | 1.328 ± 0.028 | – | – | 1.314 ± 0.022 |
| skew:0.9 | Fine-tuning, weight decay 0.01 | – | 1.291 ± 0.021 | – | – | 1.294 ± 0.020 |
| skew:0.9 | Fine-tuning, weight decay 0.1 | – | 1.425 ± 0.024 | – | – | 1.423 ± 0.023 |
| skew:0.9 | ER, memory 50, replay batch 32 | 0.004 ± 0.001 | 1.506 ± 0.062 | 1.502 ± 0.062 | 100.0 ± 0.0 | 1.420 ± 0.051 |
| skew:0.9 | ER, memory 50, replay batch 8 | 0.005 ± 0.001 | 1.479 ± 0.062 | 1.475 ± 0.062 | 100.0 ± 0.0 | 1.399 ± 0.050 |
| skew:0.9 | ER, memory 50, replay batch 2 | 0.008 ± 0.001 | 1.424 ± 0.057 | 1.417 ± 0.056 | 100.0 ± 0.0 | 1.363 ± 0.046 |
| skew:0.9 | ER, memory 200, replay batch 32 | 0.013 ± 0.001 | 1.443 ± 0.035 | 1.430 ± 0.034 | 100.0 ± 0.0 | 1.305 ± 0.028 |
| skew:0.9 | ER, memory 200, replay batch 8 | 0.018 ± 0.002 | 1.337 ± 0.046 | 1.319 ± 0.044 | 100.0 ± 0.0 | 1.224 ± 0.038 |
| skew:0.9 | ER, memory 200, replay batch 2 | 0.050 ± 0.006 | 1.134 ± 0.028 | 1.084 ± 0.024 | 99.7 ± 0.2 | 1.072 ± 0.026 |
| skew:0.9 | ER, memory 50, replay batch 32, weight decay 0.01 | 0.008 ± 0.001 | 1.365 ± 0.054 | 1.357 ± 0.053 | 100.0 ± 0.0 | 1.314 ± 0.045 |
| skew:0.9 | ER, memory 50, replay batch 32, weight decay 0.1 | 0.117 ± 0.021 | 1.170 ± 0.036 | 1.053 ± 0.048 | 97.0 ± 1.0 | 1.162 ± 0.036 |
| skew:0.9 | ER, memory 200, replay batch 32, weight decay 0.01 | 0.020 ± 0.002 | 1.241 ± 0.028 | 1.221 ± 0.027 | 100.0 ± 0.0 | 1.146 ± 0.022 |
| skew:0.9 | ER, memory 200, replay batch 32, weight decay 0.1 | 0.334 ± 0.032 | 0.846 ± 0.041 | 0.512 ± 0.072 | 85.7 ± 2.1 | 0.852 ± 0.038 |

### T10. The same, averaged over all checkpoints after each task's own phase

| Stream | Setting | memory loss | validation loss | gap (validation − memory) |
|---|---|---|---|---|
| aglen | Fine-tuning | – | 0.568 ± 0.004 | – |
| aglen | Fine-tuning, weight decay 0.01 | – | 0.566 ± 0.004 | – |
| aglen | Fine-tuning, weight decay 0.1 | – | 0.688 ± 0.001 | – |
| aglen | ER, memory 50, replay batch 32 | 0.063 ± 0.002 | 1.040 ± 0.042 | 0.977 ± 0.042 |
| aglen | ER, memory 50, replay batch 8 | 0.067 ± 0.002 | 1.034 ± 0.043 | 0.967 ± 0.042 |
| aglen | ER, memory 50, replay batch 2 | 0.079 ± 0.003 | 0.994 ± 0.039 | 0.915 ± 0.039 |
| aglen | ER, memory 200, replay batch 32 | 0.095 ± 0.001 | 0.897 ± 0.021 | 0.801 ± 0.021 |
| aglen | ER, memory 200, replay batch 8 | 0.112 ± 0.002 | 0.872 ± 0.020 | 0.759 ± 0.021 |
| aglen | ER, memory 200, replay batch 2 | 0.177 ± 0.004 | 0.786 ± 0.018 | 0.609 ± 0.019 |
| aglen | ER, memory 50, replay batch 32, weight decay 0.01 | 0.074 ± 0.002 | 0.975 ± 0.039 | 0.901 ± 0.038 |
| aglen | ER, memory 50, replay batch 32, weight decay 0.1 | 0.142 ± 0.005 | 1.001 ± 0.039 | 0.859 ± 0.040 |
| aglen | ER, memory 200, replay batch 32, weight decay 0.01 | 0.117 ± 0.002 | 0.842 ± 0.020 | 0.725 ± 0.021 |
| aglen | ER, memory 200, replay batch 32, weight decay 0.1 | 0.277 ± 0.006 | 0.830 ± 0.016 | 0.553 ± 0.018 |
| rounds | Fine-tuning | – | 0.651 ± 0.006 | – |
| rounds | Fine-tuning, weight decay 0.01 | – | 0.623 ± 0.005 | – |
| rounds | Fine-tuning, weight decay 0.1 | – | 0.694 ± 0.000 | – |
| rounds | ER, memory 50, replay batch 32 | 0.064 ± 0.002 | 0.810 ± 0.021 | 0.747 ± 0.021 |
| rounds | ER, memory 50, replay batch 8 | 0.068 ± 0.002 | 0.801 ± 0.021 | 0.733 ± 0.022 |
| rounds | ER, memory 50, replay batch 2 | 0.087 ± 0.003 | 0.772 ± 0.020 | 0.685 ± 0.020 |
| rounds | ER, memory 200, replay batch 32 | 0.111 ± 0.002 | 0.791 ± 0.019 | 0.680 ± 0.020 |
| rounds | ER, memory 200, replay batch 8 | 0.138 ± 0.003 | 0.760 ± 0.016 | 0.622 ± 0.017 |
| rounds | ER, memory 200, replay batch 2 | 0.215 ± 0.005 | 0.710 ± 0.013 | 0.495 ± 0.014 |
| rounds | ER, memory 50, replay batch 32, weight decay 0.01 | 0.077 ± 0.001 | 0.773 ± 0.019 | 0.696 ± 0.019 |
| rounds | ER, memory 50, replay batch 32, weight decay 0.1 | 0.502 ± 0.076 | 0.742 ± 0.020 | 0.240 ± 0.096 |
| rounds | ER, memory 200, replay batch 32, weight decay 0.01 | 0.138 ± 0.003 | 0.757 ± 0.017 | 0.618 ± 0.018 |
| rounds | ER, memory 200, replay batch 32, weight decay 0.1 | 0.662 ± 0.030 | 0.698 ± 0.005 | 0.036 ± 0.035 |
| skew:0.9 | Fine-tuning | – | 1.170 ± 0.008 | – |
| skew:0.9 | Fine-tuning, weight decay 0.01 | – | 1.189 ± 0.005 | – |
| skew:0.9 | Fine-tuning, weight decay 0.1 | – | 1.290 ± 0.008 | – |
| skew:0.9 | ER, memory 50, replay batch 32 | 0.042 ± 0.001 | 1.243 ± 0.025 | 1.201 ± 0.025 |
| skew:0.9 | ER, memory 50, replay batch 8 | 0.045 ± 0.001 | 1.221 ± 0.027 | 1.176 ± 0.027 |
| skew:0.9 | ER, memory 50, replay batch 2 | 0.062 ± 0.003 | 1.155 ± 0.023 | 1.094 ± 0.022 |
| skew:0.9 | ER, memory 200, replay batch 32 | 0.086 ± 0.002 | 1.077 ± 0.017 | 0.992 ± 0.016 |
| skew:0.9 | ER, memory 200, replay batch 8 | 0.105 ± 0.002 | 0.996 ± 0.016 | 0.891 ± 0.015 |
| skew:0.9 | ER, memory 200, replay batch 2 | 0.191 ± 0.006 | 0.853 ± 0.009 | 0.662 ± 0.011 |
| skew:0.9 | ER, memory 50, replay batch 32, weight decay 0.01 | 0.058 ± 0.002 | 1.141 ± 0.022 | 1.083 ± 0.022 |
| skew:0.9 | ER, memory 50, replay batch 32, weight decay 0.1 | 0.280 ± 0.013 | 0.954 ± 0.013 | 0.674 ± 0.021 |
| skew:0.9 | ER, memory 200, replay batch 32, weight decay 0.01 | 0.120 ± 0.002 | 0.960 ± 0.012 | 0.840 ± 0.011 |
| skew:0.9 | ER, memory 200, replay batch 32, weight decay 0.1 | 0.502 ± 0.020 | 0.730 ± 0.013 | 0.228 ± 0.031 |

### T11. How often a stored example is replayed (computed from the memory sizes in the logs)

| Memory | Replay batch | stored examples of task 0, phase 2 | stored examples of task 0, phase 3 | stored examples of task 0, phase 4 | stored examples in a step's batch (mean) | replays of a stored example per 1,000 steps | replays of a task-0 example kept to the end |
|---|---|---|---|---|---|---|---|
| 50 | 32 | 50 | 25 | 16 | 44.0 | 880 | 2640 |
| 50 | 8 | 50 | 25 | 16 | 16.0 | 320 | 980 |
| 50 | 2 | 50 | 25 | 16 | 4.0 | 80 | 245 |
| 200 | 32 | 200 | 100 | 66 | 64.0 | 320 | 965 |
| 200 | 8 | 200 | 100 | 66 | 16.0 | 80 | 241 |
| 200 | 2 | 200 | 100 | 66 | 4.0 | 20 | 60 |

Weight decay does not change these counts (same as replay batch 32). For comparison, a training example of the current task is drawn 15 times on `aglen`, 10 to 31 times on `rounds`, 15 to 16 times on `skew:0.9` during its own 1,000-step phase.

### T12. Final validation accuracy per task (%)

| Stream | Setting | task 0 | task 1 | task 2 | task 3 |
|---|---|---|---|---|---|
| aglen | Fine-tuning | 72.9 ± 0.6 | 64.2 ± 0.7 | 66.9 ± 1.3 | 68.1 ± 0.5 |
| aglen | Fine-tuning, weight decay 0.01 | 70.7 ± 0.4 | 63.6 ± 0.9 | 65.9 ± 1.1 | 67.4 ± 0.6 |
| aglen | Fine-tuning, weight decay 0.1 | 52.2 ± 0.9 | 48.3 ± 0.7 | 55.1 ± 1.9 | 51.0 ± 2.0 |
| aglen | ER, memory 50, replay batch 32 | 66.9 ± 1.1 | 59.8 ± 1.1 | 61.7 ± 1.0 | 67.5 ± 0.5 |
| aglen | ER, memory 50, replay batch 8 | 67.0 ± 1.2 | 60.0 ± 1.1 | 61.5 ± 1.3 | 67.6 ± 0.3 |
| aglen | ER, memory 50, replay batch 2 | 66.9 ± 1.0 | 60.1 ± 1.1 | 61.5 ± 1.3 | 67.3 ± 0.3 |
| aglen | ER, memory 200, replay batch 32 | 69.7 ± 0.8 | 61.3 ± 0.6 | 63.9 ± 1.0 | 66.3 ± 0.4 |
| aglen | ER, memory 200, replay batch 8 | 69.9 ± 0.7 | 61.1 ± 0.8 | 63.5 ± 1.0 | 66.5 ± 0.6 |
| aglen | ER, memory 200, replay batch 2 | 69.3 ± 0.9 | 61.4 ± 0.8 | 63.3 ± 1.1 | 67.0 ± 0.6 |
| aglen | ER, memory 50, replay batch 32, weight decay 0.01 | 67.0 ± 1.2 | 59.6 ± 0.9 | 61.0 ± 1.3 | 67.8 ± 0.5 |
| aglen | ER, memory 50, replay batch 32, weight decay 0.1 | 62.0 ± 1.4 | 55.6 ± 1.2 | 57.7 ± 1.2 | 66.3 ± 0.8 |
| aglen | ER, memory 200, replay batch 32, weight decay 0.01 | 69.6 ± 0.9 | 61.8 ± 0.7 | 63.8 ± 1.0 | 67.1 ± 0.5 |
| aglen | ER, memory 200, replay batch 32, weight decay 0.1 | 67.2 ± 0.9 | 60.8 ± 0.5 | 62.1 ± 0.8 | 65.1 ± 0.7 |
| rounds | Fine-tuning | 61.2 ± 0.9 | 67.6 ± 0.6 | 69.0 ± 1.1 | 68.3 ± 0.5 |
| rounds | Fine-tuning, weight decay 0.01 | 60.0 ± 0.9 | 67.2 ± 0.6 | 69.7 ± 0.8 | 67.4 ± 0.4 |
| rounds | Fine-tuning, weight decay 0.1 | 50.2 ± 0.7 | 46.0 ± 3.5 | 55.9 ± 3.5 | 48.8 ± 0.9 |
| rounds | ER, memory 50, replay batch 32 | 63.4 ± 1.1 | 66.5 ± 0.7 | 69.0 ± 0.8 | 66.5 ± 0.4 |
| rounds | ER, memory 50, replay batch 8 | 63.4 ± 1.3 | 66.3 ± 0.7 | 68.7 ± 0.7 | 67.2 ± 0.5 |
| rounds | ER, memory 50, replay batch 2 | 64.3 ± 1.2 | 66.7 ± 0.6 | 69.3 ± 0.9 | 67.3 ± 0.3 |
| rounds | ER, memory 200, replay batch 32 | 65.3 ± 1.6 | 65.3 ± 0.6 | 68.5 ± 0.9 | 66.1 ± 0.3 |
| rounds | ER, memory 200, replay batch 8 | 65.0 ± 1.5 | 65.9 ± 0.6 | 68.5 ± 0.9 | 66.7 ± 0.3 |
| rounds | ER, memory 200, replay batch 2 | 65.7 ± 1.3 | 65.8 ± 0.5 | 69.8 ± 0.5 | 66.5 ± 0.3 |
| rounds | ER, memory 50, replay batch 32, weight decay 0.01 | 63.4 ± 1.0 | 66.2 ± 0.6 | 69.5 ± 0.7 | 67.0 ± 0.4 |
| rounds | ER, memory 50, replay batch 32, weight decay 0.1 | 53.1 ± 2.0 | 55.6 ± 3.5 | 56.8 ± 4.5 | 55.1 ± 2.3 |
| rounds | ER, memory 200, replay batch 32, weight decay 0.01 | 65.3 ± 1.2 | 65.4 ± 0.7 | 68.8 ± 1.0 | 66.8 ± 0.5 |
| rounds | ER, memory 200, replay batch 32, weight decay 0.1 | 50.8 ± 1.4 | 52.8 ± 3.6 | 51.1 ± 4.3 | 51.8 ± 1.6 |
| skew:0.9 | Fine-tuning | 25.5 ± 1.1 | 89.9 ± 0.4 | 24.4 ± 1.0 | 89.8 ± 0.3 |
| skew:0.9 | Fine-tuning, weight decay 0.01 | 14.6 ± 0.6 | 90.0 ± 0.3 | 13.6 ± 0.8 | 89.2 ± 0.4 |
| skew:0.9 | Fine-tuning, weight decay 0.1 | 9.1 ± 0.4 | 89.5 ± 0.2 | 8.7 ± 0.5 | 88.9 ± 0.4 |
| skew:0.9 | ER, memory 50, replay batch 32 | 37.4 ± 1.4 | 85.8 ± 0.5 | 36.4 ± 1.7 | 86.3 ± 0.6 |
| skew:0.9 | ER, memory 50, replay batch 8 | 37.2 ± 1.3 | 86.2 ± 0.5 | 36.0 ± 1.5 | 86.3 ± 0.6 |
| skew:0.9 | ER, memory 50, replay batch 2 | 37.2 ± 1.5 | 86.3 ± 0.4 | 35.6 ± 1.7 | 86.2 ± 0.5 |
| skew:0.9 | ER, memory 200, replay batch 32 | 43.2 ± 1.2 | 83.2 ± 0.8 | 42.6 ± 1.2 | 83.2 ± 0.6 |
| skew:0.9 | ER, memory 200, replay batch 8 | 44.7 ± 1.5 | 82.2 ± 1.0 | 44.6 ± 1.8 | 81.8 ± 0.8 |
| skew:0.9 | ER, memory 200, replay batch 2 | 47.8 ± 1.4 | 79.6 ± 1.2 | 47.6 ± 1.3 | 78.6 ± 1.0 |
| skew:0.9 | ER, memory 50, replay batch 32, weight decay 0.01 | 36.2 ± 1.6 | 86.0 ± 0.6 | 35.6 ± 1.5 | 86.0 ± 0.6 |
| skew:0.9 | ER, memory 50, replay batch 32, weight decay 0.1 | 41.8 ± 2.0 | 76.5 ± 1.6 | 41.1 ± 2.2 | 76.0 ± 1.3 |
| skew:0.9 | ER, memory 200, replay batch 32, weight decay 0.01 | 45.4 ± 1.1 | 81.7 ± 1.0 | 44.7 ± 1.4 | 81.4 ± 0.7 |
| skew:0.9 | ER, memory 200, replay batch 32, weight decay 0.1 | 58.7 ± 1.9 | 66.2 ± 1.9 | 59.9 ± 2.1 | 66.5 ± 1.9 |

### T13 (exploratory). Across the six settings (memory x replay batch): correlation of log replays per stored example with the mean effect

| Stream | ER − fine-tuning, final accuracy | ER − fine-tuning, forgetting | gap validation − memory loss |
|---|---|---|---|
| aglen | -0.48 | +0.45 | +0.81 |
| rounds | -0.86 | +0.90 | +0.91 |
| skew:0.9 | -0.56 | +0.63 | +0.86 |

Six points per stream, means over ten seeds; a description, not a test.
