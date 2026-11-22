420 runs; seeds per stream, method and memory: 10 to 10.

### T1. Final accuracy (%), memory 200

| Method | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen | rounds |
|---|---|---|---|---|---|---|
| Fine-tuning | 66.38 ± 0.34 | 64.27 ± 0.50 | 57.42 ± 0.51 | 52.57 ± 0.32 | 68.00 ± 0.44 | 66.54 ± 0.56 |
| ER w=1 | 66.44 ± 0.33 | 65.41 ± 0.32 | 63.18 ± 0.32 | 61.79 ± 0.55 | 65.50 ± 0.48 | 66.50 ± 0.64 |
| PDCL | 66.37 ± 0.34 | 64.32 ± 0.62 | 61.65 ± 0.54 | 60.40 ± 0.42 | 68.22 ± 0.34 | 66.86 ± 0.56 |
| PDCL init 1 | 66.45 ± 0.30 | 64.18 ± 0.66 | 62.03 ± 0.61 | 60.50 ± 0.60 | 66.63 ± 0.53 | 66.69 ± 0.48 |

### T2. Final accuracy (%), memory 1000

| Method | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen | rounds |
|---|---|---|---|---|---|---|
| Fine-tuning | 66.38 ± 0.34 | 64.27 ± 0.50 | 57.42 ± 0.51 | 52.57 ± 0.32 | 68.00 ± 0.44 | 66.54 ± 0.56 |
| ER w=1 | 67.69 ± 0.41 | 67.19 ± 0.16 | 67.08 ± 0.27 | 66.47 ± 0.31 | 66.88 ± 0.34 | 68.72 ± 0.37 |
| PDCL | 67.82 ± 0.37 | 66.06 ± 0.45 | 66.19 ± 0.39 | 66.43 ± 0.32 | 68.36 ± 0.37 | 67.69 ± 0.41 |
| PDCL init 1 | 67.41 ± 0.38 | 65.75 ± 0.58 | 66.50 ± 0.42 | 66.57 ± 0.33 | 67.75 ± 0.28 | 67.33 ± 0.43 |

### T3. Final accuracy differences (paired by seed). "no gap" = ER w=1 is not above default PDCL by two standard errors

| Quantity | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen | rounds |
|---|---|---|---|---|---|---|
| init 1 − PDCL, memory 200 | +0.07 ± 0.20 | -0.14 ± 0.19 | **+0.38 ± 0.17** | +0.10 ± 0.35 | **-1.59 ± 0.35** | -0.17 ± 0.27 |
| init 1 − ER w=1, memory 200 | +0.01 ± 0.50 | **-1.23 ± 0.46** | **-1.15 ± 0.47** | **-1.28 ± 0.29** | **+1.13 ± 0.27** | +0.19 ± 0.38 |
| init 1 − Fine-tuning, memory 200 | +0.06 ± 0.28 | -0.09 ± 0.59 | **+4.61 ± 0.28** | **+7.93 ± 0.55** | **-1.37 ± 0.43** | +0.15 ± 0.27 |
| ER w=1 − PDCL (the gap), memory 200 | +0.07 ± 0.53 | **+1.09 ± 0.46** | **+1.54 ± 0.44** | **+1.38 ± 0.30** | **-2.72 ± 0.37** | -0.37 ± 0.35 |
| share of the gap closed, memory 200 | no gap | -13% | 25% | 7% | no gap | no gap |
| PDCL − Fine-tuning, memory 200 | -0.01 ± 0.17 | +0.04 ± 0.49 | **+4.23 ± 0.33** | **+7.83 ± 0.42** | +0.22 ± 0.24 | +0.32 ± 0.21 |
| ER w=1 − Fine-tuning, memory 200 | +0.05 ± 0.55 | **+1.14 ± 0.50** | **+5.76 ± 0.42** | **+9.21 ± 0.56** | **-2.50 ± 0.52** | -0.05 ± 0.45 |
| init 1 − PDCL, memory 1000 | -0.41 ± 0.26 | -0.31 ± 0.22 | +0.30 ± 0.20 | +0.15 ± 0.16 | **-0.61 ± 0.23** | -0.37 ± 0.20 |
| init 1 − ER w=1, memory 1000 | -0.28 ± 0.42 | **-1.44 ± 0.50** | **-0.59 ± 0.23** | +0.11 ± 0.21 | **+0.88 ± 0.27** | **-1.39 ± 0.45** |
| init 1 − Fine-tuning, memory 1000 | **+1.03 ± 0.37** | **+1.47 ± 0.49** | **+9.08 ± 0.54** | **+14.00 ± 0.39** | -0.25 ± 0.36 | **+0.78 ± 0.35** |
| ER w=1 − PDCL (the gap), memory 1000 | -0.13 ± 0.40 | **+1.13 ± 0.38** | **+0.89 ± 0.22** | +0.04 ± 0.20 | **-1.48 ± 0.28** | **+1.03 ± 0.43** |
| share of the gap closed, memory 1000 | no gap | -27% | 34% | no gap | no gap | -35% |
| PDCL − Fine-tuning, memory 1000 | **+1.44 ± 0.25** | **+1.78 ± 0.43** | **+8.77 ± 0.51** | **+13.86 ± 0.35** | +0.36 ± 0.23 | **+1.15 ± 0.21** |
| ER w=1 − Fine-tuning, memory 1000 | **+1.31 ± 0.47** | **+2.91 ± 0.44** | **+9.66 ± 0.45** | **+13.90 ± 0.44** | **-1.12 ± 0.39** | **+2.18 ± 0.48** |

### T4. Pre-registered predictions

| Cell | Prediction | Measured | Outcome |
|---|---|---|---|
| aglen, memory 200 | within 2 SE of fine-tuning | **-1.37 ± 0.43** | failed |
| aglen, memory 1000 | within 2 SE of fine-tuning | -0.25 ± 0.36 | held |
| rounds, memory 200 | within 2 SE of fine-tuning | +0.15 ± 0.27 | held |
| rounds, memory 1000 | within 2 SE of fine-tuning | **+0.78 ± 0.35** | failed |
| skew:0.9, memory 200 | closes less than half of ER w=1 − PDCL | **+0.38 ± 0.17** of **+1.54 ± 0.44** (25%) | held |
| skew:0.9, memory 1000 | closes less than half of ER w=1 − PDCL | +0.30 ± 0.20 of **+0.89 ± 0.22** (34%) | held |
| skew:0.95, memory 200 | closes less than half of ER w=1 − PDCL | +0.10 ± 0.35 of **+1.38 ± 0.30** (7%) | held |
| skew:0.95, memory 1000 | closes less than half of ER w=1 − PDCL | +0.15 ± 0.16 of +0.04 ± 0.20 (no gap) | not testable (no gap) |

### T5. Dual variables (mean over the checkpoints after each task's phase and over earlier tasks)

| Method | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen | rounds |
|---|---|---|---|---|---|---|
| PDCL, memory 200 | 0.051 ± 0.001 | 0.108 ± 0.002 | 0.243 ± 0.004 | 0.478 ± 0.011 | 0.003 ± 0.001 | 0.016 ± 0.002 |
| PDCL, memory 1000 | 0.160 ± 0.003 | 0.297 ± 0.004 | 0.558 ± 0.008 | 0.865 ± 0.017 | 0.003 ± 0.000 | 0.048 ± 0.003 |
| PDCL init 1, memory 200 | 0.083 ± 0.001 | 0.152 ± 0.002 | 0.336 ± 0.005 | 0.590 ± 0.013 | 0.045 ± 0.002 | 0.067 ± 0.002 |
| PDCL init 1, memory 1000 | 0.227 ± 0.004 | 0.379 ± 0.004 | 0.669 ± 0.011 | 0.993 ± 0.017 | 0.133 ± 0.003 | 0.163 ± 0.003 |

### T6. Dual variable of the first task over training (it becomes an earlier task after step 999; the checkpoint at step 1000 is one dual step later)

| Run | step 1000 | step 1250 | step 1500 | step 2000 | step 3000 | step 3999 |
|---|---|---|---|---|---|---|
| skew:0.9, memory 200, PDCL | 0.000 ± 0.000 | 1.075 ± 0.015 | 1.194 ± 0.024 | 0.734 ± 0.036 | 0.000 ± 0.000 | 0.018 ± 0.004 |
| skew:0.9, memory 200, PDCL init 1 | 1.000 ± 0.000 | 1.372 ± 0.018 | 1.303 ± 0.030 | 0.705 ± 0.043 | 0.000 ± 0.000 | 0.023 ± 0.003 |
| skew:0.9, memory 1000, PDCL | 0.000 ± 0.000 | 1.176 ± 0.012 | 1.562 ± 0.021 | 1.781 ± 0.039 | 0.118 ± 0.050 | 0.106 ± 0.016 |
| skew:0.9, memory 1000, PDCL init 1 | 1.000 ± 0.000 | 1.528 ± 0.014 | 1.779 ± 0.025 | 1.850 ± 0.047 | 0.152 ± 0.062 | 0.136 ± 0.021 |
| aglen, memory 200, PDCL | 0.000 ± 0.000 | 0.003 ± 0.002 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.003 ± 0.003 | 0.001 ± 0.001 |
| aglen, memory 200, PDCL init 1 | 0.999 ± 0.000 | 0.584 ± 0.019 | 0.032 ± 0.013 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| aglen, memory 1000, PDCL | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.001 | 0.002 ± 0.001 | 0.002 ± 0.001 | 0.004 ± 0.002 |
| aglen, memory 1000, PDCL init 1 | 0.999 ± 0.000 | 0.785 ± 0.013 | 0.485 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

### T7. Dual variables of tasks 1 / 2 / 3 at the last step

| Cell | PDCL | PDCL init 1 |
|---|---|---|
| skew:0.9, memory 200 | 0.018 / 0.000 / 0.064 | 0.023 / 0.000 / 0.087 |
| skew:0.9, memory 1000 | 0.106 / 0.000 / 0.472 | 0.136 / 0.000 / 0.551 |
| aglen, memory 200 | 0.001 / 0.000 / 0.003 | 0.000 / 0.000 / 0.000 |
| aglen, memory 1000 | 0.004 / 0.004 / 0.001 | 0.000 / 0.000 / 0.000 |

### T8. Context only: F1's oracle (constraints measured on validation data), from results/f1_runs.csv. These are other runs; differences are paired within F1

| Quantity | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen | rounds |
|---|---|---|---|---|---|---|
| F1: oracle − PDCL, memory 200 | -0.26 ± 0.52 | +1.04 ± 0.53 | **+1.27 ± 0.44** | **+2.35 ± 0.42** | +0.24 ± 0.24 | -0.59 ± 0.55 |
| F1: ER w=1 − PDCL, memory 200 | -0.49 ± 0.53 | **+1.21 ± 0.46** | **+1.24 ± 0.45** | **+1.67 ± 0.35** | **-2.60 ± 0.33** | -0.27 ± 0.40 |
| F1: share of that gap closed by the oracle, memory 200 | no gap | 86% | 102% | 140% | no gap | no gap |
| F1: oracle − Fine-tuning, memory 200 | +0.02 ± 0.59 | +1.03 ± 0.54 | **+5.45 ± 0.47** | **+9.72 ± 0.61** | +0.37 ± 0.31 | -0.27 ± 0.57 |
| H3: init 1 − PDCL, memory 200 | +0.07 ± 0.20 | -0.14 ± 0.19 | **+0.38 ± 0.17** | +0.10 ± 0.35 | **-1.59 ± 0.35** | -0.17 ± 0.27 |
| H3: share closed by init 1, memory 200 | no gap | -13% | 25% | 7% | no gap | no gap |
| F1: oracle − PDCL, memory 1000 | +0.15 ± 0.40 | **+1.18 ± 0.45** | +0.50 ± 0.31 | -0.26 ± 0.21 | -0.01 ± 0.28 | +0.34 ± 0.44 |
| F1: ER w=1 − PDCL, memory 1000 | +0.08 ± 0.33 | **+0.85 ± 0.32** | -0.00 ± 0.26 | -0.50 ± 0.33 | **-1.02 ± 0.41** | +0.79 ± 0.40 |
| F1: share of that gap closed by the oracle, memory 1000 | no gap | 138% | no gap | no gap | no gap | no gap |
| F1: oracle − Fine-tuning, memory 1000 | **+1.31 ± 0.35** | **+2.75 ± 0.47** | **+9.82 ± 0.40** | **+13.87 ± 0.36** | +0.36 ± 0.29 | **+1.30 ± 0.48** |
| H3: init 1 − PDCL, memory 1000 | -0.41 ± 0.26 | -0.31 ± 0.22 | +0.30 ± 0.20 | +0.15 ± 0.16 | **-0.61 ± 0.23** | -0.37 ± 0.20 |
| H3: share closed by init 1, memory 1000 | no gap | -27% | 34% | no gap | no gap | -35% |

### T9. Check: the rerun methods in H3 minus the same configuration and seeds in F1 (repeats on the shared GPUs are not bit-identical)

| Method | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen | rounds |
|---|---|---|---|---|---|---|
| Fine-tuning | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 |
| ER w=1, memory 200 | +0.26 ± 0.21 | -0.06 ± 0.18 | +0.34 ± 0.21 | +0.17 ± 0.20 | -0.04 ± 0.14 | -0.11 ± 0.28 |
| ER w=1, memory 1000 | +0.07 ± 0.24 | **+0.49 ± 0.19** | +0.34 ± 0.18 | +0.28 ± 0.27 | **-0.47 ± 0.23** | +0.42 ± 0.27 |
| PDCL, memory 200 | **-0.29 ± 0.13** | +0.06 ± 0.18 | +0.05 ± 0.23 | +0.46 ± 0.23 | +0.08 ± 0.20 | -0.01 ± 0.28 |
| PDCL, memory 1000 | +0.28 ± 0.16 | +0.21 ± 0.18 | **-0.55 ± 0.26** | -0.27 ± 0.15 | -0.01 ± 0.23 | +0.19 ± 0.17 |
