# H1, long training (240 runs from outputs/h1: 5,000 steps per task, 10 seeds per cell)

Final accuracy = validation accuracy at the last step, mean of the four tasks, in percent. Mean ± standard error over seeds; differences are paired by seed; bold: beyond two standard errors.

## Memory methods and joint ERM against fine-tuning, 5,000 steps per task

| Stream | FT, final accuracy | ER 0.03 − FT | ER 0.3 − FT | ER 1 − FT | PDCL − FT | Joint − FT |
|---|---|---|---|---|---|---|
| rounds | 65.59 ± 0.43 | +0.57 ± 0.35 | **+0.65 ± 0.32** | **+0.95 ± 0.43** | **+0.93 ± 0.35** | **+3.58 ± 0.34** |
| rounds:3210 | 66.49 ± 0.39 | +0.13 ± 0.33 | +0.63 ± 0.33 | +0.35 ± 0.38 | +0.21 ± 0.31 | **+2.51 ± 0.44** |
| aglen | 62.97 ± 0.39 | **+3.23 ± 0.47** | **+3.11 ± 0.56** | **+2.76 ± 0.48** | **+2.32 ± 0.34** | **+5.41 ± 0.47** |
| target | 62.19 ± 0.51 | **+3.36 ± 0.64** | **+3.68 ± 0.63** | **+3.18 ± 0.47** | **+2.96 ± 0.47** | **+6.08 ± 0.57** |

## PDCL against replay with a fixed weight, 5,000 steps per task

Best weight = the weight with the highest mean final accuracy on that stream, picked on the same seeds (favours replay).

| Stream | PDCL − ER 0.03 | PDCL − ER 0.3 | PDCL − ER 1 | Best weight | PDCL − best weight |
|---|---|---|---|---|---|
| rounds | +0.35 ± 0.29 | +0.27 ± 0.36 | -0.03 ± 0.37 | w = 1 | -0.03 ± 0.37 |
| rounds:3210 | +0.08 ± 0.32 | -0.41 ± 0.32 | -0.13 ± 0.27 | w = 0.3 | -0.41 ± 0.32 |
| aglen | **-0.91 ± 0.45** | -0.79 ± 0.41 | -0.44 ± 0.27 | w = 0.03 | **-0.91 ± 0.45** |
| target | -0.40 ± 0.38 | **-0.72 ± 0.29** | -0.23 ± 0.24 | w = 0.3 | **-0.72 ± 0.29** |

## Secondary: 5,000 against 1,000 steps per task, same method, same seeds

"not run": that replay weight was not run at 1,000 steps per task on that stream.

| Stream | Method | 1,000 steps per task | 5,000 steps per task | 5,000 − 1,000 | Source of the 1,000 column |
|---|---|---|---|---|---|
| rounds | Joint ERM | 70.10 ± 0.28 | 69.17 ± 0.30 | **-0.93 ± 0.28** | `results/e1_runs_main.csv` |
| rounds | Fine-tuning | 66.54 ± 0.56 | 65.59 ± 0.43 | -0.95 ± 0.69 | `results/e1_runs_main.csv` |
| rounds | ER, w = 0.03 | 67.31 ± 0.49 | 66.17 ± 0.40 | -1.14 ± 0.60 | `results/e2_runs_stream.csv` |
| rounds | ER, w = 0.3 | 67.07 ± 0.44 | 66.25 ± 0.46 | -0.83 ± 0.57 | `results/e2_runs_stream.csv` |
| rounds | ER, w = 1 | 66.85 ± 0.44 | 66.55 ± 0.35 | -0.30 ± 0.53 | `results/e1_runs_main.csv` |
| rounds | PDCL | 66.62 ± 0.47 | 66.52 ± 0.42 | -0.10 ± 0.54 | `results/e1_runs_main.csv` |
| rounds:3210 | Joint ERM | 70.01 ± 0.36 | 69.01 ± 0.25 | **-1.00 ± 0.33** | `results/e1_runs_main.csv` |
| rounds:3210 | Fine-tuning | 67.38 ± 0.31 | 66.49 ± 0.39 | **-0.89 ± 0.36** | `results/e1_runs_main.csv` |
| rounds:3210 | ER, w = 0.03 | not run | 66.63 ± 0.40 |  |  |
| rounds:3210 | ER, w = 0.3 | not run | 67.12 ± 0.27 |  |  |
| rounds:3210 | ER, w = 1 | 66.44 ± 0.24 | 66.84 ± 0.48 | +0.40 ± 0.47 | `results/e1_runs_main.csv` |
| rounds:3210 | PDCL | 67.37 ± 0.31 | 66.71 ± 0.36 | **-0.66 ± 0.27** | `results/e1_runs_main.csv` |
| aglen | Joint ERM | 69.83 ± 0.25 | 68.39 ± 0.29 | **-1.45 ± 0.38** | `results/e1_runs_main.csv` |
| aglen | Fine-tuning | 68.00 ± 0.44 | 62.97 ± 0.39 | **-5.03 ± 0.43** | `results/e1_runs_main.csv` |
| aglen | ER, w = 0.03 | 67.31 ± 0.36 | 66.20 ± 0.50 | **-1.12 ± 0.47** | `results/e2_runs_stream.csv` |
| aglen | ER, w = 0.3 | 65.71 ± 0.41 | 66.08 ± 0.40 | +0.37 ± 0.41 | `results/e2_runs_stream.csv` |
| aglen | ER, w = 1 | 65.52 ± 0.33 | 65.73 ± 0.38 | +0.21 ± 0.36 | `results/e1_runs_main.csv` |
| aglen | PDCL | 68.24 ± 0.41 | 65.29 ± 0.38 | **-2.94 ± 0.29** | `results/e1_runs_main.csv` |
| target | Joint ERM | 69.51 ± 0.32 | 68.27 ± 0.50 | **-1.24 ± 0.36** | `results/e1_runs_main.csv` |
| target | Fine-tuning | 67.48 ± 0.60 | 62.19 ± 0.51 | **-5.29 ± 0.48** | `results/e1_runs_main.csv` |
| target | ER, w = 0.03 | not run | 65.55 ± 0.50 |  |  |
| target | ER, w = 0.3 | not run | 65.87 ± 0.41 |  |  |
| target | ER, w = 1 | 65.85 ± 0.37 | 65.38 ± 0.39 | -0.48 ± 0.44 | `results/e1_runs_main.csv` |
| target | PDCL | 68.32 ± 0.45 | 65.15 ± 0.49 | **-3.17 ± 0.35** | `results/e1_runs_main.csv` |

## Gain over fine-tuning at both lengths

| Stream | Method | Method − FT at 1,000 | Method − FT at 5,000 |
|---|---|---|---|
| rounds | ER, w = 0.03 | **+0.76 ± 0.21** | +0.57 ± 0.35 |
| rounds | ER, w = 0.3 | +0.53 ± 0.47 | **+0.65 ± 0.32** |
| rounds | ER, w = 1 | +0.30 ± 0.35 | **+0.95 ± 0.43** |
| rounds | PDCL | +0.08 ± 0.24 | **+0.93 ± 0.35** |
| rounds:3210 | ER, w = 0.03 | not run | +0.13 ± 0.33 |
| rounds:3210 | ER, w = 0.3 | not run | +0.63 ± 0.33 |
| rounds:3210 | ER, w = 1 | **-0.94 ± 0.32** | +0.35 ± 0.38 |
| rounds:3210 | PDCL | -0.01 ± 0.17 | +0.21 ± 0.31 |
| aglen | ER, w = 0.03 | -0.69 ± 0.40 | **+3.23 ± 0.47** |
| aglen | ER, w = 0.3 | **-2.29 ± 0.48** | **+3.11 ± 0.56** |
| aglen | ER, w = 1 | **-2.48 ± 0.49** | **+2.76 ± 0.48** |
| aglen | PDCL | +0.24 ± 0.15 | **+2.32 ± 0.34** |
| target | ER, w = 0.03 | not run | **+3.36 ± 0.64** |
| target | ER, w = 0.3 | not run | **+3.68 ± 0.63** |
| target | ER, w = 1 | **-1.62 ± 0.60** | **+3.18 ± 0.47** |
| target | PDCL | **+0.84 ± 0.37** | **+2.96 ± 0.47** |

## Over-training: highest checkpoint of the seed-mean curve against the last step

Validation accuracy, mean of the four tasks, checkpoints every 500 steps. For the sequential methods early checkpoints precede some tasks, so only a peak in the last phase (after step 15,000) says that training went on too long.

| Stream | Method | Highest checkpoint | at step | Last step | Last − highest (paired) |
|---|---|---|---|---|---|
| rounds | Joint ERM | 70.25 | 5,500 | 69.17 | **-1.08 ± 0.24** |
| rounds | Fine-tuning | 65.81 | 19,500 | 65.59 | -0.22 ± 0.20 |
| rounds | ER, w = 0.03 | 66.64 | 17,500 | 66.17 | **-0.47 ± 0.23** |
| rounds | ER, w = 0.3 | 66.66 | 18,000 | 66.25 | -0.42 ± 0.21 |
| rounds | ER, w = 1 | 66.68 | 17,000 | 66.55 | -0.13 ± 0.34 |
| rounds | PDCL | 66.70 | 17,500 | 66.52 | -0.18 ± 0.16 |
| rounds:3210 | Joint ERM | 70.23 | 3,000 | 69.01 | **-1.22 ± 0.38** |
| rounds:3210 | Fine-tuning | 66.95 | 16,500 | 66.49 | -0.46 ± 0.29 |
| rounds:3210 | ER, w = 0.03 | 67.14 | 16,500 | 66.63 | **-0.52 ± 0.22** |
| rounds:3210 | ER, w = 0.3 | 67.25 | 16,500 | 67.12 | -0.13 ± 0.16 |
| rounds:3210 | ER, w = 1 | 67.14 | 18,500 | 66.84 | -0.30 ± 0.21 |
| rounds:3210 | PDCL | 67.39 | 17,000 | 66.71 | **-0.68 ± 0.30** |
| aglen | Joint ERM | 69.94 | 2,500 | 68.39 | **-1.56 ± 0.29** |
| aglen | Fine-tuning | 64.58 | 16,000 | 62.97 | **-1.61 ± 0.27** |
| aglen | ER, w = 0.03 | 66.62 | 17,000 | 66.20 | -0.42 ± 0.21 |
| aglen | ER, w = 0.3 | 66.84 | 15,500 | 66.08 | **-0.76 ± 0.26** |
| aglen | ER, w = 1 | 66.35 | 16,000 | 65.73 | **-0.62 ± 0.10** |
| aglen | PDCL | 66.25 | 16,000 | 65.29 | **-0.96 ± 0.21** |
| target | Joint ERM | 69.85 | 2,500 | 68.27 | **-1.58 ± 0.36** |
| target | Fine-tuning | 63.86 | 16,000 | 62.19 | **-1.66 ± 0.53** |
| target | ER, w = 0.03 | 66.19 | 16,500 | 65.55 | **-0.64 ± 0.26** |
| target | ER, w = 0.3 | 66.08 | 19,000 | 65.87 | -0.21 ± 0.12 |
| target | ER, w = 1 | 65.90 | 17,000 | 65.38 | **-0.52 ± 0.13** |
| target | PDCL | 66.17 | 16,000 | 65.15 | **-1.02 ± 0.20** |

## Methods, 5,000 steps per task

| Stream | Method | Final accuracy | Forgetting | Learned accuracy |
|---|---|---|---|---|
| rounds | Joint ERM | 69.17 ± 0.30 | 0.83 ± 0.18 | 69.30 ± 0.23 |
| rounds | Fine-tuning | 65.59 ± 0.43 | 2.97 ± 0.45 | 67.74 ± 0.35 |
| rounds | ER, w = 0.03 | 66.17 ± 0.40 | 3.28 ± 0.45 | 68.30 ± 0.28 |
| rounds | ER, w = 0.3 | 66.25 ± 0.46 | 2.69 ± 0.40 | 67.92 ± 0.39 |
| rounds | ER, w = 1 | 66.55 ± 0.35 | 2.20 ± 0.63 | 67.91 ± 0.43 |
| rounds | PDCL | 66.52 ± 0.42 | 2.71 ± 0.52 | 68.22 ± 0.27 |
| rounds:3210 | Joint ERM | 69.01 ± 0.25 | 0.75 ± 0.22 | 69.27 ± 0.32 |
| rounds:3210 | Fine-tuning | 66.49 ± 0.39 | 4.18 ± 0.39 | 69.48 ± 0.30 |
| rounds:3210 | ER, w = 0.03 | 66.63 ± 0.40 | 3.88 ± 0.42 | 69.36 ± 0.22 |
| rounds:3210 | ER, w = 0.3 | 67.12 ± 0.27 | 3.36 ± 0.43 | 69.34 ± 0.19 |
| rounds:3210 | ER, w = 1 | 66.84 ± 0.48 | 3.46 ± 0.46 | 69.12 ± 0.29 |
| rounds:3210 | PDCL | 66.71 ± 0.36 | 3.53 ± 0.42 | 69.23 ± 0.20 |
| aglen | Joint ERM | 68.39 ± 0.29 | 1.05 ± 0.22 | 68.85 ± 0.24 |
| aglen | Fine-tuning | 62.97 ± 0.39 | 7.57 ± 0.66 | 68.39 ± 0.42 |
| aglen | ER, w = 0.03 | 66.20 ± 0.50 | 3.46 ± 0.68 | 68.52 ± 0.43 |
| aglen | ER, w = 0.3 | 66.08 ± 0.40 | 3.24 ± 0.62 | 68.16 ± 0.43 |
| aglen | ER, w = 1 | 65.73 ± 0.38 | 3.82 ± 0.40 | 68.20 ± 0.42 |
| aglen | PDCL | 65.29 ± 0.38 | 4.12 ± 0.42 | 68.14 ± 0.38 |
| target | Joint ERM | 68.27 ± 0.50 | 1.26 ± 0.27 | 68.97 ± 0.47 |
| target | Fine-tuning | 62.19 ± 0.51 | 9.21 ± 0.60 | 69.10 ± 0.35 |
| target | ER, w = 0.03 | 65.55 ± 0.50 | 4.97 ± 0.43 | 69.17 ± 0.39 |
| target | ER, w = 0.3 | 65.87 ± 0.41 | 4.12 ± 0.47 | 68.94 ± 0.47 |
| target | ER, w = 1 | 65.38 ± 0.39 | 5.24 ± 0.45 | 69.21 ± 0.40 |
| target | PDCL | 65.15 ± 0.49 | 5.30 ± 0.39 | 69.06 ± 0.45 |
