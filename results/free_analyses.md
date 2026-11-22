# Analyses that need no new runs

Written by `python -m domainbed.scripts.free_analyses`. Mean ± standard error over seeds; differences paired by seed, bold above two standard errors; accuracies in percent. Final accuracy = mean over tasks of the last checkpoint's validation accuracy; forgetting as in `stream_analysis.run_row`. A–C answer scope questions; G1–G3 are exploratory and were not pre-registered.

Runs used (whole-stream runs with a `done` file):

| Root | Path | Finished runs | Run folders | Streams | Steps | Seeds |
|---|---|---|---|---|---|---|
| E1 | `pdcl-e1-retention-dial-6f00e4/outputs/e1` | 440 | 440 | 11 | 4000 | 10 |
| E1 long | `pdcl-e1-retention-dial-6f00e4/outputs/e1_long` | 180 | 185 | 9 | 20000 | 5 |
| E2 | `pdcl-e2-adaptivity-09393c/outputs/e2` | 900 | 900 | 5 | 4000 | 10 |
| E3 | `pdcl-e3-allocation-3e05ed/outputs/e3` | 400 | 400 | 12 | 4000, 8000 | 10 |
| F1 | `pdcl-f1-honest-constraints-2145b2/outputs/f1` | 900 | 900 | 6 | 4000 | 10 |
| F2 | `pdcl-f2-replay-memorization-3869d5/outputs/f2` | 390 | 390 | 3 | 4000 | 10 |
| F3 | `pdcl-f3-tolerable-drift-0389fa/outputs/f3` | 200 | 200 | 5 | 4000 | 10 |

## A. Two-by-two: what makes fine-tuning forget (E1 main, 4,000 steps)

| Antigens | Binder rate | Stream | Fine-tuning forgetting | Fine-tuning final acc. | Joint final acc. | Joint − fine-tuning | Seeds |
|---|---|---|---|---|---|---|---|
| shared | stable | skew:0.5 | -0.30 ± 0.42 | 68.15 ± 0.36 | 69.42 ± 0.27 | **+1.28 ± 0.30** | 10 |
| shared | stable | rounds (real) | 3.80 ± 0.67 | 66.54 ± 0.56 | 70.10 ± 0.28 | **+3.56 ± 0.50** | 10 |
| shared | shifting | skew:0.9 | 43.97 ± 0.78 | 57.42 ± 0.51 | 69.84 ± 0.18 | **+12.42 ± 0.44** | 10 |
| new | stable | aglen | 3.18 ± 0.55 | 68.00 ± 0.44 | 69.83 ± 0.25 | **+1.83 ± 0.49** | 10 |
| new | stable | target | 3.44 ± 0.60 | 67.48 ± 0.60 | 69.51 ± 0.32 | **+2.03 ± 0.39** | 10 |
| new | shifting | agrate | 22.15 ± 0.62 | 54.05 ± 0.23 | 69.89 ± 0.23 | **+15.83 ± 0.42** | 10 |

## B. Where the real rounds sit on the binder-rate dial (E1 main + F3)

Swing = highest minus lowest binder rate among the tasks: 2Q − 1 for skew:Q, D for drift:D, 0.20 for the real rounds (binder rates 0.49, 0.59, 0.39, 0.53). Differences are in final accuracy.

| Swing | Stream | Dial | Root | Fine-tuning forgetting | ER − fine-tuning | PDCL − fine-tuning | Seeds |
|---|---|---|---|---|---|---|---|
| 0.00 | skew:0.5 | alternating | E1 | -0.30 ± 0.42 | **-0.96 ± 0.41** | **-0.17 ± 0.08** | 10 |
| 0.20 | rounds (real) | real rounds | E1 | 3.80 ± 0.67 | +0.30 ± 0.35 | +0.08 ± 0.24 | 10 |
| 0.20 | rounds, reversed | real rounds | E1 | 3.21 ± 0.29 | **-0.94 ± 0.32** | -0.01 ± 0.17 | 10 |
| 0.20 | skew:0.6 | alternating | E1 | 2.87 ± 0.59 | -0.79 ± 0.46 | -0.01 ± 0.15 | 10 |
| 0.40 | skew:0.7 | alternating | E1 | 10.16 ± 0.58 | +0.06 ± 0.58 | +0.17 ± 0.17 | 10 |
| 0.40 | drift:0.4 | gradual | F3 | 9.50 ± 0.90 | **+1.46 ± 0.60** | **+0.64 ± 0.24** | 10 |
| 0.50 | skew:0.75 | alternating | F3 | 16.83 ± 0.91 | +0.87 ± 0.61 | **+0.89 ± 0.33** | 10 |
| 0.60 | skew:0.8 | alternating | E1 | 22.38 ± 0.63 | +0.95 ± 0.55 | +0.02 ± 0.45 | 10 |
| 0.60 | drift:0.6 | gradual | F3 | 19.59 ± 0.81 | **+2.36 ± 0.56** | **+1.58 ± 0.27** | 10 |
| 0.70 | skew:0.85 | alternating | F3 | 34.05 ± 0.81 | **+3.31 ± 0.69** | **+2.48 ± 0.44** | 10 |
| 0.80 | skew:0.9 | alternating | E1 | 43.97 ± 0.78 | **+5.44 ± 0.42** | **+4.07 ± 0.34** | 10 |
| 0.80 | drift:0.8 | gradual | F3 | 30.75 ± 0.74 | **+3.94 ± 0.57** | **+2.98 ± 0.50** | 10 |
| 0.90 | skew:0.95 | alternating | E1 | 56.51 ± 0.56 | **+8.89 ± 0.47** | **+7.43 ± 0.43** | 10 |

## C. PDCL minus fine-tuning, every cell

A cell is one stream × PDCL variant × memory size × length inside one root, compared with the fine-tuning runs of the same root, stream, length and seeds (final accuracy, paired). `default` = the published rule (the dual variables read the loss of the replayed memory), including the cells that only change memory size, allocation α, dual step size or ε scale; `val` and `split` are the F1 variants. Nothing is corrected for the number of cells.

| PDCL | Cells | > 2 SE below fine-tuning | > 2 SE above | Within 2 SE | Lowest difference |
|---|---|---|---|---|---|
| default | 101 | 2 | 51 | 48 | E1 long skew:0.6 PDCL_RS (memory 200, 20000 steps): -0.57 ± 0.28 |
| val | 12 | 0 | 7 | 5 | F1 rounds (real) PDCL_RS__val_buf200 (memory 200, 4000 steps): -0.27 ± 0.57 |
| split | 12 | 1 | 5 | 6 | F1 aglen PDCL_RS__split_buf200 (memory 200, 4000 steps): -1.98 ± 0.56 |

Cells more than two standard errors below fine-tuning, `default`: 

- E1 long skew:0.6 PDCL_RS (memory 200, 20000 steps): -0.57 ± 0.28
- E1 skew:0.5 PDCL_RS (memory 200, 4000 steps): -0.17 ± 0.08

Cells more than two standard errors below fine-tuning, `val`: none.


Cells more than two standard errors below fine-tuning, `split`: 

- F1 aglen PDCL_RS__split_buf200 (memory 200, 4000 steps): -1.98 ± 0.56

| PDCL | Root | Cells | Below | Above | Lowest | Median | Highest |
|---|---|---|---|---|---|---|---|
| default | E1 | 11 | 1 | 4 | -0.17 | +0.17 | +7.43 |
| default | E1 long | 9 | 1 | 5 | -0.57 | +1.55 | +4.56 |
| default | E2 | 40 | 0 | 10 | -0.22 | +0.17 | +14.30 |
| default | E3 | 24 | 0 | 19 | +0.18 | +1.13 | +6.11 |
| default | F1 | 12 | 0 | 8 | -0.01 | +1.06 | +14.12 |
| default | F3 | 5 | 0 | 5 | +0.64 | +1.58 | +2.98 |
| split | F1 | 12 | 1 | 5 | -1.98 | +0.54 | +12.11 |
| val | F1 | 12 | 0 | 7 | -0.27 | +1.30 | +13.87 |

All cells:

| PDCL | Root | Stream | Method | Memory | Steps | PDCL − fine-tuning | Seeds |
|---|---|---|---|---|---|---|---|
| default | E1 | aglen | `PDCL_RS` | 200 | 4000 | +0.24 ± 0.15 | 10 |
| default | E1 | agrate | `PDCL_RS` | 200 | 4000 | **+5.65 ± 0.33** | 10 |
| default | E1 | rounds (real) | `PDCL_RS` | 200 | 4000 | +0.08 ± 0.24 | 10 |
| default | E1 | rounds, reversed | `PDCL_RS` | 200 | 4000 | -0.01 ± 0.17 | 10 |
| default | E1 | skew:0.5 | `PDCL_RS` | 200 | 4000 | **-0.17 ± 0.08** | 10 |
| default | E1 | skew:0.6 | `PDCL_RS` | 200 | 4000 | -0.01 ± 0.15 | 10 |
| default | E1 | skew:0.7 | `PDCL_RS` | 200 | 4000 | +0.17 ± 0.17 | 10 |
| default | E1 | skew:0.8 | `PDCL_RS` | 200 | 4000 | +0.02 ± 0.45 | 10 |
| default | E1 | skew:0.9 | `PDCL_RS` | 200 | 4000 | **+4.07 ± 0.34** | 10 |
| default | E1 | skew:0.95 | `PDCL_RS` | 200 | 4000 | **+7.43 ± 0.43** | 10 |
| default | E1 | target | `PDCL_RS` | 200 | 4000 | **+0.84 ± 0.37** | 10 |
| default | E1 long | aglen | `PDCL_RS` | 200 | 20000 | **+1.91 ± 0.75** | 5 |
| default | E1 long | agrate | `PDCL_RS` | 200 | 20000 | **+4.56 ± 0.70** | 5 |
| default | E1 long | rounds (real) | `PDCL_RS` | 200 | 20000 | +0.76 ± 0.40 | 5 |
| default | E1 long | rounds, reversed | `PDCL_RS` | 200 | 20000 | **+1.55 ± 0.29** | 5 |
| default | E1 long | skew:0.5 | `PDCL_RS` | 200 | 20000 | +0.00 ± 0.26 | 5 |
| default | E1 long | skew:0.6 | `PDCL_RS` | 200 | 20000 | **-0.57 ± 0.28** | 5 |
| default | E1 long | skew:0.7 | `PDCL_RS` | 200 | 20000 | +0.36 ± 0.48 | 5 |
| default | E1 long | skew:0.8 | `PDCL_RS` | 200 | 20000 | **+1.77 ± 0.81** | 5 |
| default | E1 long | target | `PDCL_RS` | 200 | 20000 | **+1.74 ± 0.65** | 5 |
| default | E2 | aglen | `PDCL_RS` | 200 | 4000 | +0.30 ± 0.28 | 10 |
| default | E2 | aglen | `PDCL_RS__buf1000` | 1000 | 4000 | +0.06 ± 0.24 | 10 |
| default | E2 | aglen | `PDCL_RS__buf50` | 50 | 4000 | -0.12 ± 0.23 | 10 |
| default | E2 | aglen | `PDCL_RS__dlr0.001` | 200 | 4000 | +0.09 ± 0.18 | 10 |
| default | E2 | aglen | `PDCL_RS__dlr0.1` | 200 | 4000 | +0.36 ± 0.27 | 10 |
| default | E2 | aglen | `PDCL_RS__es1.0` | 200 | 4000 | +0.04 ± 0.31 | 10 |
| default | E2 | aglen | `PDCL_RS__es1.5` | 200 | 4000 | -0.22 ± 0.21 | 10 |
| default | E2 | aglen | `PDCL_RS__es2.0` | 200 | 4000 | -0.16 ± 0.17 | 10 |
| default | E2 | rounds (real) | `PDCL_RS` | 200 | 4000 | +0.30 ± 0.24 | 10 |
| default | E2 | rounds (real) | `PDCL_RS__buf1000` | 1000 | 4000 | **+1.01 ± 0.33** | 10 |
| default | E2 | rounds (real) | `PDCL_RS__buf50` | 50 | 4000 | +0.22 ± 0.21 | 10 |
| default | E2 | rounds (real) | `PDCL_RS__dlr0.001` | 200 | 4000 | +0.10 ± 0.22 | 10 |
| default | E2 | rounds (real) | `PDCL_RS__dlr0.1` | 200 | 4000 | +0.40 ± 0.28 | 10 |
| default | E2 | rounds (real) | `PDCL_RS__es1.0` | 200 | 4000 | +0.24 ± 0.29 | 10 |
| default | E2 | rounds (real) | `PDCL_RS__es1.5` | 200 | 4000 | -0.02 ± 0.16 | 10 |
| default | E2 | rounds (real) | `PDCL_RS__es2.0` | 200 | 4000 | +0.12 ± 0.18 | 10 |
| default | E2 | skew:0.6 | `PDCL_RS` | 200 | 4000 | -0.12 ± 0.13 | 10 |
| default | E2 | skew:0.6 | `PDCL_RS__buf1000` | 1000 | 4000 | -0.10 ± 0.20 | 10 |
| default | E2 | skew:0.6 | `PDCL_RS__buf50` | 50 | 4000 | -0.05 ± 0.16 | 10 |
| default | E2 | skew:0.6 | `PDCL_RS__dlr0.001` | 200 | 4000 | -0.08 ± 0.23 | 10 |
| default | E2 | skew:0.6 | `PDCL_RS__dlr0.1` | 200 | 4000 | -0.08 ± 0.23 | 10 |
| default | E2 | skew:0.6 | `PDCL_RS__es1.0` | 200 | 4000 | -0.07 ± 0.19 | 10 |
| default | E2 | skew:0.6 | `PDCL_RS__es1.5` | 200 | 4000 | -0.17 ± 0.11 | 10 |
| default | E2 | skew:0.6 | `PDCL_RS__es2.0` | 200 | 4000 | -0.15 ± 0.13 | 10 |
| default | E2 | skew:0.8 | `PDCL_RS` | 200 | 4000 | +0.18 ± 0.50 | 10 |
| default | E2 | skew:0.8 | `PDCL_RS__buf1000` | 1000 | 4000 | **+1.69 ± 0.44** | 10 |
| default | E2 | skew:0.8 | `PDCL_RS__buf50` | 50 | 4000 | +0.10 ± 0.13 | 10 |
| default | E2 | skew:0.8 | `PDCL_RS__dlr0.001` | 200 | 4000 | +0.16 ± 0.53 | 10 |
| default | E2 | skew:0.8 | `PDCL_RS__dlr0.1` | 200 | 4000 | +0.29 ± 0.43 | 10 |
| default | E2 | skew:0.8 | `PDCL_RS__es1.0` | 200 | 4000 | -0.08 ± 0.45 | 10 |
| default | E2 | skew:0.8 | `PDCL_RS__es1.5` | 200 | 4000 | +0.38 ± 0.37 | 10 |
| default | E2 | skew:0.8 | `PDCL_RS__es2.0` | 200 | 4000 | +0.18 ± 0.16 | 10 |
| default | E2 | skew:0.95 | `PDCL_RS` | 200 | 4000 | **+7.20 ± 0.47** | 10 |
| default | E2 | skew:0.95 | `PDCL_RS__buf1000` | 1000 | 4000 | **+14.30 ± 0.40** | 10 |
| default | E2 | skew:0.95 | `PDCL_RS__buf50` | 50 | 4000 | **+3.40 ± 0.44** | 10 |
| default | E2 | skew:0.95 | `PDCL_RS__dlr0.001` | 200 | 4000 | **+8.50 ± 0.56** | 10 |
| default | E2 | skew:0.95 | `PDCL_RS__dlr0.1` | 200 | 4000 | **+7.17 ± 0.61** | 10 |
| default | E2 | skew:0.95 | `PDCL_RS__es1.0` | 200 | 4000 | **+7.73 ± 0.41** | 10 |
| default | E2 | skew:0.95 | `PDCL_RS__es1.5` | 200 | 4000 | **+6.70 ± 0.45** | 10 |
| default | E2 | skew:0.95 | `PDCL_RS__es2.0` | 200 | 4000 | **+5.93 ± 0.59** | 10 |
| default | E3 | agrate_m200 | `PDCL_RS__a0` | 200 | 4000 | **+6.11 ± 0.29** | 10 |
| default | E3 | agrate_m200 | `PDCL_RS__a0.5` | 200 | 4000 | **+5.82 ± 0.42** | 10 |
| default | E3 | agrate_m200 | `PDCL_RS__a1` | 200 | 4000 | **+4.28 ± 0.31** | 10 |
| default | E3 | agrate_m40 | `PDCL_RS__a0` | 40 | 4000 | **+1.80 ± 0.17** | 10 |
| default | E3 | agrate_m40 | `PDCL_RS__a0.5` | 40 | 4000 | **+1.73 ± 0.12** | 10 |
| default | E3 | agrate_m40 | `PDCL_RS__a1` | 40 | 4000 | **+1.35 ± 0.18** | 10 |
| default | E3 | hetero_m200 | `PDCL_RS__a0` | 200 | 4000 | **+2.78 ± 0.33** | 10 |
| default | E3 | hetero_m200 | `PDCL_RS__a0.5` | 200 | 4000 | **+2.93 ± 0.32** | 10 |
| default | E3 | hetero_m200 | `PDCL_RS__a1` | 200 | 4000 | **+3.23 ± 0.54** | 10 |
| default | E3 | hetero_m40 | `PDCL_RS__a0` | 40 | 4000 | **+0.91 ± 0.23** | 10 |
| default | E3 | hetero_m40 | `PDCL_RS__a0.5` | 40 | 4000 | **+0.80 ± 0.21** | 10 |
| default | E3 | hetero_m40 | `PDCL_RS__a1` | 40 | 4000 | **+0.53 ± 0.26** | 10 |
| default | E3 | rounds_m200 | `PDCL_RS__a0` | 200 | 4000 | **+0.43 ± 0.20** | 10 |
| default | E3 | rounds_m200 | `PDCL_RS__a0.5` | 200 | 4000 | +0.42 ± 0.22 | 10 |
| default | E3 | rounds_m200 | `PDCL_RS__a1` | 200 | 4000 | +0.31 ± 0.18 | 10 |
| default | E3 | rounds_m40 | `PDCL_RS__a0` | 40 | 4000 | +0.36 ± 0.19 | 10 |
| default | E3 | rounds_m40 | `PDCL_RS__a0.5` | 40 | 4000 | +0.30 ± 0.17 | 10 |
| default | E3 | rounds_m40 | `PDCL_RS__a1` | 40 | 4000 | +0.18 ± 0.19 | 10 |
| default | E3 | skew:0.9x8_m200 | `PDCL_RS__a0` | 200 | 8000 | **+2.75 ± 0.27** | 10 |
| default | E3 | skew:0.9x8_m200 | `PDCL_RS__a0.5` | 200 | 8000 | **+3.40 ± 0.31** | 10 |
| default | E3 | skew:0.9x8_m200 | `PDCL_RS__a1` | 200 | 8000 | **+3.99 ± 0.40** | 10 |
| default | E3 | skew:0.9x8_m40 | `PDCL_RS__a0` | 40 | 8000 | **+0.58 ± 0.17** | 10 |
| default | E3 | skew:0.9x8_m40 | `PDCL_RS__a0.5` | 40 | 8000 | **+0.64 ± 0.28** | 10 |
| default | E3 | skew:0.9x8_m40 | `PDCL_RS__a1` | 40 | 8000 | **+0.85 ± 0.12** | 10 |
| default | F1 | aglen | `PDCL_RS__memory_buf1000` | 1000 | 4000 | **+0.37 ± 0.15** | 10 |
| default | F1 | aglen | `PDCL_RS__memory_buf200` | 200 | 4000 | +0.13 ± 0.18 | 10 |
| default | F1 | rounds (real) | `PDCL_RS__memory_buf1000` | 1000 | 4000 | **+0.96 ± 0.24** | 10 |
| default | F1 | rounds (real) | `PDCL_RS__memory_buf200` | 200 | 4000 | +0.33 ± 0.28 | 10 |
| default | F1 | skew:0.7 | `PDCL_RS__memory_buf1000` | 1000 | 4000 | **+1.16 ± 0.30** | 10 |
| default | F1 | skew:0.7 | `PDCL_RS__memory_buf200` | 200 | 4000 | +0.28 ± 0.20 | 10 |
| default | F1 | skew:0.8 | `PDCL_RS__memory_buf1000` | 1000 | 4000 | **+1.57 ± 0.49** | 10 |
| default | F1 | skew:0.8 | `PDCL_RS__memory_buf200` | 200 | 4000 | -0.01 ± 0.45 | 10 |
| default | F1 | skew:0.9 | `PDCL_RS__memory_buf1000` | 1000 | 4000 | **+9.32 ± 0.58** | 10 |
| default | F1 | skew:0.9 | `PDCL_RS__memory_buf200` | 200 | 4000 | **+4.18 ± 0.31** | 10 |
| default | F1 | skew:0.95 | `PDCL_RS__memory_buf1000` | 1000 | 4000 | **+14.12 ± 0.39** | 10 |
| default | F1 | skew:0.95 | `PDCL_RS__memory_buf200` | 200 | 4000 | **+7.37 ± 0.39** | 10 |
| default | F3 | drift:0.4 | `PDCL_RS` | 200 | 4000 | **+0.64 ± 0.24** | 10 |
| default | F3 | drift:0.6 | `PDCL_RS` | 200 | 4000 | **+1.58 ± 0.27** | 10 |
| default | F3 | drift:0.8 | `PDCL_RS` | 200 | 4000 | **+2.98 ± 0.50** | 10 |
| default | F3 | skew:0.75 | `PDCL_RS` | 200 | 4000 | **+0.89 ± 0.33** | 10 |
| default | F3 | skew:0.85 | `PDCL_RS` | 200 | 4000 | **+2.48 ± 0.44** | 10 |
| split | F1 | aglen | `PDCL_RS__split_buf1000` | 1000 | 4000 | -0.15 ± 0.37 | 10 |
| split | F1 | aglen | `PDCL_RS__split_buf200` | 200 | 4000 | **-1.98 ± 0.56** | 10 |
| split | F1 | rounds (real) | `PDCL_RS__split_buf1000` | 1000 | 4000 | +0.69 ± 0.47 | 10 |
| split | F1 | rounds (real) | `PDCL_RS__split_buf200` | 200 | 4000 | -0.19 ± 0.49 | 10 |
| split | F1 | skew:0.7 | `PDCL_RS__split_buf1000` | 1000 | 4000 | +0.40 ± 0.38 | 10 |
| split | F1 | skew:0.7 | `PDCL_RS__split_buf200` | 200 | 4000 | -0.32 ± 0.27 | 10 |
| split | F1 | skew:0.8 | `PDCL_RS__split_buf1000` | 1000 | 4000 | **+1.82 ± 0.47** | 10 |
| split | F1 | skew:0.8 | `PDCL_RS__split_buf200` | 200 | 4000 | +0.30 ± 0.53 | 10 |
| split | F1 | skew:0.9 | `PDCL_RS__split_buf1000` | 1000 | 4000 | **+7.87 ± 0.49** | 10 |
| split | F1 | skew:0.9 | `PDCL_RS__split_buf200` | 200 | 4000 | **+5.48 ± 0.40** | 10 |
| split | F1 | skew:0.95 | `PDCL_RS__split_buf1000` | 1000 | 4000 | **+12.11 ± 0.44** | 10 |
| split | F1 | skew:0.95 | `PDCL_RS__split_buf200` | 200 | 4000 | **+7.53 ± 0.34** | 10 |
| val | F1 | aglen | `PDCL_RS__val_buf1000` | 1000 | 4000 | +0.36 ± 0.29 | 10 |
| val | F1 | aglen | `PDCL_RS__val_buf200` | 200 | 4000 | +0.37 ± 0.31 | 10 |
| val | F1 | rounds (real) | `PDCL_RS__val_buf1000` | 1000 | 4000 | **+1.30 ± 0.48** | 10 |
| val | F1 | rounds (real) | `PDCL_RS__val_buf200` | 200 | 4000 | -0.27 ± 0.57 | 10 |
| val | F1 | skew:0.7 | `PDCL_RS__val_buf1000` | 1000 | 4000 | **+1.31 ± 0.35** | 10 |
| val | F1 | skew:0.7 | `PDCL_RS__val_buf200` | 200 | 4000 | +0.02 ± 0.59 | 10 |
| val | F1 | skew:0.8 | `PDCL_RS__val_buf1000` | 1000 | 4000 | **+2.75 ± 0.47** | 10 |
| val | F1 | skew:0.8 | `PDCL_RS__val_buf200` | 200 | 4000 | +1.03 ± 0.54 | 10 |
| val | F1 | skew:0.9 | `PDCL_RS__val_buf1000` | 1000 | 4000 | **+9.82 ± 0.40** | 10 |
| val | F1 | skew:0.9 | `PDCL_RS__val_buf200` | 200 | 4000 | **+5.45 ± 0.47** | 10 |
| val | F1 | skew:0.95 | `PDCL_RS__val_buf1000` | 1000 | 4000 | **+13.87 ± 0.36** | 10 |
| val | F1 | skew:0.95 | `PDCL_RS__val_buf200` | 200 | 4000 | **+9.72 ± 0.61** | 10 |

## G1 (exploratory). Last task against earlier tasks at the end of training

Earlier = mean final validation accuracy of tasks 0–2; last = final validation accuracy of task 3. The ER curve joins fine-tuning (replay weight 0) and the fixed replay weights in order of weight, all at memory 200. The gap is PDCL's last-task accuracy minus the curve, interpolated linearly at PDCL's earlier-task accuracy; its standard error is a bootstrap over seeds (2000 resamples, the same seeds drawn for every method), and `inside` is the share of resamples in which PDCL fell inside the curve's range. Above/below = more than two standard errors. `Dominated by` lists the replay weights whose mean is at least as high as PDCL's on both axes. F1 has no joint-training runs; its joint row is taken from E1 main (same stream and seeds).

| Root | Stream | PDCL | Earlier acc. | Last acc. | Last − ER curve | Inside | Verdict | Dominated by ER weight |
|---|---|---|---|---|---|---|---|---|
| E2 | rounds (real) | PDCL | 66.37 ± 0.70 | 68.25 ± 0.38 | +0.24 ± 0.37 | 1.00 | on | none |
| E2 | skew:0.6 | PDCL | 66.90 ± 0.47 | 70.70 ± 0.70 | +0.46 ± 0.74 | 0.90 | on | none |
| E2 | skew:0.8 | PDCL | 59.42 ± 0.99 | 79.57 ± 0.61 | -0.78 ± 0.41 | 0.82 | on | 0.03 |
| E2 | skew:0.95 | PDCL | 49.52 ± 0.64 | 90.55 ± 0.45 | **-1.15 ± 0.26** | 1.00 | below | 0.1 |
| E2 | aglen | PDCL | 68.24 ± 0.42 | 68.48 ± 0.45 | +0.01 ± 0.21 | 0.65 | on | none |
| F1 | rounds (real) | PDCL | 66.38 ± 0.61 | 68.34 ± 0.46 | +0.12 ± 0.40 | 0.98 | on | none |
| F1 | rounds (real) | PDCL val | 66.04 ± 0.96 | 66.97 ± 0.33 | **-1.34 ± 0.64** | 0.63 | below | 0.03, 0.3 |
| F1 | rounds (real) | PDCL split | 66.09 ± 0.49 | 67.14 ± 0.34 | -1.17 ± 0.69 | 0.72 | on | 0.03, 0.3 |
| F1 | skew:0.7 | PDCL | 64.15 ± 0.50 | 74.23 ± 0.54 | -0.22 ± 0.29 | 0.99 | on | 0.03 |
| F1 | skew:0.7 | PDCL val | 64.95 ± 0.52 | 70.75 ± 0.64 | – | 0.50 | outside the curve (PDCL keeps more earlier-task accuracy than any point of it; against its nearest end, ER weight 0.3: earlier +0.08 ± 0.21, last **-1.77 ± 0.63**) | none |
| F1 | skew:0.7 | PDCL split | 64.17 ± 0.18 | 71.73 ± 0.42 | **-2.72 ± 1.35** | 0.99 | below | 0.03, 0.3 |
| F1 | skew:0.8 | PDCL | 59.21 ± 0.93 | 79.42 ± 0.57 | **-1.06 ± 0.36** | 0.80 | below | 0.03 |
| F1 | skew:0.8 | PDCL val | 62.61 ± 0.42 | 73.38 ± 0.67 | -0.71 ± 0.44 | 0.65 | on | 3 |
| F1 | skew:0.8 | PDCL split | 60.58 ± 0.62 | 76.56 ± 0.47 | -1.48 ± 0.94 | 0.99 | on | none |
| F1 | skew:0.9 | PDCL | 53.19 ± 0.80 | 86.84 ± 0.31 | -0.76 ± 0.48 | 1.00 | on | none |
| F1 | skew:0.9 | PDCL val | 56.86 ± 0.51 | 80.90 ± 0.82 | – | 0.44 | outside the curve (PDCL keeps more earlier-task accuracy than any point of it; against its nearest end, ER weight 3: earlier +0.09 ± 0.24, last -1.08 ± 0.59) | none |
| F1 | skew:0.9 | PDCL split | 55.93 ± 0.63 | 83.80 ± 0.45 | -0.77 ± 0.90 | 0.98 | on | 0.3 |
| F1 | skew:0.95 | PDCL | 49.76 ± 0.60 | 90.51 ± 0.51 | +0.26 ± 0.25 | 1.00 | on | none |
| F1 | skew:0.95 | PDCL val | 54.55 ± 0.83 | 85.52 ± 0.57 | – | 0.00 | outside the curve (PDCL keeps more earlier-task accuracy than any point of it; against its nearest end, ER weight 3: earlier **+1.31 ± 0.45**, last **-1.03 ± 0.34**) | none |
| F1 | skew:0.95 | PDCL split | 50.63 ± 0.60 | 88.52 ± 0.39 | **-1.12 ± 0.25** | 1.00 | below | none |
| F1 | aglen | PDCL | 68.01 ± 0.48 | 68.52 ± 0.52 | – | 0.42 | outside the curve (PDCL keeps more earlier-task accuracy than any point of it; against its nearest end, fine-tuning: earlier +0.05 ± 0.25, last +0.39 ± 0.30) | none |
| F1 | aglen | PDCL val | 68.25 ± 0.42 | 68.72 ± 0.46 | – | 0.23 | outside the curve (PDCL keeps more earlier-task accuracy than any point of it; against its nearest end, fine-tuning: earlier +0.29 ± 0.42, last +0.60 ± 0.32) | none |
| F1 | aglen | PDCL split | 65.48 ± 0.75 | 67.63 ± 0.50 | -0.16 ± 0.44 | 0.85 | on | 0.03 |

Count of streams by verdict:

| Root | PDCL | below | on | outside the curve |
|---|---|---|---|---|
| E2 | PDCL | 1 | 4 | 0 |
| F1 | PDCL | 1 | 4 | 1 |
| F1 | PDCL split | 2 | 4 | 0 |
| F1 | PDCL val | 1 | 1 | 4 |

E2, memory 200: earlier-task accuracy / last-task accuracy

| Method | rounds (real) | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| `Finetune` | 65.9 ± 0.7 / 68.3 ± 0.5 | 67.0 ± 0.4 / 71.0 ± 0.6 | 58.7 ± 0.7 / 81.0 ± 0.4 | 38.7 ± 0.4 / 94.3 ± 0.4 | 68.0 ± 0.6 / 68.1 ± 0.5 |
| `ER__w0.01` | 66.7 ± 0.7 / 67.8 ± 0.5 | 67.1 ± 0.4 / 70.6 ± 0.7 | 59.2 ± 0.7 / 80.6 ± 0.4 | 40.8 ± 0.6 / 94.1 ± 0.4 | 68.3 ± 0.4 / 68.5 ± 0.5 |
| `ER__w0.03` | 67.1 ± 0.6 / 67.9 ± 0.4 | 67.1 ± 0.4 / 70.5 ± 0.5 | 59.8 ± 0.8 / 79.9 ± 0.5 | 44.9 ± 0.9 / 93.6 ± 0.4 | 66.9 ± 0.4 / 68.6 ± 0.6 |
| `ER__w0.1` | 67.0 ± 0.5 / 67.4 ± 0.5 | 66.9 ± 0.5 / 70.2 ± 0.6 | 60.7 ± 0.9 / 78.6 ± 0.6 | 50.3 ± 0.6 / 91.4 ± 0.4 | 65.4 ± 0.6 / 67.7 ± 0.5 |
| `ER__w0.3` | 66.9 ± 0.6 / 67.7 ± 0.4 | 66.6 ± 0.5 / 69.7 ± 0.6 | 61.4 ± 0.7 / 76.5 ± 0.8 | 52.5 ± 0.8 / 88.6 ± 0.5 | 65.4 ± 0.5 / 66.7 ± 0.3 |
| `ER__w1` | 66.4 ± 0.8 / 66.6 ± 0.3 | 66.5 ± 0.4 / 69.4 ± 0.8 | 62.4 ± 0.6 / 75.0 ± 0.7 | 53.2 ± 0.8 / 87.9 ± 0.5 | 65.5 ± 0.6 / 67.0 ± 0.5 |
| `ER__w3` | 66.0 ± 0.5 / 65.9 ± 0.4 | 65.9 ± 0.4 / 69.1 ± 0.6 | 62.7 ± 0.5 / 73.7 ± 0.8 | 53.6 ± 0.7 / 87.1 ± 0.6 | 65.3 ± 0.5 / 65.4 ± 0.5 |
| `PDCL_RS` | 66.4 ± 0.7 / 68.2 ± 0.4 | 66.9 ± 0.5 / 70.7 ± 0.7 | 59.4 ± 1.0 / 79.6 ± 0.6 | 49.5 ± 0.6 / 90.6 ± 0.5 | 68.2 ± 0.4 / 68.5 ± 0.5 |
| `ERM` | 70.5 ± 0.4 / 69.0 ± 0.3 | 69.8 ± 0.6 / 69.4 ± 0.6 | 69.1 ± 0.5 / 71.1 ± 0.6 | 69.5 ± 0.5 / 70.0 ± 2.1 | 70.5 ± 0.3 / 67.9 ± 0.4 |

F1, memory 200: earlier-task accuracy / last-task accuracy

| Method | rounds (real) | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen |
|---|---|---|---|---|---|---|
| `Finetune` | 65.9 ± 0.7 / 68.3 ± 0.5 | 63.7 ± 0.5 / 74.5 ± 0.5 | 58.7 ± 0.7 / 81.0 ± 0.4 | 46.6 ± 0.7 / 89.8 ± 0.3 | 38.7 ± 0.4 / 94.3 ± 0.4 | 68.0 ± 0.6 / 68.1 ± 0.5 |
| `ER__w0.03_buf200` | 67.2 ± 0.6 / 68.0 ± 0.4 | 64.7 ± 0.6 / 74.4 ± 0.5 | 59.8 ± 0.7 / 79.9 ± 0.7 | 51.9 ± 0.7 / 89.0 ± 0.4 | 44.9 ± 0.9 / 93.6 ± 0.4 | 66.9 ± 0.5 / 68.1 ± 0.5 |
| `ER__w0.3_buf200` | 66.7 ± 0.6 / 67.5 ± 0.6 | 64.9 ± 0.5 / 72.5 ± 0.7 | 61.4 ± 0.7 / 76.2 ± 0.8 | 56.3 ± 0.7 / 84.2 ± 0.5 | 52.6 ± 0.8 / 88.3 ± 0.4 | 64.9 ± 0.5 / 67.7 ± 0.5 |
| `ER__w1_buf200` | 66.5 ± 0.6 / 67.0 ± 0.4 | 64.4 ± 0.5 / 71.4 ± 0.7 | 62.2 ± 0.5 / 75.2 ± 0.9 | 56.2 ± 0.6 / 82.8 ± 0.6 | 53.1 ± 0.9 / 87.3 ± 0.5 | 65.1 ± 0.5 / 66.7 ± 0.5 |
| `ER__w3_buf200` | 66.1 ± 0.7 / 66.1 ± 0.4 | 64.1 ± 0.4 / 70.9 ± 0.6 | 62.7 ± 0.5 / 73.8 ± 0.7 | 56.8 ± 0.6 / 82.0 ± 0.8 | 53.2 ± 0.8 / 86.5 ± 0.7 | 65.2 ± 0.4 / 65.9 ± 0.7 |
| `PDCL_RS__memory_buf200` | 66.4 ± 0.6 / 68.3 ± 0.5 | 64.1 ± 0.5 / 74.2 ± 0.5 | 59.2 ± 0.9 / 79.4 ± 0.6 | 53.2 ± 0.8 / 86.8 ± 0.3 | 49.8 ± 0.6 / 90.5 ± 0.5 | 68.0 ± 0.5 / 68.5 ± 0.5 |
| `PDCL_RS__val_buf200` | 66.0 ± 1.0 / 67.0 ± 0.3 | 65.0 ± 0.5 / 70.8 ± 0.6 | 62.6 ± 0.4 / 73.4 ± 0.7 | 56.9 ± 0.5 / 80.9 ± 0.8 | 54.5 ± 0.8 / 85.5 ± 0.6 | 68.3 ± 0.4 / 68.7 ± 0.5 |
| `PDCL_RS__split_buf200` | 66.1 ± 0.5 / 67.1 ± 0.3 | 64.2 ± 0.2 / 71.7 ± 0.4 | 60.6 ± 0.6 / 76.6 ± 0.5 | 55.9 ± 0.6 / 83.8 ± 0.4 | 50.6 ± 0.6 / 88.5 ± 0.4 | 65.5 ± 0.8 / 67.6 ± 0.5 |
| `ERM` | 70.5 ± 0.4 / 69.0 ± 0.3 | 69.5 ± 0.6 / 69.6 ± 0.7 | 69.1 ± 0.5 / 71.1 ± 0.6 | 69.7 ± 0.3 / 70.2 ± 0.8 | 69.5 ± 0.5 / 70.0 ± 2.1 | 70.5 ± 0.3 / 67.9 ± 0.4 |

## G2 (exploratory). Accuracy on the next round before training on it (real rounds)

Validation accuracy on round k+1 at the checkpoint that ends the phase of round k (steps 1000, 2000, 3000). That checkpoint has taken one update of round k+1 (see the header of the script); the `strict` table uses the last checkpoint before it (steps 750, 1750, 2750) instead. Joint training (ERM) draws from all four rounds from the first step, so it has always seen round k+1 and is not a valid "rounds seen so far" baseline; it is left out. In the first phase no method has anything to replay, so the three methods are the same model there and differ only from the second phase on. Roots are not pooled; E1 main is the primary table. In brackets: difference from fine-tuning, paired by seed.

E1 main (primary), memory 200:

| After round | `Finetune` | `ER` | `PDCL_RS` |
|---|---|---|---|
| 1 → round 2 | 56.02 ± 0.79 | 55.74 ± 0.76 (-0.28 ± 0.16) | 56.02 ± 0.79 (+0.00 ± 0.00) |
| 2 → round 3 | 65.03 ± 0.66 | 65.55 ± 0.90 (+0.52 ± 1.22) | 64.97 ± 0.67 (-0.06 ± 0.20) |
| 3 → round 4 | 62.92 ± 0.55 | 63.68 ± 0.57 (+0.76 ± 0.60) | 63.04 ± 0.49 (+0.12 ± 0.15) |
| mean of 2 and 3 | 63.97 ± 0.48 | 64.61 ± 0.41 (+0.64 ± 0.61) | 64.01 ± 0.49 (+0.03 ± 0.11) |

E1 main, paired differences:

| After round | ER − fine-tuning | PDCL − fine-tuning | PDCL − ER |
|---|---|---|---|
| 1 → round 2 | -0.28 ± 0.16 | +0.00 ± 0.00 | +0.28 ± 0.16 |
| 2 → round 3 | +0.52 ± 1.22 | -0.06 ± 0.20 | -0.57 ± 1.21 |
| 3 → round 4 | +0.76 ± 0.60 | +0.12 ± 0.15 | -0.64 ± 0.53 |
| mean of 2 and 3 | +0.64 ± 0.61 | +0.03 ± 0.11 | -0.61 ± 0.60 |

E1 main, strict (last checkpoint before the phase ends):

| After round | `Finetune` | `ER` | `PDCL_RS` |
|---|---|---|---|
| 1 → round 2 | 53.68 ± 0.97 | 53.68 ± 0.97 (+0.00 ± 0.00) | 53.68 ± 0.97 (+0.00 ± 0.00) |
| 2 → round 3 | 64.94 ± 1.23 | 63.76 ± 1.04 (-1.18 ± 1.79) | 65.00 ± 1.21 (+0.06 ± 0.17) |
| 3 → round 4 | 62.44 ± 0.46 | 63.88 ± 0.59 (**+1.44 ± 0.60**) | 62.62 ± 0.50 (+0.18 ± 0.19) |
| mean of 2 and 3 | 63.69 ± 0.55 | 63.82 ± 0.61 (+0.13 ± 0.94) | 63.81 ± 0.51 (+0.12 ± 0.14) |

E2 (secondary), memory 200:

| After round | `Finetune` | `ER__w0.01` | `ER__w0.03` | `ER__w0.1` | `ER__w0.3` | `ER__w1` | `ER__w3` | `PDCL_RS` |
|---|---|---|---|---|---|---|---|---|
| 1 → round 2 | 56.02 ± 0.79 | 56.09 ± 0.77 (+0.06 ± 0.14) | 56.09 ± 0.77 (+0.06 ± 0.13) | 56.07 ± 0.76 (+0.05 ± 0.14) | 56.01 ± 0.74 (-0.01 ± 0.15) | 56.15 ± 0.81 (+0.13 ± 0.20) | 56.03 ± 0.86 (+0.01 ± 0.27) | 56.09 ± 0.77 (+0.06 ± 0.13) |
| 2 → round 3 | 65.03 ± 0.66 | 65.03 ± 0.81 (-0.00 ± 0.26) | 65.11 ± 0.74 (+0.09 ± 0.10) | 65.20 ± 0.64 (+0.17 ± 0.20) | 65.43 ± 0.56 (+0.40 ± 0.79) | 64.83 ± 1.12 (-0.20 ± 1.34) | 63.33 ± 1.21 (-1.70 ± 1.43) | 64.77 ± 0.75 (-0.26 ± 0.25) |
| 3 → round 4 | 62.92 ± 0.55 | 63.28 ± 0.45 (+0.37 ± 0.18) | 63.30 ± 0.47 (+0.38 ± 0.26) | 63.41 ± 0.39 (+0.49 ± 0.39) | 63.85 ± 0.52 (+0.94 ± 0.56) | 63.31 ± 0.63 (+0.39 ± 0.61) | 62.86 ± 0.63 (-0.05 ± 0.56) | 63.24 ± 0.53 (+0.33 ± 0.17) |
| mean of 2 and 3 | 63.97 ± 0.48 | 64.16 ± 0.52 (+0.18 ± 0.20) | 64.21 ± 0.48 (**+0.23 ± 0.11**) | 64.30 ± 0.38 (+0.33 ± 0.22) | 64.64 ± 0.31 (**+0.67 ± 0.33**) | 64.07 ± 0.60 (+0.10 ± 0.70) | 63.10 ± 0.61 (-0.87 ± 0.71) | 64.01 ± 0.53 (+0.03 ± 0.18) |

F1 (secondary), memory 200:

| After round | `Finetune` | `ER__w0.03_buf200` | `ER__w0.3_buf200` | `ER__w1_buf200` | `ER__w3_buf200` | `PDCL_RS__memory_buf200` | `PDCL_RS__val_buf200` | `PDCL_RS__split_buf200` |
|---|---|---|---|---|---|---|---|---|
| 1 → round 2 | 56.02 ± 0.79 | 56.07 ± 0.77 (+0.05 ± 0.13) | 55.88 ± 0.79 (-0.14 ± 0.21) | 56.23 ± 0.80 (+0.21 ± 0.19) | 55.78 ± 0.81 (-0.25 ± 0.36) | 56.09 ± 0.77 (+0.06 ± 0.13) | 56.09 ± 0.77 (+0.06 ± 0.13) | 56.09 ± 0.77 (+0.06 ± 0.13) |
| 2 → round 3 | 65.03 ± 0.66 | 65.09 ± 0.75 (+0.06 ± 0.19) | 64.94 ± 0.52 (-0.09 ± 0.74) | 65.32 ± 1.10 (+0.29 ± 1.43) | 64.34 ± 0.95 (-0.69 ± 1.22) | 64.57 ± 0.62 (-0.46 ± 0.25) | 64.80 ± 0.71 (-0.23 ± 0.24) | 65.29 ± 0.58 (+0.26 ± 0.31) |
| 3 → round 4 | 62.92 ± 0.55 | 63.45 ± 0.55 (+0.53 ± 0.31) | 63.70 ± 0.51 (+0.79 ± 0.52) | 63.65 ± 0.60 (+0.73 ± 0.65) | 63.09 ± 0.47 (+0.18 ± 0.57) | 63.45 ± 0.51 (**+0.53 ± 0.24**) | 62.73 ± 0.54 (-0.19 ± 0.50) | 63.22 ± 0.55 (+0.30 ± 0.62) |
| mean of 2 and 3 | 63.97 ± 0.48 | 64.27 ± 0.51 (**+0.29 ± 0.12**) | 64.32 ± 0.41 (+0.35 ± 0.40) | 64.48 ± 0.60 (+0.51 ± 0.82) | 63.72 ± 0.54 (-0.26 ± 0.60) | 64.01 ± 0.45 (+0.03 ± 0.16) | 63.76 ± 0.47 (-0.21 ± 0.32) | 64.25 ± 0.35 (+0.28 ± 0.40) |

## G3 (exploratory). Constraints on validation data

Per run: the share (%) of earlier tasks (0–2) whose final validation loss is at most 1.1 × its validation loss at the end of that task's own phase; mean ± standard error over seeds. Bold = differs from fine-tuning on the same stream by more than two standard errors (paired); the last column averages each seed over the streams first. Per-task validation loss is logged only in F1 and F2 (`val_loss_k` in F1, `env{k}_out_loss` in F2); E1, E1 long, E2, E3, F3 log validation accuracy and the memory's loss only, so E1 main and the E2 fixed weights cannot be computed. F1 holds the default PDCL (`memory`), the fixed replay weights and both variants at memory 200 and 1000; F2 has replay and fine-tuning only.

F1, memory 200:

| Method | rounds (real) | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen | All streams (− fine-tuning) |
|---|---|---|---|---|---|---|---|
| `Finetune` | 53 ± 5 | 33 ± 0 | 30 ± 3 | 27 ± 4 | 33 ± 0 | 67 ± 5 | 41 ± 1 |
| `ER__w0.03_buf200` | 53 ± 7 | 33 ± 0 | 27 ± 4 | 27 ± 4 | 23 ± 5 | **20 ± 5** | 31 ± 2 (**-10 ± 2**) |
| `ER__w0.3_buf200` | **27 ± 4** | **20 ± 5** | 17 ± 6 | 27 ± 4 | 23 ± 5 | **0 ± 0** | 19 ± 2 (**-22 ± 2**) |
| `ER__w1_buf200` | **17 ± 6** | **20 ± 5** | 23 ± 5 | 30 ± 3 | 30 ± 3 | **0 ± 0** | 20 ± 2 (**-21 ± 2**) |
| `ER__w3_buf200` | **7 ± 4** | 23 ± 5 | 23 ± 5 | 30 ± 3 | 33 ± 0 | **0 ± 0** | 19 ± 1 (**-21 ± 1**) |
| `PDCL_RS__memory_buf200` | 57 ± 7 | 33 ± 0 | 30 ± 3 | 33 ± 0 | 33 ± 0 | 70 ± 6 | 43 ± 2 (+2 ± 2) |
| `PDCL_RS__val_buf200` | **27 ± 10** | **10 ± 5** | **13 ± 5** | 30 ± 3 | 33 ± 0 | 73 ± 8 | 31 ± 2 (**-9 ± 2**) |
| `PDCL_RS__split_buf200` | **10 ± 5** | **20 ± 5** | 27 ± 4 | 33 ± 0 | 33 ± 0 | **3 ± 3** | 21 ± 1 (**-19 ± 2**) |

F1, memory 1000:

| Method | rounds (real) | skew:0.7 | skew:0.8 | skew:0.9 | skew:0.95 | aglen | All streams (− fine-tuning) |
|---|---|---|---|---|---|---|---|
| `Finetune` | 53 ± 5 | 33 ± 0 | 30 ± 3 | 27 ± 4 | 33 ± 0 | 67 ± 5 | 41 ± 1 |
| `ER__w0.03_buf1000` | **73 ± 10** | 33 ± 0 | 33 ± 0 | 27 ± 4 | 30 ± 3 | 83 ± 7 | 47 ± 2 (**+6 ± 2**) |
| `ER__w0.3_buf1000` | 47 ± 7 | **20 ± 5** | 17 ± 6 | 17 ± 6 | **20 ± 5** | **3 ± 3** | 21 ± 3 (**-20 ± 2**) |
| `ER__w1_buf1000` | **13 ± 5** | **10 ± 5** | 17 ± 6 | 30 ± 3 | 33 ± 0 | **0 ± 0** | 17 ± 2 (**-23 ± 2**) |
| `ER__w3_buf1000` | **3 ± 3** | **3 ± 3** | 20 ± 5 | 23 ± 5 | 33 ± 0 | **0 ± 0** | 14 ± 2 (**-27 ± 2**) |
| `PDCL_RS__memory_buf1000` | 70 ± 9 | 33 ± 0 | 33 ± 0 | 33 ± 0 | 33 ± 0 | 60 ± 7 | 44 ± 2 (+3 ± 2) |
| `PDCL_RS__val_buf1000` | 63 ± 13 | **13 ± 5** | **3 ± 3** | 27 ± 4 | 33 ± 0 | 67 ± 7 | 34 ± 3 (**-6 ± 2**) |
| `PDCL_RS__split_buf1000` | **3 ± 3** | **0 ± 0** | **13 ± 5** | 30 ± 3 | 30 ± 3 | **40 ± 10** | 19 ± 2 (**-21 ± 2**) |

F1, all streams, paired differences in the share (points):

| Pair | Difference |
|---|---|
| `PDCL_RS__val_buf200` − `PDCL_RS__memory_buf200` | **-12 ± 2** |
| `PDCL_RS__memory_buf200` − `ER__w1_buf200` | **+23 ± 2** |
| `PDCL_RS__val_buf200` − `ER__w1_buf200` | **+11 ± 1** |
| `PDCL_RS__val_buf200` − `ER__w3_buf200` | **+12 ± 2** |
| `PDCL_RS__val_buf1000` − `PDCL_RS__memory_buf1000` | **-9 ± 2** |
| `PDCL_RS__memory_buf1000` − `ER__w1_buf1000` | **+27 ± 3** |
| `PDCL_RS__val_buf1000` − `ER__w1_buf1000` | **+17 ± 3** |
| `PDCL_RS__val_buf1000` − `ER__w3_buf1000` | **+21 ± 3** |

F2 (replay batch and weight decay; no PDCL):

| Method | rounds (real) | skew:0.9 | aglen | All streams (− fine-tuning) |
|---|---|---|---|---|
| `Finetune` | 53 ± 5 | 27 ± 4 | 67 ± 5 | 49 ± 2 |
| `ER__m200_rb2` | **7 ± 4** | 33 ± 0 | **0 ± 0** | 13 ± 1 (**-36 ± 2**) |
| `ER__m200_rb32` | **13 ± 5** | 30 ± 3 | **0 ± 0** | 14 ± 2 (**-34 ± 3**) |
| `ER__m200_rb8` | **17 ± 6** | 27 ± 4 | **0 ± 0** | 14 ± 3 (**-34 ± 3**) |
| `ER__m200_wd0.01` | **20 ± 5** | 33 ± 0 | **0 ± 0** | 18 ± 2 (**-31 ± 2**) |
| `ER__m200_wd0.1` | **90 ± 10** | 33 ± 0 | **0 ± 0** | 41 ± 3 (**-8 ± 4**) |
| `ER__m50_rb2` | **23 ± 7** | 33 ± 0 | **0 ± 0** | 19 ± 2 (**-30 ± 3**) |
| `ER__m50_rb32` | **30 ± 6** | 33 ± 0 | **0 ± 0** | 21 ± 2 (**-28 ± 3**) |
| `ER__m50_rb8` | **27 ± 7** | 33 ± 0 | **0 ± 0** | 20 ± 2 (**-29 ± 3**) |
| `ER__m50_wd0.01` | **37 ± 6** | 33 ± 0 | **0 ± 0** | 23 ± 2 (**-26 ± 3**) |
| `ER__m50_wd0.1` | 67 ± 14 | 30 ± 3 | **0 ± 0** | 32 ± 5 (**-17 ± 6**) |
| `Finetune__wd0.01` | **73 ± 8** | 33 ± 0 | 73 ± 7 | 60 ± 3 (**+11 ± 3**) |
| `Finetune__wd0.1` | **100 ± 0** | 33 ± 0 | 73 ± 4 | 69 ± 1 (**+20 ± 3**) |
