# F3. Tolerable drift

Design: `findings.md` section 0b, followed without change. Fine-tuning, ER, PDCL (as
published) and joint ERM, whole stream, 1,000 steps per task, seeds 0–9, final accuracy,
differences paired by seed, real = beyond two standard errors.

Data: 200 new runs (`skew:0.75`, `skew:0.85`, `drift:0.4`, `0.6`, `0.8`; spec
`results/f3_sweeps.json`, per-run table `results/f3_runs.csv`) plus 280 runs of the first
round (`skew:0.5`–`0.95` and `rounds`) read from `results/e1_runs_main.csv`, which is
`results/e1_runs.csv` of branch `claude/pdcl-e1-retention-dial-6f00e4`, copied unchanged.
Everything below is recomputed by `python -m domainbed.scripts.f3_analysis`
(`results/f3_tables.md`, `paper/figures/fig_f3_dial.{pdf,png}`).

Swing = highest minus lowest binder rate among the four tasks. Threshold of a method = the
smallest swing at which it is more than two standard errors above fine-tuning and stays so
at all larger swings.

## Thresholds

| Method | Alternating (swings 0 to 0.90) | Gradual (swings 0.40, 0.60, 0.80) |
|---|---|---|
| ER | 0.70 | 0.40 or lower |
| PDCL | 0.70 | 0.40 or lower |

- Alternating: both methods have the same threshold, swing 0.70 (`skew:0.85`): ER
  +3.31 ± 0.69, PDCL +2.48 ± 0.44. Below it neither holds a gain. PDCL is above two
  standard errors once at swing 0.50 (+0.89 ± 0.33) and back to a null at 0.60
  (+0.02 ± 0.45), so 0.50 is not its threshold. ER at 0.60 is +0.95 ± 0.55, under two
  standard errors.
- Gradual: both methods are already above fine-tuning at the smallest drift tested, 0.40
  (ER +1.46 ± 0.60, PDCL +0.64 ± 0.24), and stay so at 0.60 and 0.80. The threshold is
  0.40 or lower; no smaller drift was run, so its exact place is unknown.
- The thresholds are the same for ER and PDCL on each dial, and differ between the dials:
  0.70 alternating, at most 0.40 gradual.
- ER is never really below PDCL where memory helps: PDCL − ER is −0.8 to −1.5 from swing
  0.60 up on both dials, real at alternating 0.80 and 0.90 and gradual 0.80. At swing 0 ER is
  0.96 ± 0.41 below fine-tuning and PDCL 0.17 ± 0.08 below it (both beyond two standard
  errors; PDCL's cost is small).
- Real rounds (swing 0.20): ER +0.30 ± 0.35, PDCL +0.08 ± 0.24. Nulls.

## Predictions

| Prediction (section 0b) | Outcome |
|---|---|
| Alternating: ER's threshold is at swing 0.6 | **Failed.** It is 0.70; at 0.60 ER is +0.95 ± 0.55. |
| Alternating: PDCL's threshold is at 0.7 or 0.8 | **Held.** 0.70. |
| At equal swing, gradual drift costs fine-tuning less final accuracy than alternation | **Failed.** Fine-tuning's final accuracy is lower under gradual drift at swing 0.40 (−2.32 ± 0.67) and 0.60 (−3.81 ± 0.70) and equal at 0.80 (+0.06 ± 0.61). The same holds measured against joint training. Forgetting, in contrast, is lower under gradual drift (−2.79 ± 1.16 at 0.60, −13.22 ± 1.04 at 0.80). |
| Thresholds are higher under gradual drift | **Failed, reversed.** At most 0.40 for both methods, against 0.70 alternating. |
| The real rounds' swing of 0.20 is below every threshold | **Held on the alternating dial** (0.70). **Not settled on the gradual dial**: the threshold there is only bounded above by 0.40. On the real rounds themselves both methods are nulls. |

## Table

Mean ± standard error over 10 seeds; bold: beyond two standard errors.

| Dial | Stream | Swing | Joint, final accuracy | FT, final accuracy | FT, forgetting | ER − FT | PDCL − FT | PDCL − ER |
|---|---|---|---|---|---|---|---|---|
| Alternating | skew:0.5 | 0.00 | 69.42 ± 0.27 | 68.15 ± 0.36 | -0.30 ± 0.42 | **-0.96 ± 0.41** | **-0.17 ± 0.08** | **+0.79 ± 0.37** |
| Alternating | skew:0.6 | 0.20 | 69.70 ± 0.41 | 67.97 ± 0.41 | 2.87 ± 0.59 | -0.79 ± 0.46 | -0.01 ± 0.15 | +0.78 ± 0.52 |
| Alternating | skew:0.7 | 0.40 | 69.54 ± 0.30 | 66.38 ± 0.34 | 10.16 ± 0.58 | +0.06 ± 0.58 | +0.17 ± 0.17 | +0.11 ± 0.58 |
| Alternating | skew:0.75 | 0.50 | 69.41 ± 0.25 | 65.12 ± 0.47 | 16.83 ± 0.91 | +0.87 ± 0.61 | **+0.89 ± 0.33** | +0.02 ± 0.58 |
| Alternating | skew:0.8 | 0.60 | 69.62 ± 0.33 | 64.27 ± 0.50 | 22.38 ± 0.63 | +0.95 ± 0.55 | +0.02 ± 0.45 | -0.93 ± 0.47 |
| Alternating | skew:0.85 | 0.70 | 69.47 ± 0.34 | 60.45 ± 0.60 | 34.05 ± 0.81 | **+3.31 ± 0.69** | **+2.48 ± 0.44** | -0.83 ± 0.46 |
| Alternating | skew:0.9 | 0.80 | 69.84 ± 0.18 | 57.42 ± 0.51 | 43.97 ± 0.78 | **+5.44 ± 0.42** | **+4.07 ± 0.34** | **-1.37 ± 0.39** |
| Alternating | skew:0.95 | 0.90 | 69.64 ± 0.40 | 52.57 ± 0.32 | 56.51 ± 0.56 | **+8.89 ± 0.47** | **+7.43 ± 0.43** | **-1.46 ± 0.31** |
| Gradual | drift:0.4 | 0.40 | 70.14 ± 0.36 | 64.06 ± 0.69 | 9.50 ± 0.90 | **+1.46 ± 0.60** | **+0.64 ± 0.24** | -0.83 ± 0.55 |
| Gradual | drift:0.6 | 0.60 | 69.81 ± 0.33 | 60.46 ± 0.44 | 19.59 ± 0.81 | **+2.36 ± 0.56** | **+1.58 ± 0.27** | -0.79 ± 0.65 |
| Gradual | drift:0.8 | 0.80 | 69.88 ± 0.27 | 57.48 ± 0.42 | 30.75 ± 0.74 | **+3.94 ± 0.57** | **+2.98 ± 0.50** | **-0.96 ± 0.39** |
| Reference | Rounds (real) | 0.20 | 70.10 ± 0.28 | 66.54 ± 0.56 | 3.80 ± 0.67 | +0.30 ± 0.35 | +0.08 ± 0.24 | -0.22 ± 0.24 |

Gradual minus alternating at equal swing, same seeds:

| Swing | Alternating | Gradual | FT final accuracy | FT forgetting | FT − Joint |
|---|---|---|---|---|---|
| 0.40 | skew:0.7 | drift:0.4 | **-2.32 ± 0.67** | -0.66 ± 0.97 | **-2.93 ± 0.71** |
| 0.60 | skew:0.8 | drift:0.6 | **-3.81 ± 0.70** | **-2.79 ± 1.16** | **-4.00 ± 0.87** |
| 0.80 | skew:0.9 | drift:0.8 | +0.06 ± 0.61 | **-13.22 ± 1.04** | +0.02 ± 0.66 |

Per-method final accuracy, forgetting and learned accuracy of every stream:
`results/f3_tables.md`.

## Limits

- The gradual dial has three points and starts at 0.40, where both methods are already
  above fine-tuning. Whether a gradual drift of 0.20, the size of the real rounds' swing,
  is below the threshold was not tested; the pre-registered design did not include it.
- Thresholds rest on a two-standard-error rule with 10 seeds. Standard errors of the gains
  are 0.2 to 0.7 points, so gains under about 1 point (ER at alternating 0.50 and 0.60)
  cannot be told from zero, and the alternating threshold is resolved only to the grid
  step of 0.10. PDCL's single crossing at 0.50 followed by a null at 0.60 is what one
  false positive among eight tests would look like.
- Swing is not a complete description of a stream: at equal swing the two dials differ in
  fine-tuning's final accuracy, its forgetting and the gain from memory. Under gradual
  drift the last task has the most extreme binder rate of the stream (0.72 to 0.92),
  under alternation the lowest of two; this was not separated from gradualness.
- The synthetic streams change only the binder rate; the real rounds also change antigens
  and sequence design. Their place on the dial is by swing alone.
- 1,000 steps per task, memory and hyperparameters at defaults; no other budget was run.
- The first training process was stopped from outside after 160 runs and resumed on
  another GPU; the trainer restarts unfinished runs from scratch, and all 200 have `done`.
