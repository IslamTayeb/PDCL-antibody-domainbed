# H1. Long training

Design: `findings.md` section 0c, followed without change. Joint ERM, fine-tuning, PDCL (as
published) and ER with replay weight 0.03, 0.3, 1 on `rounds`, `rounds:3210`, `aglen`,
`target`; whole stream, 5,000 steps per task, memory 200, seeds 0–9. Outcome: final accuracy
(validation accuracy at the last step, mean of the four tasks). Differences are paired by
seed; real = beyond two standard errors.

Data: 240 new runs, all finished (spec `results/h1_sweeps.json`, per-run table
`results/h1_runs.csv`, checkpoints `results/h1_curves.csv`). The 1,000-steps-per-task
numbers are the first round's, same seeds: `results/e1_runs_main.csv` (joint ERM,
fine-tuning, PDCL, ER w=1; all four streams) and `results/e2_runs_stream.csv` (ER w=0.03 and
0.3; `rounds` and `aglen` only). Everything below is recomputed by
`python -m domainbed.scripts.h1_analysis` (`results/h1_tables.md`,
`paper/figures/fig_h1_long.{pdf,png}`). The runs were split by stream over GPUs 6, 0 and 5,
one process each, same spec.

## Answer

**Do memory methods beat fine-tuning on natural streams at this length?** Yes on the two
streams with new antigens, weakly on the real rounds, no on the reversed rounds.

- `aglen`: every memory method is 2.3 to 3.2 points above fine-tuning. `target`: 3.0 to 3.7.
  All eight differences are beyond two standard errors.
- `rounds`: +0.6 to +1.0; ER w=0.3, ER w=1 and PDCL are just beyond two standard errors, ER
  w=0.03 is not. `rounds:3210`: +0.1 to +0.6, all nulls.
- The gain comes from fine-tuning getting worse, not from memory getting better. From
  1,000 to 5,000 steps per task fine-tuning loses 5.0 points on `aglen` and 5.3 on `target`;
  ER w=1 does not move (+0.2, −0.5, nulls). No method at 5,000 steps reaches what
  fine-tuning or PDCL reach at 1,000 steps on these two streams (68.0 to 68.3 on `aglen`,
  67.5 to 68.3 on `target`, against at most 66.2 and 65.9 with memory at 5,000). Long
  training with memory is a repair of long training, not an improvement over short training.

**Does PDCL match or beat replay there?** It matches replay with weight 1 and never beats
replay.

- PDCL − ER w=1 is a null on all four streams (−0.03, −0.13, −0.44, −0.23).
- Against the best of the three weights PDCL is behind on `aglen` (−0.91 ± 0.45, w=0.03) and
  `target` (−0.72 ± 0.29, w=0.3), both beyond two standard errors, and level on the two
  rounds streams. The best weight is picked on the same seeds, which favours replay.
- The three replay weights are within 0.5 points of each other on every stream, so at this
  length the weight hardly matters.

## Predictions

| Prediction (section 0c) | Outcome |
|---|---|
| ER w=1 − fine-tuning is positive beyond two standard errors on `aglen` and `target` | **Held.** +2.76 ± 0.48 and +3.18 ± 0.47. |
| PDCL − fine-tuning is positive on at least one of the four streams | **Held.** `rounds` +0.93 ± 0.35, `aglen` +2.32 ± 0.34, `target` +2.96 ± 0.47; null on `rounds:3210` (+0.21 ± 0.31). |
| ... but smaller than ER's gain | **Held in sign only.** PDCL's gain is below ER w=1's on all four streams, by 0.03 to 0.44 points, never beyond two standard errors. Against the best weight it is really smaller on `aglen` and `target`. |
| PDCL does not beat the best of the three replay weights on any stream | **Held.** −0.03 ± 0.37, −0.41 ± 0.32, −0.91 ± 0.45, −0.72 ± 0.29. |

## Table

Final accuracy at 5,000 steps per task, mean ± standard error over 10 seeds; bold: beyond
two standard errors.

| Stream | FT | ER 0.03 − FT | ER 0.3 − FT | ER 1 − FT | PDCL − FT | Joint − FT | PDCL − ER 1 | PDCL − best weight |
|---|---|---|---|---|---|---|---|---|
| rounds | 65.59 ± 0.43 | +0.57 ± 0.35 | **+0.65 ± 0.32** | **+0.95 ± 0.43** | **+0.93 ± 0.35** | **+3.58 ± 0.34** | −0.03 ± 0.37 | −0.03 ± 0.37 (w=1) |
| rounds:3210 | 66.49 ± 0.39 | +0.13 ± 0.33 | +0.63 ± 0.33 | +0.35 ± 0.38 | +0.21 ± 0.31 | **+2.51 ± 0.44** | −0.13 ± 0.27 | −0.41 ± 0.32 (w=0.3) |
| aglen | 62.97 ± 0.39 | **+3.23 ± 0.47** | **+3.11 ± 0.56** | **+2.76 ± 0.48** | **+2.32 ± 0.34** | **+5.41 ± 0.47** | −0.44 ± 0.27 | **−0.91 ± 0.45** (w=0.03) |
| target | 62.19 ± 0.51 | **+3.36 ± 0.64** | **+3.68 ± 0.63** | **+3.18 ± 0.47** | **+2.96 ± 0.47** | **+6.08 ± 0.57** | −0.23 ± 0.24 | **−0.72 ± 0.29** (w=0.3) |

## Secondary: 5,000 against 1,000 steps per task (no prediction)

Same method, same seeds, 5,000 − 1,000. "–": not run at 1,000 on that stream.

| Stream | Joint | FT | ER 0.03 | ER 0.3 | ER 1 | PDCL |
|---|---|---|---|---|---|---|
| rounds | **−0.93 ± 0.28** | −0.95 ± 0.69 | −1.14 ± 0.60 | −0.83 ± 0.57 | −0.30 ± 0.53 | −0.10 ± 0.54 |
| rounds:3210 | **−1.00 ± 0.33** | **−0.89 ± 0.36** | – | – | +0.40 ± 0.47 | **−0.66 ± 0.27** |
| aglen | **−1.45 ± 0.38** | **−5.03 ± 0.43** | **−1.12 ± 0.47** | +0.37 ± 0.41 | +0.21 ± 0.36 | **−2.94 ± 0.29** |
| target | **−1.24 ± 0.36** | **−5.29 ± 0.48** | – | – | −0.48 ± 0.44 | **−3.17 ± 0.35** |

- Nothing improves with longer training. Joint ERM loses 0.9 to 1.5 points on every stream;
  its seed-mean curve peaks between step 2,500 and 5,500 of 20,000 and declines after
  (figure, panel a).
- Fine-tuning and PDCL lose most on `aglen` and `target` (−5.0, −5.3 and −2.9, −3.2); ER
  w=0.3 and w=1 lose nothing there. This reverses the first round: at 1,000 steps per task
  ER w=1 was 2.5 below fine-tuning on `aglen` and 1.6 below on `target`, at 5,000 it is 2.8
  and 3.2 above.
- Inside the last phase the last step is not the best one: on `aglen` and `target`
  fine-tuning ends 1.6 to 1.7 below its highest checkpoint and PDCL 1.0 below, the replay
  weights 0.2 to 0.8 (`results/h1_tables.md`, over-training table).

## Limits

- Final accuracy is validation accuracy on the tasks of the stream at the last step; no
  held-out task and no checkpoint selection, as pre-registered.
- One memory size (200), default learning rate, one long length. Whether memory helps at
  lengths between 1,000 and 5,000 was not run.
- ER w=0.03 and 0.3 have no 1,000-step runs on `rounds:3210` and `target`.
- The ER w=1 reference at 1,000 steps is the first round's `ER` (default weight 1) from
  `results/e1_runs_main.csv`; `results/e2_runs_stream.csv` holds a second w=1 sweep on
  `rounds` and `aglen` that differs from it by up to 0.4 points in the mean (different
  replay draws).
- "Best weight" and the over-training table's "highest checkpoint" are picked on the same
  seeds they are evaluated on.
- The `rounds` gains are 2.2 to 2.7 standard errors with four comparisons per stream; treat
  them as weak.
