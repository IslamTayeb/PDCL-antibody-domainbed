# E2. Is PDCL's adaptivity needed?

Branch `claude/pdcl-e2-adaptivity-09393c`, cut from `pdcl-scope` (203960c). Design: `findings.md`
section 0, followed without change. 1,220 runs, all with `train_batched.py` (code unchanged).

- Spec: `results/e2_sweeps.json`. Per-run tables: `results/e2_runs_stream.csv` (900 whole-stream
  runs), `results/e2_runs_loo.csv` (320 leave-one-task-out runs). All tables: `results/e2_tables.md`
  (copied in the appendix below).
- Analysis and figures: `python -m domainbed.scripts.e2_analysis` →
  `paper/figures/fig_e2_{weight,memory,heldout}.{pdf,png}`.
- Whole stream: 4,000 steps (1,000 per task), seeds 0–9. Numbers are mean ± standard error over
  ten seeds; differences are paired by seed; **bold** = larger than two standard errors, anything
  else is a null. "ER w" is replay with a fixed weight w on every earlier task; ER w=1 is plain ER.
  Memory is 200 unless stated. Streams are ordered by how much fine-tuning forgets, except `aglen`
  (antigen-disjoint tasks), which is listed last.

## Answer in short

1. **No single fixed weight is within 0.5 points of PDCL on every stream** (pre-registered
   question: answer is no). Every one of the six weights is 2.3 to 5.6 points below PDCL on some
   stream. Small weights fail on `skew:0.95`, where replay is needed; weights of 0.1 and above
   fail on `aglen`, where replay with a 200-example memory *lowers* accuracy by 2 to 3 points and
   PDCL switches it off.
2. **PDCL does not match replay tuned per stream either.** The best fixed weight of each stream
   is at or above PDCL on all five streams, by +1.13 ± 0.36 on `skew:0.8` and +2.23 ± 0.35 on
   `skew:0.95`. Where replay helps, PDCL uses too little of it: on `skew:0.8` it is
   indistinguishable from fine-tuning (+0.18 ± 0.50) while ER w=1 gains +1.31 ± 0.50.
3. **So the adaptivity buys safety, not accuracy.** PDCL is never below fine-tuning (worst
   −0.12 ± 0.13) and never needs the weight to be chosen, but it pays for that with 1 to 2 points
   on the streams where replay matters. Averaged over the five streams PDCL and the middle fixed
   weights are the same (ER w − PDCL: −0.14, −0.07, −0.05 for w = 0.1, 0.3, 1). The adaptivity
   moves the loss from the streams where replay hurts to the streams where it helps; it does not
   remove it.
4. On the paper's stream (`rounds`) everything is within a point of everything else, as in the paper.

## 1. Final accuracy and forgetting

Final accuracy (%):

| Method | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| Joint ERM | 70.10 ± 0.28 | 69.70 ± 0.41 | 69.62 ± 0.33 | 69.64 ± 0.40 | 69.83 ± 0.25 |
| Fine-tuning | 66.54 ± 0.56 | 67.97 ± 0.41 | 64.27 ± 0.50 | 52.57 ± 0.32 | 68.00 ± 0.44 |
| ER w=0.01 | 66.94 ± 0.56 | 67.95 ± 0.38 | 64.55 ± 0.48 | 54.13 ± 0.43 | 68.35 ± 0.33 |
| ER w=0.03 | 67.31 ± 0.49 | 67.95 ± 0.37 | 64.83 ± 0.47 | 57.07 ± 0.66 | 67.31 ± 0.36 |
| ER w=0.1 | 67.13 ± 0.32 | 67.69 ± 0.47 | 65.15 ± 0.57 | 60.57 ± 0.45 | 65.99 ± 0.48 |
| ER w=0.3 | 67.07 ± 0.44 | 67.39 ± 0.39 | 65.19 ± 0.41 | 61.51 ± 0.60 | 65.71 ± 0.41 |
| ER w=1 | 66.43 ± 0.60 | 67.23 ± 0.37 | 65.58 ± 0.34 | 61.83 ± 0.52 | 65.90 ± 0.49 |
| ER w=3 | 66.00 ± 0.38 | 66.67 ± 0.34 | 65.49 ± 0.31 | 62.01 ± 0.50 | 65.29 ± 0.40 |
| PDCL | 66.84 ± 0.56 | 67.85 ± 0.41 | 64.45 ± 0.61 | 59.78 ± 0.46 | 68.30 ± 0.35 |

Forgetting (points):

| Method | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| Joint ERM | 0.43 ± 0.26 | 0.42 ± 0.29 | 1.29 ± 0.39 | 2.73 ± 1.34 | 0.04 ± 0.27 |
| Fine-tuning | 3.80 ± 0.67 | 2.87 ± 0.59 | 22.38 ± 0.63 | 56.51 ± 0.56 | 3.18 ± 0.55 |
| ER w=0.01 | 2.97 ± 0.64 | 2.96 ± 0.49 | 21.79 ± 0.68 | 54.35 ± 0.66 | 2.81 ± 0.31 |
| ER w=0.1 | 2.54 ± 0.43 | 2.54 ± 0.46 | 19.17 ± 1.07 | 43.37 ± 0.55 | 5.50 ± 0.52 |
| ER w=1 | 3.06 ± 0.55 | 1.78 ± 0.45 | 13.55 ± 0.80 | 33.69 ± 0.83 | 4.35 ± 0.44 |
| ER w=3 | 2.91 ± 0.44 | 1.57 ± 0.36 | 12.87 ± 0.66 | 32.48 ± 0.98 | 4.08 ± 0.41 |
| PDCL | 3.42 ± 0.60 | 2.90 ± 0.53 | 19.72 ± 1.19 | 38.17 ± 0.56 | 2.88 ± 0.33 |

Fixed-weight replay minus PDCL, final accuracy:

| w | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| 0.01 | +0.10 ± 0.14 | +0.10 ± 0.15 | +0.10 ± 0.39 | **-5.64 ± 0.41** | +0.05 ± 0.22 |
| 0.03 | **+0.47 ± 0.19** | +0.10 ± 0.17 | +0.37 ± 0.25 | **-2.70 ± 0.43** | **-0.98 ± 0.27** |
| 0.1 | +0.28 ± 0.29 | -0.16 ± 0.28 | **+0.69 ± 0.33** | **+0.80 ± 0.35** | **-2.31 ± 0.40** |
| 0.3 | +0.23 ± 0.47 | -0.46 ± 0.39 | +0.74 ± 0.38 | **+1.73 ± 0.30** | **-2.59 ± 0.25** |
| 1 | -0.41 ± 0.36 | -0.62 ± 0.43 | **+1.13 ± 0.36** | **+2.05 ± 0.43** | **-2.40 ± 0.38** |
| 3 | **-0.84 ± 0.33** | **-1.18 ± 0.43** | +1.03 ± 0.52 | **+2.23 ± 0.35** | **-3.01 ± 0.30** |

Fixed-weight replay, PDCL and joint ERM minus fine-tuning, final accuracy:

| w | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| 0.01 | +0.40 ± 0.32 | -0.02 ± 0.13 | +0.28 ± 0.15 | **+1.56 ± 0.26** | +0.35 ± 0.29 |
| 0.03 | **+0.76 ± 0.21** | -0.02 ± 0.22 | **+0.55 ± 0.28** | **+4.50 ± 0.63** | -0.69 ± 0.40 |
| 0.1 | +0.58 ± 0.32 | -0.28 ± 0.31 | +0.87 ± 0.67 | **+8.00 ± 0.48** | **-2.01 ± 0.58** |
| 0.3 | +0.53 ± 0.47 | -0.58 ± 0.37 | +0.92 ± 0.59 | **+8.93 ± 0.54** | **-2.29 ± 0.48** |
| 1 | -0.11 ± 0.45 | -0.74 ± 0.40 | **+1.31 ± 0.50** | **+9.25 ± 0.54** | **-2.10 ± 0.52** |
| 3 | -0.55 ± 0.43 | **-1.30 ± 0.41** | **+1.21 ± 0.55** | **+9.43 ± 0.41** | **-2.71 ± 0.42** |
| PDCL | +0.30 ± 0.24 | -0.12 ± 0.13 | +0.18 ± 0.50 | **+7.20 ± 0.47** | +0.30 ± 0.28 |
| Joint ERM | **+3.56 ± 0.50** | **+1.73 ± 0.47** | **+5.34 ± 0.53** | **+17.07 ± 0.49** | **+1.83 ± 0.49** |

What the tables show:

- `rounds`, `skew:0.6`: nothing differs from fine-tuning by more than a point. PDCL − fine-tuning
  is a null on both. Heavy replay (w=3) is about a point below PDCL.
- `skew:0.8`: fine-tuning forgets 22 points, yet PDCL − fine-tuning is a null (+0.18 ± 0.50);
  replay with w ≥ 1 gains 1.2 to 1.3 points.
- `skew:0.95`: PDCL gains 7.2 points over fine-tuning; replay with w ≥ 0.1 gains 8.0 to 9.4. Even
  the best replay stays 7.6 points below joint training.
- `aglen`: replay with w ≥ 0.1 is 2.0 to 2.7 points *below* fine-tuning, and it raises forgetting
  (5.5 against 3.2 at w=0.1) instead of lowering it. PDCL equals fine-tuning (+0.30 ± 0.28). This
  is the one stream on which PDCL clearly beats plain ER (+2.40 ± 0.38).

Figure: `paper/figures/fig_e2_weight.png` (final accuracy against w, one panel per stream; bands
are PDCL and fine-tuning ± SE; the dashed vertical line is PDCL's mean dual variable, drawn at the
left edge on `aglen` where it is 0.003).

## 2. The pre-registered question: one fixed weight for all streams?

| w | worst stream against PDCL | ER(w) − PDCL there | never more than 0.5 below PDCL | mean over streams |
|---|---|---|---|---|
| 0.01 | skew:0.95 | **-5.64 ± 0.41** | no | -1.06 |
| 0.03 | skew:0.95 | **-2.70 ± 0.43** | no | -0.55 |
| 0.1 | aglen | **-2.31 ± 0.40** | no | -0.14 |
| 0.3 | aglen | **-2.59 ± 0.25** | no | -0.07 |
| 1 | aglen | **-2.40 ± 0.38** | no | -0.05 |
| 3 | aglen | **-3.01 ± 0.30** | no | -0.36 |

**Answer: no.** No fixed weight is within 0.5 points of PDCL on every stream, whether "within" is
read one-sided (not more than 0.5 below) or two-sided. The closest, w=0.1, is 2.3 points below
PDCL on `aglen`.

Pre-registered prediction, in two parts:

- "No fixed w is best at both ends of the dial": **held.** The best weight is 0.01 to 0.03 on
  `rounds`, `skew:0.6`, `aglen` and 1 to 3 on `skew:0.8`, `skew:0.95`.
- "A middle w may still be close everywhere": **failed.** The middle weights are close on the four
  streams of the dial but not on `aglen` (w=0.1: −2.31; w=0.03: −0.98 there and −2.70 on `skew:0.95`).
  Within the skew dial alone, w=0.1 is at or above PDCL at every Q (−0.16, +0.69, +0.80), so on
  the dial a single weight does suffice; it is the antigen-disjoint stream that breaks it.

Oracle-tuned replay (best fixed weight per stream) against PDCL:

| Stream | best w | ER(best w) | PDCL | ER(best w) − PDCL | ER(best w) − fine-tuning | PDCL mean dual |
|---|---|---|---|---|---|---|
| rounds | 0.03 | 67.31 ± 0.49 | 66.84 ± 0.56 | **+0.47 ± 0.19** | **+0.76 ± 0.21** | 0.015 |
| skew:0.6 | 0.01 | 67.95 ± 0.38 | 67.85 ± 0.41 | +0.10 ± 0.15 | -0.02 ± 0.13 | 0.018 |
| skew:0.8 | 1 | 65.58 ± 0.34 | 64.45 ± 0.61 | **+1.13 ± 0.36** | **+1.31 ± 0.50** | 0.108 |
| skew:0.95 | 3 | 62.01 ± 0.50 | 59.78 ± 0.46 | **+2.23 ± 0.35** | **+9.43 ± 0.41** | 0.480 |
| aglen | 0.01 | 68.35 ± 0.33 | 68.30 ± 0.35 | +0.05 ± 0.22 | +0.35 ± 0.29 | 0.003 |

The best weight is picked on the same ten seeds it is scored on, so these differences are biased
upward. That matters for `rounds` (+0.47 is the best of six and should be read as weak) but not
for `skew:0.95`, where every w ≥ 0.1 beats PDCL, or `skew:0.8`, where w=0.1 and w=1 both do.

**Verdict.** PDCL's adaptivity does not buy accuracy over a replay weight tuned per stream: tuned
replay is at or above PDCL everywhere and 1 to 2 points above it where replay matters. It does
save the search, but only partly: an untuned PDCL is as good as an untuned middle weight on
average and has its losses in different places.

*Exploratory* (not pre-registered): shortfall of each method from the best of fine-tuning, the
six weights and PDCL on each stream.

| Method | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen | worst stream | mean |
|---|---|---|---|---|---|---|---|
| Fine-tuning | **-0.76 ± 0.21** | best | **-1.31 ± 0.50** | **-9.43 ± 0.41** | -0.35 ± 0.29 | -9.43 | -2.37 |
| ER w=0.01 | -0.37 ± 0.25 | -0.02 ± 0.13 | **-1.03 ± 0.43** | **-7.87 ± 0.40** | best | -7.87 | -1.86 |
| ER w=0.03 | best | -0.02 ± 0.22 | **-0.75 ± 0.32** | **-4.93 ± 0.53** | **-1.03 ± 0.16** | -4.93 | -1.35 |
| ER w=0.1 | -0.18 ± 0.27 | -0.28 ± 0.31 | -0.44 ± 0.31 | **-1.43 ± 0.42** | **-2.35 ± 0.32** | -2.35 | -0.94 |
| ER w=0.3 | -0.23 ± 0.47 | -0.58 ± 0.37 | **-0.39 ± 0.18** | **-0.50 ± 0.20** | **-2.64 ± 0.26** | -2.64 | -0.87 |
| ER w=1 | **-0.88 ± 0.43** | -0.74 ± 0.40 | best | -0.18 ± 0.24 | **-2.45 ± 0.32** | -2.45 | -0.85 |
| ER w=3 | **-1.31 ± 0.32** | **-1.30 ± 0.41** | -0.10 ± 0.26 | best | **-3.06 ± 0.27** | -3.06 | -1.15 |
| PDCL | **-0.47 ± 0.19** | -0.12 ± 0.13 | **-1.13 ± 0.36** | **-2.23 ± 0.35** | -0.05 ± 0.22 | -2.23 | -0.80 |

PDCL's worst case (−2.23) and mean (−0.80) are the same as those of a fixed weight of 0.1 to 1
(worst −2.35 to −2.64, mean −0.85 to −0.94). PDCL is not a better untuned default than a middle
fixed weight on these five streams; it is a different one.

## 3. What weight does PDCL use?

Dual variable of each earlier task, default PDCL:

| Stream | mean after its phase: task 0 | task 1 | task 2 | at the last step: task 0 | task 1 | task 2 | best fixed w |
|---|---|---|---|---|---|---|---|
| rounds | 0.014 ± 0.002 | 0.012 ± 0.002 | 0.021 ± 0.005 | 0.004 ± 0.002 | 0.003 ± 0.002 | 0.013 ± 0.003 | 0.03 |
| skew:0.6 | 0.012 ± 0.003 | 0.015 ± 0.002 | 0.027 ± 0.004 | 0.004 ± 0.002 | 0.004 ± 0.001 | 0.019 ± 0.003 | 0.01 |
| skew:0.8 | 0.154 ± 0.005 | 0.070 ± 0.003 | 0.100 ± 0.006 | 0.012 ± 0.004 | 0.000 ± 0.000 | 0.035 ± 0.003 | 1 |
| skew:0.95 | 0.769 ± 0.035 | 0.111 ± 0.005 | 0.560 ± 0.031 | 0.040 ± 0.007 | 0.000 ± 0.000 | 0.365 ± 0.035 | 3 |
| aglen | 0.001 ± 0.000 | 0.002 ± 0.001 | 0.004 ± 0.001 | 0.002 ± 0.001 | 0.001 ± 0.000 | 0.002 ± 0.001 | 0.01 |

- **The order is right.** The mean dual variable ranks the streams as the best fixed weight does
  (0.003 on `aglen`, 0.015 to 0.018 on `rounds` and `skew:0.6`, 0.11 on `skew:0.8`, 0.48 on
  `skew:0.95`), and within the skewed streams it is larger for the tasks whose binder rate
  differs from the current one (tasks 0 and 2) than for task 1.
- **The size is too small where replay helps**: 0.11 against a best weight of 1 on `skew:0.8`,
  0.48 against 3 on `skew:0.95`. The accuracy curve is flat above w=0.3, so the factor itself
  means little, but a *fixed* weight equal to PDCL's own average does better than PDCL (w=0.1
  on `skew:0.8`: +0.69 ± 0.33; w=0.3 on `skew:0.95`: +1.73 ± 0.30).
- A likely reason, from the logged quantities and not tested by an intervention (*exploratory*,
  appendix T17): the dual variables decay towards zero by the end of the stream (last-step values
  0.00 to 0.04 for tasks 0 and 1). The constraint is checked on the stored examples, which the
  model is being trained on. On `skew:0.95` the loss on task 0's stored examples ends at 0.162,
  below its constraint level of 0.172, so the constraint counts as satisfied while validation
  accuracy on task 0 is 29%. ER w=1 drives the same memory loss to 0.017.

## 4. Sensitivity of PDCL, and memory size

PDCL variant minus default PDCL, final accuracy:

| Variant | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| dual step 0.001 | -0.20 ± 0.14 | +0.05 ± 0.17 | -0.02 ± 0.13 | **+1.30 ± 0.38** | -0.20 ± 0.22 |
| dual step 0.1 | +0.10 ± 0.15 | +0.05 ± 0.15 | +0.11 ± 0.67 | -0.04 ± 0.37 | +0.06 ± 0.17 |
| epsilon x1.0 | -0.06 ± 0.22 | +0.06 ± 0.11 | -0.26 ± 0.15 | **+0.52 ± 0.20** | -0.26 ± 0.24 |
| epsilon x1.5 | -0.31 ± 0.22 | -0.05 ± 0.11 | +0.19 ± 0.31 | -0.51 ± 0.29 | **-0.52 ± 0.22** |
| epsilon x2.0 | -0.18 ± 0.15 | -0.02 ± 0.15 | -0.00 ± 0.54 | **-1.27 ± 0.36** | **-0.46 ± 0.19** |

PDCL is insensitive to its two settings on four of the five streams (all within about half a
point; the two bold `aglen` entries are about half a point). Only `skew:0.95` moves: a looser
constraint costs up to 1.3 points, a slower dual step gains 1.3. No variant reaches replay with
w ≥ 0.3 there (best PDCL variant 61.07 ± 0.59, ER w=3 62.01 ± 0.50), and none changes the null
on `skew:0.8` (PDCL − fine-tuning between −0.08 and +0.38 for all six settings).

Memory size (final accuracy minus fine-tuning, and PDCL minus ER w=1):

| Quantity | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| ER − fine-tuning, memory 50 | +0.09 ± 0.32 | **-0.97 ± 0.30** | -0.68 ± 0.45 | **+5.21 ± 0.44** | **-3.96 ± 0.69** |
| ER − fine-tuning, memory 200 | -0.11 ± 0.45 | -0.74 ± 0.40 | **+1.31 ± 0.50** | **+9.25 ± 0.54** | **-2.10 ± 0.52** |
| ER − fine-tuning, memory 1000 | **+2.03 ± 0.55** | -0.43 ± 0.31 | **+2.75 ± 0.43** | **+13.81 ± 0.42** | -0.64 ± 0.42 |
| PDCL − fine-tuning, memory 50 | +0.22 ± 0.21 | -0.05 ± 0.16 | +0.10 ± 0.13 | **+3.40 ± 0.44** | -0.12 ± 0.23 |
| PDCL − fine-tuning, memory 200 | +0.30 ± 0.24 | -0.12 ± 0.13 | +0.18 ± 0.50 | **+7.20 ± 0.47** | +0.30 ± 0.28 |
| PDCL − fine-tuning, memory 1000 | **+1.01 ± 0.33** | -0.10 ± 0.20 | **+1.69 ± 0.44** | **+14.30 ± 0.40** | +0.06 ± 0.24 |
| PDCL − ER, memory 50 | +0.13 ± 0.26 | **+0.91 ± 0.25** | +0.78 ± 0.45 | **-1.81 ± 0.46** | **+3.84 ± 0.55** |
| PDCL − ER, memory 200 | +0.41 ± 0.36 | +0.62 ± 0.43 | **-1.13 ± 0.36** | **-2.05 ± 0.43** | **+2.40 ± 0.38** |
| PDCL − ER, memory 1000 | **-1.02 ± 0.48** | +0.33 ± 0.29 | **-1.06 ± 0.29** | +0.49 ± 0.25 | **+0.70 ± 0.30** |
| PDCL mean dual, memory 50 | 0.005 | 0.005 | 0.028 | 0.245 | 0.001 |
| PDCL mean dual, memory 200 | 0.015 | 0.018 | 0.108 | 0.480 | 0.003 |
| PDCL mean dual, memory 1000 | 0.049 | 0.056 | 0.297 | 0.867 | 0.003 |

- **Replay's cost and benefit both depend on memory size.** Its cost on `aglen` shrinks from
  −3.96 (memory 50) to −2.10 (200) to a null (−0.64 ± 0.42 at 1,000); its benefit on `skew:0.95`
  grows from +5.2 to +13.8. With 1,000 stored examples replay also helps on the paper's stream
  (`rounds`, +2.03 ± 0.55), which it does not at 200. The harm of replay is a small-memory effect.
- **PDCL tracks this in direction.** Its dual variables grow with the memory on every stream but
  `aglen` (`skew:0.95`: 0.25, 0.48, 0.87), and it is never below fine-tuning at any size (worst
  −0.12 ± 0.23). With the small memory it stays at fine-tuning where ER loses 1 to 4 points.
- **It lags replay when memory is large and replay helps**: at 1,000, PDCL is a point below ER
  on `rounds` and `skew:0.8`, equal on `skew:0.95`, and 0.7 above on `aglen`.

Figure: `paper/figures/fig_e2_memory.png`.

## 5. Held-out tasks (secondary)

Leave one task out, 3,000 steps, checkpoint with the lowest validation loss inside the last
phase, averaged over the four held-out tasks per seed.

| Method | aglen | skew:0.9 |
|---|---|---|
| Joint ERM | 47.39 ± 0.47 | 54.02 ± 0.28 |
| Fine-tuning | 47.53 ± 0.33 | 39.66 ± 0.73 |
| Replay (ER, w=1) | 48.19 ± 0.32 | 52.40 ± 0.43 |
| PDCL | 47.38 ± 0.35 | 51.18 ± 0.43 |

| Difference | aglen | skew:0.9 |
|---|---|---|
| ER − fine-tuning | **+0.66 ± 0.15** | **+12.74 ± 0.94** |
| PDCL − fine-tuning | -0.15 ± 0.13 | **+11.52 ± 0.82** |
| PDCL − ER | **-0.81 ± 0.16** | **-1.22 ± 0.47** |
| ER − joint ERM | **+0.80 ± 0.35** | **-1.61 ± 0.46** |
| PDCL − joint ERM | -0.02 ± 0.23 | **-2.84 ± 0.39** |

- **The paper's pattern (replay hurts on held-out data, PDCL avoids it) does not appear on these
  streams; the sign is the other way.** Replay is above fine-tuning on both, and PDCL is below
  replay on both (by 0.8 and 1.2 points).
- `aglen`: every method is at 47 to 48%, below 50%. Nothing transfers to antigens that were not
  trained on, so the 0.7-point spread is among models that all fail. Replay costs 2 points on
  the trained tasks of this stream (section 1) and gains 0.7 on the held-out one.
- `skew:0.9`: the numbers mostly reflect the binder rate of the last trained task. Fine-tuning
  scores 89% when the held-out task has the same binder rate as the last trained one (held-out
  task 1) and 19 to 27% otherwise; replay and PDCL pull predictions towards the pooled rate, which
  costs 23 to 24 points on task 1 and gains 15 to 31 on the others (appendix T14). Joint training
  is best (54.0). This is a label-prior effect, not evidence about generalisation.
- With the last checkpoint instead of the validation-loss one, PDCL − ER is −3.58 ± 0.43 on
  `skew:0.9` and a null on `aglen` (−0.33 ± 0.23).

Figure: `paper/figures/fig_e2_heldout.png`.

## 6. Predictions, one by one

| Pre-registered statement | Outcome |
|---|---|
| Question: is one fixed w within 0.5 points of PDCL on every stream? | **No.** Each w is 2.3 to 5.6 points below PDCL on some stream. |
| "If yes, PDCL's adaptivity saves a search and nothing else." | Condition not met. But tuned replay ≥ PDCL on every stream (up to +2.23 ± 0.35), so the adaptivity buys no accuracy over a tuned weight. |
| Prediction: no fixed w is best at both ends of the dial. | **Held** (best w 0.01 at `skew:0.6`, 3 at `skew:0.95`). |
| Prediction: a middle w may still be close everywhere. | **Failed** because of `aglen` (w=0.1: −2.31 ± 0.40). Holds on the skew dial alone. |
| Secondary: replay hurts on held-out tasks and PDCL avoids that. | **Not found.** Replay helps on held-out tasks of both streams; PDCL is 0.8 to 1.2 below replay. |

## 7. Limits

- Five streams, ten seeds, one network, 1,000 steps per task; three of the streams are a
  controlled label-skew dial, and `aglen` is the only natural stream on which replay hurts.
- The "0.5 points" criterion is applied to means; the per-stream standard errors are 0.15 to 0.5,
  so differences near 0.5 are not resolved. The no-answers above rest on differences of 2.3 points
  and more.
- Fixed-weight replay uses one weight for all earlier tasks; PDCL has one per task. A per-task
  fixed weight was not tried.
- Section 3's explanation of why PDCL under-weights replay is read off logs, not tested.

## Appendix: all tables

### T1. Final accuracy (%)

| Method | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| Joint ERM | 70.10 ± 0.28 | 69.70 ± 0.41 | 69.62 ± 0.33 | 69.64 ± 0.40 | 69.83 ± 0.25 |
| Fine-tuning | 66.54 ± 0.56 | 67.97 ± 0.41 | 64.27 ± 0.50 | 52.57 ± 0.32 | 68.00 ± 0.44 |
| ER w=0.01 | 66.94 ± 0.56 | 67.95 ± 0.38 | 64.55 ± 0.48 | 54.13 ± 0.43 | 68.35 ± 0.33 |
| ER w=0.03 | 67.31 ± 0.49 | 67.95 ± 0.37 | 64.83 ± 0.47 | 57.07 ± 0.66 | 67.31 ± 0.36 |
| ER w=0.1 | 67.13 ± 0.32 | 67.69 ± 0.47 | 65.15 ± 0.57 | 60.57 ± 0.45 | 65.99 ± 0.48 |
| ER w=0.3 | 67.07 ± 0.44 | 67.39 ± 0.39 | 65.19 ± 0.41 | 61.51 ± 0.60 | 65.71 ± 0.41 |
| ER w=1 | 66.43 ± 0.60 | 67.23 ± 0.37 | 65.58 ± 0.34 | 61.83 ± 0.52 | 65.90 ± 0.49 |
| ER w=3 | 66.00 ± 0.38 | 66.67 ± 0.34 | 65.49 ± 0.31 | 62.01 ± 0.50 | 65.29 ± 0.40 |
| PDCL | 66.84 ± 0.56 | 67.85 ± 0.41 | 64.45 ± 0.61 | 59.78 ± 0.46 | 68.30 ± 0.35 |
| PDCL, dual step 0.001 | 66.64 ± 0.48 | 67.89 ± 0.40 | 64.43 ± 0.59 | 61.07 ± 0.59 | 68.09 ± 0.41 |
| PDCL, dual step 0.1 | 66.94 ± 0.50 | 67.90 ± 0.49 | 64.56 ± 0.51 | 59.74 ± 0.72 | 68.36 ± 0.28 |
| PDCL, epsilon x1.0 | 66.78 ± 0.50 | 67.91 ± 0.38 | 64.20 ± 0.61 | 60.30 ± 0.49 | 68.04 ± 0.37 |
| PDCL, epsilon x1.5 | 66.53 ± 0.46 | 67.80 ± 0.37 | 64.65 ± 0.54 | 59.27 ± 0.57 | 67.78 ± 0.31 |
| PDCL, epsilon x2.0 | 66.66 ± 0.52 | 67.83 ± 0.40 | 64.45 ± 0.51 | 58.50 ± 0.66 | 67.84 ± 0.34 |
| ER w=1, memory 50 | 66.63 ± 0.46 | 67.01 ± 0.27 | 63.59 ± 0.55 | 57.78 ± 0.32 | 64.04 ± 0.41 |
| ER w=1, memory 1000 | 68.57 ± 0.35 | 67.54 ± 0.40 | 67.03 ± 0.29 | 66.38 ± 0.33 | 67.36 ± 0.34 |
| PDCL, memory 50 | 66.76 ± 0.44 | 67.92 ± 0.34 | 64.38 ± 0.57 | 55.97 ± 0.59 | 67.88 ± 0.25 |
| PDCL, memory 1000 | 67.55 ± 0.39 | 67.87 ± 0.39 | 65.97 ± 0.41 | 66.87 ± 0.32 | 68.06 ± 0.43 |

### T2. Forgetting (points)

| Method | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| Joint ERM | 0.43 ± 0.26 | 0.42 ± 0.29 | 1.29 ± 0.39 | 2.73 ± 1.34 | 0.04 ± 0.27 |
| Fine-tuning | 3.80 ± 0.67 | 2.87 ± 0.59 | 22.38 ± 0.63 | 56.51 ± 0.56 | 3.18 ± 0.55 |
| ER w=0.01 | 2.97 ± 0.64 | 2.96 ± 0.49 | 21.79 ± 0.68 | 54.35 ± 0.66 | 2.81 ± 0.31 |
| ER w=0.03 | 2.57 ± 0.62 | 2.70 ± 0.47 | 21.05 ± 0.70 | 50.35 ± 0.94 | 4.07 ± 0.34 |
| ER w=0.1 | 2.54 ± 0.43 | 2.54 ± 0.46 | 19.17 ± 1.07 | 43.37 ± 0.55 | 5.50 ± 0.52 |
| ER w=0.3 | 2.98 ± 0.33 | 2.02 ± 0.44 | 16.07 ± 0.98 | 37.49 ± 0.60 | 5.36 ± 0.42 |
| ER w=1 | 3.06 ± 0.55 | 1.78 ± 0.45 | 13.55 ± 0.80 | 33.69 ± 0.83 | 4.35 ± 0.44 |
| ER w=3 | 2.91 ± 0.44 | 1.57 ± 0.36 | 12.87 ± 0.66 | 32.48 ± 0.98 | 4.08 ± 0.41 |
| PDCL | 3.42 ± 0.60 | 2.90 ± 0.53 | 19.72 ± 1.19 | 38.17 ± 0.56 | 2.88 ± 0.33 |
| PDCL, dual step 0.001 | 3.57 ± 0.56 | 2.93 ± 0.37 | 18.61 ± 1.21 | 36.32 ± 0.68 | 3.09 ± 0.39 |
| PDCL, dual step 0.1 | 3.08 ± 0.53 | 2.68 ± 0.52 | 20.21 ± 0.73 | 40.54 ± 1.01 | 2.80 ± 0.35 |
| PDCL, epsilon x1.0 | 3.37 ± 0.45 | 2.67 ± 0.47 | 19.39 ± 1.22 | 37.33 ± 0.55 | 2.96 ± 0.31 |
| PDCL, epsilon x1.5 | 3.71 ± 0.60 | 2.83 ± 0.43 | 20.54 ± 0.83 | 40.60 ± 0.55 | 3.33 ± 0.42 |
| PDCL, epsilon x2.0 | 3.58 ± 0.61 | 2.79 ± 0.54 | 21.17 ± 0.64 | 42.69 ± 0.73 | 3.21 ± 0.45 |
| ER w=1, memory 50 | 2.67 ± 0.44 | 2.45 ± 0.43 | 19.15 ± 1.09 | 44.76 ± 0.42 | 7.40 ± 0.56 |
| ER w=1, memory 1000 | 1.35 ± 0.28 | 1.37 ± 0.23 | 8.77 ± 0.52 | 19.58 ± 0.59 | 2.36 ± 0.28 |
| PDCL, memory 50 | 3.54 ± 0.56 | 2.90 ± 0.46 | 21.33 ± 0.81 | 49.38 ± 0.76 | 3.28 ± 0.43 |
| PDCL, memory 1000 | 2.66 ± 0.49 | 2.82 ± 0.57 | 12.96 ± 1.03 | 15.60 ± 0.71 | 3.01 ± 0.47 |

### T3. Accuracy of each task at the end of its own phase (%)

| Method | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| Joint ERM | 69.44 ± 0.26 | 69.18 ± 0.29 | 69.80 ± 0.37 | 69.79 ± 0.99 | 69.41 ± 0.30 |
| Fine-tuning | 69.16 ± 0.30 | 69.50 ± 0.29 | 80.88 ± 0.15 | 94.96 ± 0.19 | 70.23 ± 0.32 |
| ER w=0.01 | 69.01 ± 0.20 | 69.57 ± 0.35 | 80.76 ± 0.17 | 94.89 ± 0.21 | 70.23 ± 0.30 |
| ER w=0.03 | 68.97 ± 0.26 | 69.36 ± 0.30 | 80.52 ± 0.21 | 94.83 ± 0.23 | 70.22 ± 0.33 |
| ER w=0.1 | 68.78 ± 0.21 | 69.08 ± 0.30 | 79.50 ± 0.29 | 93.10 ± 0.21 | 70.01 ± 0.29 |
| ER w=0.3 | 68.85 ± 0.21 | 68.57 ± 0.34 | 77.24 ± 0.40 | 89.63 ± 0.28 | 69.64 ± 0.30 |
| ER w=1 | 68.10 ± 0.21 | 68.39 ± 0.40 | 75.75 ± 0.36 | 87.10 ± 0.29 | 69.01 ± 0.29 |
| ER w=3 | 67.46 ± 0.23 | 67.69 ± 0.32 | 75.14 ± 0.39 | 86.37 ± 0.35 | 68.29 ± 0.33 |
| PDCL | 69.18 ± 0.26 | 69.40 ± 0.32 | 79.20 ± 0.43 | 88.40 ± 0.32 | 70.28 ± 0.32 |
| PDCL, dual step 0.001 | 69.19 ± 0.21 | 69.46 ± 0.30 | 78.39 ± 0.43 | 88.31 ± 0.27 | 70.26 ± 0.33 |
| PDCL, dual step 0.1 | 69.08 ± 0.20 | 69.34 ± 0.28 | 79.68 ± 0.34 | 90.15 ± 0.51 | 70.32 ± 0.32 |
| PDCL, epsilon x1.0 | 69.04 ± 0.23 | 69.34 ± 0.31 | 78.71 ± 0.50 | 88.30 ± 0.31 | 70.18 ± 0.33 |
| PDCL, epsilon x1.5 | 69.20 ± 0.26 | 69.39 ± 0.28 | 79.98 ± 0.26 | 89.72 ± 0.28 | 70.12 ± 0.35 |
| PDCL, epsilon x2.0 | 69.22 ± 0.29 | 69.37 ± 0.29 | 80.24 ± 0.17 | 90.52 ± 0.28 | 70.11 ± 0.35 |
| ER w=1, memory 50 | 68.30 ± 0.27 | 68.42 ± 0.24 | 77.93 ± 0.37 | 91.35 ± 0.29 | 69.59 ± 0.31 |
| ER w=1, memory 1000 | 68.47 ± 0.21 | 68.31 ± 0.34 | 73.59 ± 0.37 | 81.07 ± 0.55 | 68.85 ± 0.26 |
| PDCL, memory 50 | 69.20 ± 0.23 | 69.56 ± 0.28 | 80.24 ± 0.22 | 93.00 ± 0.25 | 70.19 ± 0.33 |
| PDCL, memory 1000 | 69.25 ± 0.24 | 69.35 ± 0.32 | 75.58 ± 0.61 | 78.57 ± 0.34 | 70.21 ± 0.35 |

### T4. Fixed-weight replay minus PDCL, final accuracy (paired)

| w | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| 0.01 | +0.10 ± 0.14 | +0.10 ± 0.15 | +0.10 ± 0.39 | **-5.64 ± 0.41** | +0.05 ± 0.22 |
| 0.03 | **+0.47 ± 0.19** | +0.10 ± 0.17 | +0.37 ± 0.25 | **-2.70 ± 0.43** | **-0.98 ± 0.27** |
| 0.1 | +0.28 ± 0.29 | -0.16 ± 0.28 | **+0.69 ± 0.33** | **+0.80 ± 0.35** | **-2.31 ± 0.40** |
| 0.3 | +0.23 ± 0.47 | -0.46 ± 0.39 | +0.74 ± 0.38 | **+1.73 ± 0.30** | **-2.59 ± 0.25** |
| 1 | -0.41 ± 0.36 | -0.62 ± 0.43 | **+1.13 ± 0.36** | **+2.05 ± 0.43** | **-2.40 ± 0.38** |
| 3 | **-0.84 ± 0.33** | **-1.18 ± 0.43** | +1.03 ± 0.52 | **+2.23 ± 0.35** | **-3.01 ± 0.30** |

### T5. Fixed-weight replay minus fine-tuning, final accuracy (paired)

| w | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| 0.01 | +0.40 ± 0.32 | -0.02 ± 0.13 | +0.28 ± 0.15 | **+1.56 ± 0.26** | +0.35 ± 0.29 |
| 0.03 | **+0.76 ± 0.21** | -0.02 ± 0.22 | **+0.55 ± 0.28** | **+4.50 ± 0.63** | -0.69 ± 0.40 |
| 0.1 | +0.58 ± 0.32 | -0.28 ± 0.31 | +0.87 ± 0.67 | **+8.00 ± 0.48** | **-2.01 ± 0.58** |
| 0.3 | +0.53 ± 0.47 | -0.58 ± 0.37 | +0.92 ± 0.59 | **+8.93 ± 0.54** | **-2.29 ± 0.48** |
| 1 | -0.11 ± 0.45 | -0.74 ± 0.40 | **+1.31 ± 0.50** | **+9.25 ± 0.54** | **-2.10 ± 0.52** |
| 3 | -0.55 ± 0.43 | **-1.30 ± 0.41** | **+1.21 ± 0.55** | **+9.43 ± 0.41** | **-2.71 ± 0.42** |
| PDCL | +0.30 ± 0.24 | -0.12 ± 0.13 | +0.18 ± 0.50 | **+7.20 ± 0.47** | +0.30 ± 0.28 |
| Joint ERM | **+3.56 ± 0.50** | **+1.73 ± 0.47** | **+5.34 ± 0.53** | **+17.07 ± 0.49** | **+1.83 ± 0.49** |

### T6. Fixed-weight replay minus PDCL, forgetting (paired; negative = forgets less)

| w | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| 0.01 | -0.45 ± 0.28 | +0.06 ± 0.25 | **+2.07 ± 0.96** | **+16.18 ± 0.80** | -0.07 ± 0.26 |
| 0.03 | **-0.86 ± 0.34** | -0.21 ± 0.24 | +1.32 ± 0.72 | **+12.18 ± 0.88** | **+1.19 ± 0.38** |
| 0.1 | -0.88 ± 0.47 | -0.36 ± 0.43 | -0.55 ± 0.39 | **+5.20 ± 0.58** | **+2.62 ± 0.60** |
| 0.3 | -0.45 ± 0.51 | -0.88 ± 0.58 | **-3.66 ± 0.49** | -0.67 ± 0.62 | **+2.48 ± 0.43** |
| 1 | -0.36 ± 0.39 | -1.12 ± 0.69 | **-6.17 ± 0.60** | **-4.47 ± 0.90** | **+1.48 ± 0.50** |
| 3 | -0.51 ± 0.43 | **-1.33 ± 0.61** | **-6.85 ± 0.71** | **-5.68 ± 0.89** | **+1.21 ± 0.47** |
| Fine-tuning | +0.37 ± 0.27 | -0.03 ± 0.29 | **+2.65 ± 1.12** | **+18.35 ± 0.82** | +0.30 ± 0.34 |

### T7. One fixed weight for every stream? ER(w) minus PDCL

| w | worst stream | difference there | largest gap (either sign) | difference there | never more than 0.5 below PDCL | within ±0.5 everywhere | mean over streams |
|---|---|---|---|---|---|---|---|
| 0.01 | skew:0.95 | **-5.64 ± 0.41** | skew:0.95 | **-5.64 ± 0.41** | no | no | -1.06 |
| 0.03 | skew:0.95 | **-2.70 ± 0.43** | skew:0.95 | **-2.70 ± 0.43** | no | no | -0.55 |
| 0.1 | aglen | **-2.31 ± 0.40** | aglen | **-2.31 ± 0.40** | no | no | -0.14 |
| 0.3 | aglen | **-2.59 ± 0.25** | aglen | **-2.59 ± 0.25** | no | no | -0.07 |
| 1 | aglen | **-2.40 ± 0.38** | aglen | **-2.40 ± 0.38** | no | no | -0.05 |
| 3 | aglen | **-3.01 ± 0.30** | aglen | **-3.01 ± 0.30** | no | no | -0.36 |

### T8. Oracle-tuned replay: the best fixed weight of each stream against PDCL

| Stream | best w | ER(best w) | PDCL | ER(best w) − PDCL | ER(best w) − fine-tuning | PDCL mean dual |
|---|---|---|---|---|---|---|
| rounds | 0.03 | 67.31 ± 0.49 | 66.84 ± 0.56 | **+0.47 ± 0.19** | **+0.76 ± 0.21** | 0.015 |
| skew:0.6 | 0.01 | 67.95 ± 0.38 | 67.85 ± 0.41 | +0.10 ± 0.15 | -0.02 ± 0.13 | 0.018 |
| skew:0.8 | 1 | 65.58 ± 0.34 | 64.45 ± 0.61 | **+1.13 ± 0.36** | **+1.31 ± 0.50** | 0.108 |
| skew:0.95 | 3 | 62.01 ± 0.50 | 59.78 ± 0.46 | **+2.23 ± 0.35** | **+9.43 ± 0.41** | 0.480 |
| aglen | 0.01 | 68.35 ± 0.33 | 68.30 ± 0.35 | +0.05 ± 0.22 | +0.35 ± 0.29 | 0.003 |

### T9. PDCL dual variables (the replay weight PDCL uses), default settings

| Stream | mean after phase, task 0 | mean after phase, task 1 | mean after phase, task 2 | final, task 0 | final, task 1 | final, task 2 | epsilon, task 0 | epsilon, task 1 | epsilon, task 2 | seeds with all final duals 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| rounds | 0.014 ± 0.002 | 0.012 ± 0.002 | 0.021 ± 0.005 | 0.004 ± 0.002 | 0.003 ± 0.002 | 0.013 ± 0.003 | 0.564 ± 0.002 | 0.550 ± 0.002 | 0.386 ± 0.004 | 0% |
| skew:0.6 | 0.012 ± 0.003 | 0.015 ± 0.002 | 0.027 ± 0.004 | 0.004 ± 0.002 | 0.004 ± 0.001 | 0.019 ± 0.003 | 0.560 ± 0.003 | 0.521 ± 0.003 | 0.478 ± 0.003 | 0% |
| skew:0.8 | 0.154 ± 0.005 | 0.070 ± 0.003 | 0.100 ± 0.006 | 0.012 ± 0.004 | 0.000 ± 0.000 | 0.035 ± 0.003 | 0.411 ± 0.003 | 0.422 ± 0.005 | 0.322 ± 0.002 | 0% |
| skew:0.95 | 0.769 ± 0.035 | 0.111 ± 0.005 | 0.560 ± 0.031 | 0.040 ± 0.007 | 0.000 ± 0.000 | 0.365 ± 0.035 | 0.172 ± 0.003 | 0.358 ± 0.008 | 0.111 ± 0.003 | 0% |
| aglen | 0.001 ± 0.000 | 0.002 ± 0.001 | 0.004 ± 0.001 | 0.002 ± 0.001 | 0.001 ± 0.000 | 0.002 ± 0.001 | 0.501 ± 0.003 | 0.572 ± 0.003 | 0.537 ± 0.002 | 0% |

### T10. PDCL sensitivity: variant minus default PDCL (final accuracy unless noted)

| Variant | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| dual step 0.001 | -0.20 ± 0.14 | +0.05 ± 0.17 | -0.02 ± 0.13 | **+1.30 ± 0.38** | -0.20 ± 0.22 |
| dual step 0.1 | +0.10 ± 0.15 | +0.05 ± 0.15 | +0.11 ± 0.67 | -0.04 ± 0.37 | +0.06 ± 0.17 |
| epsilon x1.0 | -0.06 ± 0.22 | +0.06 ± 0.11 | -0.26 ± 0.15 | **+0.52 ± 0.20** | -0.26 ± 0.24 |
| epsilon x1.5 | -0.31 ± 0.22 | -0.05 ± 0.11 | +0.19 ± 0.31 | -0.51 ± 0.29 | **-0.52 ± 0.22** |
| epsilon x2.0 | -0.18 ± 0.15 | -0.02 ± 0.15 | -0.00 ± 0.54 | **-1.27 ± 0.36** | **-0.46 ± 0.19** |
| dual step 0.001, forgetting | +0.15 ± 0.16 | +0.03 ± 0.27 | **-1.11 ± 0.24** | **-1.85 ± 0.75** | +0.21 ± 0.24 |
| dual step 0.1, forgetting | -0.34 ± 0.19 | -0.22 ± 0.22 | +0.49 ± 1.22 | **+2.38 ± 0.82** | -0.08 ± 0.22 |
| epsilon x1.0, forgetting | -0.05 ± 0.41 | -0.23 ± 0.13 | -0.34 ± 0.34 | **-0.83 ± 0.35** | +0.08 ± 0.31 |
| epsilon x1.5, forgetting | +0.29 ± 0.24 | -0.07 ± 0.19 | +0.81 ± 0.66 | **+2.44 ± 0.64** | +0.45 ± 0.27 |
| epsilon x2.0, forgetting | +0.15 ± 0.21 | -0.11 ± 0.22 | +1.45 ± 1.09 | **+4.52 ± 0.77** | +0.33 ± 0.28 |
| default, mean dual | 0.015 | 0.018 | 0.108 | 0.480 | 0.003 |
| dual step 0.001, mean dual | 0.014 | 0.016 | 0.109 | 0.288 | 0.003 |
| dual step 0.1, mean dual | 0.017 | 0.022 | 0.092 | 0.443 | 0.003 |
| epsilon x1.0, mean dual | 0.027 | 0.031 | 0.125 | 0.520 | 0.006 |
| epsilon x1.5, mean dual | 0.001 | 0.001 | 0.063 | 0.346 | 0.000 |
| epsilon x2.0, mean dual | 0.000 | 0.000 | 0.031 | 0.238 | 0.000 |
| default, minus fine-tuning | +0.30 ± 0.24 | -0.12 ± 0.13 | +0.18 ± 0.50 | **+7.20 ± 0.47** | +0.30 ± 0.28 |
| dual step 0.001, minus fine-tuning | +0.10 ± 0.22 | -0.08 ± 0.23 | +0.16 ± 0.53 | **+8.50 ± 0.56** | +0.09 ± 0.18 |
| dual step 0.1, minus fine-tuning | +0.40 ± 0.28 | -0.08 ± 0.23 | +0.29 ± 0.43 | **+7.17 ± 0.61** | +0.36 ± 0.27 |
| epsilon x1.0, minus fine-tuning | +0.24 ± 0.29 | -0.07 ± 0.19 | -0.08 ± 0.45 | **+7.73 ± 0.41** | +0.04 ± 0.31 |
| epsilon x1.5, minus fine-tuning | -0.02 ± 0.16 | -0.17 ± 0.11 | +0.38 ± 0.37 | **+6.70 ± 0.45** | -0.22 ± 0.21 |
| epsilon x2.0, minus fine-tuning | +0.12 ± 0.18 | -0.15 ± 0.13 | +0.18 ± 0.16 | **+5.93 ± 0.59** | -0.16 ± 0.17 |

### T11. Memory size

| Quantity | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen |
|---|---|---|---|---|---|
| ER w=1, memory 50, final acc. | 66.63 ± 0.46 | 67.01 ± 0.27 | 63.59 ± 0.55 | 57.78 ± 0.32 | 64.04 ± 0.41 |
| PDCL, memory 50, final acc. | 66.76 ± 0.44 | 67.92 ± 0.34 | 64.38 ± 0.57 | 55.97 ± 0.59 | 67.88 ± 0.25 |
| ER w=1, memory 200, final acc. | 66.43 ± 0.60 | 67.23 ± 0.37 | 65.58 ± 0.34 | 61.83 ± 0.52 | 65.90 ± 0.49 |
| PDCL, memory 200, final acc. | 66.84 ± 0.56 | 67.85 ± 0.41 | 64.45 ± 0.61 | 59.78 ± 0.46 | 68.30 ± 0.35 |
| ER w=1, memory 1000, final acc. | 68.57 ± 0.35 | 67.54 ± 0.40 | 67.03 ± 0.29 | 66.38 ± 0.33 | 67.36 ± 0.34 |
| PDCL, memory 1000, final acc. | 67.55 ± 0.39 | 67.87 ± 0.39 | 65.97 ± 0.41 | 66.87 ± 0.32 | 68.06 ± 0.43 |
| ER − fine-tuning, memory 50 | +0.09 ± 0.32 | **-0.97 ± 0.30** | -0.68 ± 0.45 | **+5.21 ± 0.44** | **-3.96 ± 0.69** |
| ER − fine-tuning, memory 200 | -0.11 ± 0.45 | -0.74 ± 0.40 | **+1.31 ± 0.50** | **+9.25 ± 0.54** | **-2.10 ± 0.52** |
| ER − fine-tuning, memory 1000 | **+2.03 ± 0.55** | -0.43 ± 0.31 | **+2.75 ± 0.43** | **+13.81 ± 0.42** | -0.64 ± 0.42 |
| PDCL − fine-tuning, memory 50 | +0.22 ± 0.21 | -0.05 ± 0.16 | +0.10 ± 0.13 | **+3.40 ± 0.44** | -0.12 ± 0.23 |
| PDCL − fine-tuning, memory 200 | +0.30 ± 0.24 | -0.12 ± 0.13 | +0.18 ± 0.50 | **+7.20 ± 0.47** | +0.30 ± 0.28 |
| PDCL − fine-tuning, memory 1000 | **+1.01 ± 0.33** | -0.10 ± 0.20 | **+1.69 ± 0.44** | **+14.30 ± 0.40** | +0.06 ± 0.24 |
| PDCL − ER, memory 50 | +0.13 ± 0.26 | **+0.91 ± 0.25** | +0.78 ± 0.45 | **-1.81 ± 0.46** | **+3.84 ± 0.55** |
| PDCL − ER, memory 200 | +0.41 ± 0.36 | +0.62 ± 0.43 | **-1.13 ± 0.36** | **-2.05 ± 0.43** | **+2.40 ± 0.38** |
| PDCL − ER, memory 1000 | **-1.02 ± 0.48** | +0.33 ± 0.29 | **-1.06 ± 0.29** | +0.49 ± 0.25 | **+0.70 ± 0.30** |
| Replay (ER): memory 50 − memory 200 | +0.20 ± 0.31 | -0.22 ± 0.35 | **-1.99 ± 0.41** | **-4.04 ± 0.51** | **-1.86 ± 0.46** |
| Replay (ER): memory 1000 − memory 200 | **+2.14 ± 0.41** | +0.31 ± 0.25 | **+1.44 ± 0.28** | **+4.55 ± 0.47** | **+1.47 ± 0.39** |
| PDCL: memory 50 − memory 200 | -0.08 ± 0.19 | +0.07 ± 0.14 | -0.08 ± 0.45 | **-3.81 ± 0.38** | **-0.42 ± 0.19** |
| PDCL: memory 1000 − memory 200 | +0.71 ± 0.36 | +0.02 ± 0.21 | **+1.51 ± 0.36** | **+7.09 ± 0.36** | -0.24 ± 0.18 |
| ER forgetting, memory 50 | 2.67 ± 0.44 | 2.45 ± 0.43 | 19.15 ± 1.09 | 44.76 ± 0.42 | 7.40 ± 0.56 |
| ER forgetting, memory 200 | 3.06 ± 0.55 | 1.78 ± 0.45 | 13.55 ± 0.80 | 33.69 ± 0.83 | 4.35 ± 0.44 |
| ER forgetting, memory 1000 | 1.35 ± 0.28 | 1.37 ± 0.23 | 8.77 ± 0.52 | 19.58 ± 0.59 | 2.36 ± 0.28 |
| PDCL forgetting, memory 50 | 3.54 ± 0.56 | 2.90 ± 0.46 | 21.33 ± 0.81 | 49.38 ± 0.76 | 3.28 ± 0.43 |
| PDCL forgetting, memory 200 | 3.42 ± 0.60 | 2.90 ± 0.53 | 19.72 ± 1.19 | 38.17 ± 0.56 | 2.88 ± 0.33 |
| PDCL forgetting, memory 1000 | 2.66 ± 0.49 | 2.82 ± 0.57 | 12.96 ± 1.03 | 15.60 ± 0.71 | 3.01 ± 0.47 |
| PDCL mean dual, memory 50 | 0.005 | 0.005 | 0.028 | 0.245 | 0.001 |
| PDCL mean dual, memory 200 | 0.015 | 0.018 | 0.108 | 0.480 | 0.003 |
| PDCL mean dual, memory 1000 | 0.049 | 0.056 | 0.297 | 0.867 | 0.003 |

### T12. Held-out task accuracy (%), leave one task out, 3,000 steps

| Method | aglen, val.-loss checkpoint | skew:0.9, val.-loss checkpoint | aglen, last step | skew:0.9, last step |
|---|---|---|---|---|
| Joint ERM | 47.39 ± 0.47 | 54.02 ± 0.28 | 47.66 ± 0.44 | 56.55 ± 0.63 |
| Fine-tuning | 47.53 ± 0.33 | 39.66 ± 0.73 | 47.62 ± 0.39 | 39.90 ± 0.44 |
| Replay (ER) | 48.19 ± 0.32 | 52.40 ± 0.43 | 47.97 ± 0.28 | 48.74 ± 0.59 |
| PDCL | 47.38 ± 0.35 | 51.18 ± 0.43 | 47.63 ± 0.37 | 45.15 ± 0.54 |

### T13. Held-out task accuracy, paired differences

| Difference | aglen, val.-loss checkpoint | skew:0.9, val.-loss checkpoint | aglen, last step | skew:0.9, last step |
|---|---|---|---|---|
| Replay (ER) − Fine-tuning | **+0.66 ± 0.15** | **+12.74 ± 0.94** | +0.35 ± 0.31 | **+8.84 ± 0.76** |
| PDCL − Fine-tuning | -0.15 ± 0.13 | **+11.52 ± 0.82** | +0.02 ± 0.16 | **+5.26 ± 0.62** |
| PDCL − Replay (ER) | **-0.81 ± 0.16** | **-1.22 ± 0.47** | -0.33 ± 0.23 | **-3.58 ± 0.43** |
| Replay (ER) − Joint ERM | **+0.80 ± 0.35** | **-1.61 ± 0.46** | +0.31 ± 0.27 | **-7.81 ± 0.61** |
| PDCL − Joint ERM | -0.02 ± 0.23 | **-2.84 ± 0.39** | -0.03 ± 0.29 | **-11.40 ± 0.86** |
| Fine-tuning − Joint ERM | +0.13 ± 0.32 | **-14.36 ± 0.73** | -0.04 ± 0.39 | **-16.65 ± 0.89** |

### T14. Held-out accuracy by held-out task

| Stream | held-out task | Joint ERM | Fine-tuning | Replay (ER) | PDCL | ER − fine-tuning | PDCL − fine-tuning | PDCL − ER |
|---|---|---|---|---|---|---|---|---|
| aglen | 0 | 49.12 ± 1.06 | 47.90 ± 0.83 | 49.39 ± 0.81 | 47.55 ± 0.78 | +1.48 ± 0.86 | -0.35 ± 0.51 | **-1.83 ± 0.61** |
| aglen | 1 | 48.34 ± 0.57 | 48.25 ± 0.54 | 49.17 ± 0.56 | 48.23 ± 0.56 | **+0.91 ± 0.23** | -0.03 ± 0.19 | **-0.94 ± 0.32** |
| aglen | 2 | 43.31 ± 0.64 | 44.20 ± 0.47 | 43.73 ± 0.44 | 43.93 ± 0.40 | -0.47 ± 0.38 | -0.27 ± 0.19 | +0.20 ± 0.29 |
| aglen | 3 | 48.80 ± 0.69 | 49.76 ± 0.34 | 50.48 ± 0.43 | 49.80 ± 0.45 | +0.72 ± 0.54 | +0.05 ± 0.32 | -0.68 ± 0.61 |
| skew:0.9 | 0 | 54.51 ± 0.73 | 23.52 ± 1.42 | 51.10 ± 0.85 | 49.27 ± 1.08 | **+27.57 ± 1.77** | **+25.74 ± 1.71** | **-1.83 ± 0.75** |
| skew:0.9 | 1 | 54.13 ± 0.71 | 89.44 ± 0.20 | 64.99 ± 1.03 | 66.73 ± 1.24 | **-24.44 ± 0.99** | **-22.70 ± 1.20** | +1.74 ± 1.20 |
| skew:0.9 | 2 | 53.22 ± 0.66 | 26.83 ± 1.63 | 43.37 ± 1.02 | 41.65 ± 0.66 | **+16.54 ± 1.33** | **+14.81 ± 1.57** | **-1.73 ± 0.86** |
| skew:0.9 | 3 | 54.20 ± 0.65 | 18.84 ± 1.04 | 50.15 ± 1.13 | 47.07 ± 0.73 | **+31.31 ± 1.61** | **+28.23 ± 1.48** | **-3.07 ± 1.45** |

### T15. PDCL dual variables in the leave-one-out runs

| Stream | task | mean after phase | final |
|---|---|---|---|
| aglen | 0 | 0.002 ± 0.001 | 0.003 ± 0.001 |
| aglen | 1 | 0.002 ± 0.001 | 0.002 ± 0.001 |
| skew:0.9 | 0 | 0.477 ± 0.007 | 0.039 ± 0.004 |
| skew:0.9 | 1 | 0.249 ± 0.006 | 0.092 ± 0.005 |

### T16 (exploratory). Shortfall from the best of fine-tuning, the six fixed weights and PDCL on each stream

| Method | rounds | skew:0.6 | skew:0.8 | skew:0.95 | aglen | worst stream | mean over streams |
|---|---|---|---|---|---|---|---|
| Fine-tuning | **-0.76 ± 0.21** | best | **-1.31 ± 0.50** | **-9.43 ± 0.41** | -0.35 ± 0.29 | -9.43 | -2.37 |
| ER w=0.01 | -0.37 ± 0.25 | -0.02 ± 0.13 | **-1.03 ± 0.43** | **-7.87 ± 0.40** | best | -7.87 | -1.86 |
| ER w=0.03 | best | -0.02 ± 0.22 | **-0.75 ± 0.32** | **-4.93 ± 0.53** | **-1.03 ± 0.16** | -4.93 | -1.35 |
| ER w=0.1 | -0.18 ± 0.27 | -0.28 ± 0.31 | -0.44 ± 0.31 | **-1.43 ± 0.42** | **-2.35 ± 0.32** | -2.35 | -0.94 |
| ER w=0.3 | -0.23 ± 0.47 | -0.58 ± 0.37 | **-0.39 ± 0.18** | **-0.50 ± 0.20** | **-2.64 ± 0.26** | -2.64 | -0.87 |
| ER w=1 | **-0.88 ± 0.43** | -0.74 ± 0.40 | best | -0.18 ± 0.24 | **-2.45 ± 0.32** | -2.45 | -0.85 |
| ER w=3 | **-1.31 ± 0.32** | **-1.30 ± 0.41** | -0.10 ± 0.26 | best | **-3.06 ± 0.27** | -3.06 | -1.15 |
| PDCL | **-0.47 ± 0.19** | -0.12 ± 0.13 | **-1.13 ± 0.36** | **-2.23 ± 0.35** | -0.05 ± 0.22 | -2.23 | -0.80 |

### T17 (exploratory). Loss on the stored examples at the end of the stream, the constraint level, and final validation accuracy per task

| Stream | Method | memory loss, task 0 | memory loss, task 1 | memory loss, task 2 | epsilon, task 0 | epsilon, task 1 | epsilon, task 2 | val. acc., task 0 | val. acc., task 1 | val. acc., task 2 | val. acc., task 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rounds | ER w=1 | 0.015 | 0.018 | 0.020 | 0.564 | 0.547 | 0.312 | 65.6 | 65.4 | 68.1 | 66.6 |
| rounds | PDCL | 0.544 | 0.514 | 0.384 | 0.564 | 0.550 | 0.386 | 61.9 | 67.5 | 69.7 | 68.2 |
| skew:0.6 | ER w=1 | 0.016 | 0.017 | 0.026 | 0.560 | 0.537 | 0.450 | 65.0 | 68.3 | 66.3 | 69.4 |
| skew:0.6 | PDCL | 0.515 | 0.503 | 0.477 | 0.560 | 0.521 | 0.478 | 65.5 | 70.1 | 65.2 | 70.7 |
| skew:0.8 | ER w=1 | 0.017 | 0.012 | 0.026 | 0.411 | 0.458 | 0.340 | 55.7 | 75.7 | 55.9 | 75.0 |
| skew:0.8 | PDCL | 0.402 | 0.299 | 0.315 | 0.411 | 0.422 | 0.322 | 49.2 | 79.1 | 49.9 | 79.6 |
| skew:0.95 | ER w=1 | 0.017 | 0.003 | 0.022 | 0.172 | 0.312 | 0.141 | 36.4 | 87.6 | 35.4 | 87.9 |
| skew:0.95 | PDCL | 0.162 | 0.078 | 0.052 | 0.172 | 0.358 | 0.111 | 29.3 | 90.7 | 28.5 | 90.6 |
| aglen | ER w=1 | 0.007 | 0.012 | 0.022 | 0.501 | 0.548 | 0.467 | 70.3 | 62.1 | 64.2 | 67.0 |
| aglen | PDCL | 0.447 | 0.551 | 0.524 | 0.501 | 0.572 | 0.537 | 73.0 | 64.7 | 67.0 | 68.5 |
