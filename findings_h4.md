# H4. Sample-level dual variables

Branch `claude/pdcl-h4-sample-duals-b71bb7`. Design: `findings.md`, section 0c, H4,
followed without change. 360 runs under `outputs/h4` (spec `results/h4_sweeps.json`), ten
seeds per setting, 4,000 steps (1,000 per task). Every number below is written by
`python -m domainbed.scripts.h4_analysis` from the raw runs into `results/h4_tables.md`
and `results/h4_runs.csv`. "a ± b" is mean ± standard error over seeds, differences are
paired by seed, and a difference is called real only above two standard errors.

## Answer

**Not a null: choosing stored examples by their dual variables is worse than choosing
them at random in all 16 cells**, by 1.6 to 15.7 points of final accuracy, every cell
beyond four standard errors. The pre-registered prediction (a null in at least 12 of 16
cells) **fails**: 0 nulls, 0 wins, 16 losses. On `agrate`, `aglen` and `rounds` the
memory selected this way makes PDCL and replay equal to or worse than fine-tuning
without any memory.

## The rule, and how far it was checked against the paper

Source: Elenter et al. 2023, arXiv 2310.00154v2, HTML version, Section 4 ("sample
level") read in full on 2026-11-21; the appendices (proofs, experimental settings,
hyperparameters) were **not** read.

What the paper states:

1. *Constraints.* One per stored example of every earlier task: loss(x, y) ≤ ε(x, y),
   in place of the task-level constraint (mean loss of the task's stored examples ≤ ε_k).
2. *Dual update*, every training step, in the sample-wise version of its Algorithm 1:
   λ(x, y) ← [λ(x, y) + η_d (loss(x, y) − ε(x, y))]₊. The duals "accumulate the
   sample-wise constraint slacks over the entire learning procedure".
3. *Which examples are stored.* The task-level rule first sets how many examples each
   task gets (its Eq. PB, already in this repo). Within a task, the examples are drawn
   **without replacement with probability proportional to their dual variables**
   (λ_k / ‖λ_k‖₁). There is no top-k rule, no removal of outliers and no class
   balancing. The paper itself warns that large duals "can correspond to both outliers
   and inliers" and reports one dataset where sample selection does not help.
4. *Justification.* Corollary 1: −λ*(x, y) is a (sub)gradient of the optimal value with
   respect to ε(x, y), so a large dual marks an example whose constraint is costly to
   satisfy; examples with λ = 0 are called uninformative.

What is implemented (`train_batched.py`, hparam `selection: "dual"`, off by default;
default PDCL_RS and ER runs are bit-identical before and after the change):

- One dual variable per example and run, starting at 0, with the update of point 2 and
  step size `dual_lr` (0.01), applied every step to every example in the batch.
- Stored examples of earlier tasks: ε is the task's constraint level ε_k of this repo
  (1.1 × the training loss at the end of the task's phase).
- Selection as in point 3, both when a task's candidates (the pool of 1,000) are first
  stored and when an earlier task's memory is cut down. If fewer examples have a
  positive dual than there are places, the rest is filled at random from the others.
- The replay loss is weighted as before (task-level duals for PDCL, weight 1 for ER),
  so the runs differ from their controls only in which examples are stored.

Fidelity: **points 2 and 3 match the paper's text; two choices are mine and
unverified.** (a) The paper's Section 4 does not say how examples of the *current* task
get a dual before they are first stored (at the task level it uses an "artificial"
dual). Here they follow the same update during their own phase, with ε = 1.1 × the mean
training loss of the last 200 steps, the level the task would be given at that moment.
(b) ε(x, y) is taken equal to the task's ε_k; the paper gives no formula in the part I
read. The paper's variant with sample selection (PDCL-S) also weights each stored
example's loss by its own dual; that is not done here, by design of H4.

Sanity check on a short run before the sweep: 10 to 35% of a task's candidates had a
positive dual (so the duals are not all equal), and the stored examples had a mean loss
of 1.0 to 1.7 against 0.3 to 0.4 for the candidates.

## Pre-registered prediction and its outcome

Prediction: "a null (within two standard errors) in at least 12 of the 16 cells; no
prediction of sign for the rest." Outcome: **failed**, 0 of 16 nulls.

| stream | memory | method | random | dual | dual − random, final acc. | dual − random, forgetting |
|---|---|---|---|---|---|---|
| `skew:0.9` | 50 | PDCL | 59.91 ± 0.56 | 58.35 ± 0.57 | **−1.57 ± 0.27** | **+3.28 ± 0.48** |
| `skew:0.9` | 50 | ER | 61.48 ± 0.68 | 59.37 ± 0.62 | **−2.11 ± 0.35** | **+5.37 ± 0.66** |
| `skew:0.9` | 200 | PDCL | 61.69 ± 0.53 | 59.66 ± 0.57 | **−2.03 ± 0.38** | **+4.45 ± 0.50** |
| `skew:0.9` | 200 | ER | 62.95 ± 0.52 | 60.09 ± 0.40 | **−2.86 ± 0.40** | **+4.90 ± 0.61** |
| `agrate` | 50 | PDCL | 55.96 ± 0.36 | 54.22 ± 0.22 | **−1.75 ± 0.29** | **+2.30 ± 0.39** |
| `agrate` | 50 | ER | 61.18 ± 0.47 | 54.35 ± 0.43 | **−6.83 ± 0.59** | **+8.25 ± 0.76** |
| `agrate` | 200 | PDCL | 59.82 ± 0.41 | 53.64 ± 0.29 | **−6.19 ± 0.37** | **+8.34 ± 0.52** |
| `agrate` | 200 | ER | 64.36 ± 0.37 | 52.15 ± 0.32 | **−12.21 ± 0.50** | **+16.23 ± 0.64** |
| `aglen` | 50 | PDCL | 67.64 ± 0.34 | 58.16 ± 0.48 | **−9.48 ± 0.50** | **+12.63 ± 0.76** |
| `aglen` | 50 | ER | 63.89 ± 0.36 | 52.86 ± 0.37 | **−11.04 ± 0.62** | **+14.76 ± 0.79** |
| `aglen` | 200 | PDCL | 68.23 ± 0.34 | 55.41 ± 0.33 | **−12.82 ± 0.40** | **+17.04 ± 0.68** |
| `aglen` | 200 | ER | 65.59 ± 0.41 | 49.89 ± 0.35 | **−15.71 ± 0.33** | **+20.85 ± 0.62** |
| `rounds` | 50 | PDCL | 66.83 ± 0.52 | 65.14 ± 0.43 | **−1.70 ± 0.19** | **+1.80 ± 0.23** |
| `rounds` | 50 | ER | 66.91 ± 0.38 | 62.40 ± 0.39 | **−4.50 ± 0.38** | **+4.04 ± 0.61** |
| `rounds` | 200 | PDCL | 67.11 ± 0.49 | 63.53 ± 0.24 | **−3.58 ± 0.34** | **+4.04 ± 0.42** |
| `rounds` | 200 | ER | 66.48 ± 0.46 | 60.29 ± 0.37 | **−6.19 ± 0.42** | **+5.77 ± 0.53** |

(Table 1 of `results/h4_tables.md`; figure `paper/figures/fig_h4_selection`, panels a, b.)
Fine-tuning for reference: 57.42 (`skew:0.9`), 54.05 (`agrate`), 68.00 (`aglen`), 66.54
(`rounds`); each setting against fine-tuning is in table 2.

Three patterns in the table, none predicted:

- The loss is in forgetting, not in learning: accuracy at the end of each task's own
  phase moves by at most 2 points, forgetting rises by 2 to 21.
- In all 8 pairs a memory of 200 loses more than a memory of 50, and in all 8 pairs ER
  loses more than PDCL. More weight on the selected examples means more harm.
- With selection by duals, PDCL and ER end below fine-tuning beyond two standard errors
  in 10 of the 12 cells on `agrate`, `aglen` and `rounds` (down to −18.1 on `aglen`) and
  equal to it in the other two; only on `skew:0.9` does a gain over fine-tuning remain
  (+0.9 to +2.7).

## What the dual variables select

Per task, 16 to 41% of the candidates end their phase with a positive dual, and the
memory is drawn from those (table 3, figure panels c, d):

| stream | misclassified at selection, candidates | stored | loss, candidates | stored | minority class, candidates | stored |
|---|---|---|---|---|---|---|
| `skew:0.9` | 8% | 46–89% | 0.22 | 0.9–1.8 | 10% | 49–89% |
| `agrate` | 26% | 80–89% | 0.52 | 1.1 | 35% | 80–88% |
| `aglen` | 22% | 73–85% | 0.46 | 1.1–1.2 | 46% | 49–52% |
| `rounds` | 20% | 68–79% | 0.42 | 1.0–1.1 | 43% | 58–62% |

The stored examples are mostly the ones the model gets wrong at the end of the task's
phase. On `agrate`, `aglen` and `rounds` the validation accuracy of a task is about 70% even
right after training on it (87% on `skew:0.9`), so these are not rare informative cases but the part of the task the
network cannot fit; a memory made of them is a sample of the task with the labels
effectively opposite to what the model learned, and replaying it pushes the earlier
task's predictions the wrong way. This reading is consistent with the three patterns
above but was not tested by a separate experiment. On `skew:0.9` the selection also
turns a 90/10 memory into one dominated by the minority class, which could have helped
there; it does not.

## Limits

- The two unverified choices above (duals of current-task examples, ε per example).
  A different convention for the current task, for example duals that only start once
  the example is in memory, would select differently; it was not run.
- Selection only. The paper's full sample-level variant also weights each stored
  example by its dual, and was tested on image and audio benchmarks with near-perfect
  training accuracy, where a high-loss example is rarer than here.
- One training length (1,000 steps per task), one `dual_lr`, one `epsilon_scale`; each
  example is seen about 15 times in its phase, so a dual is the sum of few slacks.
- The explanation of the harm is an interpretation of table 3, not a tested claim.
