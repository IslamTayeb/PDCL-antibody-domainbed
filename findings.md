# Where does PDCL's idea help on Antibody DomainBed?

Branch `pdcl-scope`. The paper's results are untouched. Everything here is new: about
6,600 runs made with `domainbed/scripts/train_batched.py`, in twelve experiments whose
designs and predictions were written down before their runs existed (sections 0 to 0d,
kept verbatim at the end of this file). Each experiment has its own report
(`findings_e1.md` … `findings_r1.md`), tables (`results/*_tables.md`) and an analysis
script that recomputes every number from the raw runs.

Conventions: whole stream of four tasks, 1,000 steps per task, memory of 200 stored
examples, ten seeds, unless stated. Final accuracy is validation accuracy at the last
step averaged over the tasks. "a ± b" is mean ± standard error over seeds; differences
are paired by seed; a difference is called real only beyond two standard errors, and
everything else is reported as a null. PDCL is the published method (`PDCL_RS`); ER is
replay with weight 1 unless a weight is given.

## The story in one paragraph

On this data a model forgets only when the labels of a later task contradict those of an
earlier one, and the design rounds do not do that. So with a small memory there is
nothing for PDCL to protect, and it correctly does nothing. That restraint is PDCL's
real merit here: replay with 200 stored examples loses 2 to 2.5 points on streams with
new antigens, because it fits the stored examples and gets worse on the tasks they came
from, and PDCL avoids the loss without any tuning. The same mechanism is PDCL's limit.
It measures its constraints on those memorised examples, so when retention does become
expensive its dual variables stay too small and it ends 1 to 2 points below replay.
Measuring the constraints on unseen data removes that shortfall; no deployable variant we
tried does. PDCL never beats replay with a tuned weight, a larger memory or longer
training, its memory allocation is a null, and its sample selection, as we implemented
it, is harmful.

## 1. Why the rounds are easy

- **The rounds share what determines the label.** 97 to 100% of a later round's examples
  have an antigen that also occurs in round 1 or 3, and the binder rate of the rounds
  moves only between 0.39 and 0.59. Fine-tuning forgets 3.8 ± 0.7 points.
- **New antigens alone cause no extra forgetting; conflicting labels do** (E1,
  two-by-two, fine-tuning's forgetting in points):

  | | Binder rate stable | Binder rate shifting |
  |---|---|---|
  | Antigens shared | −0.3 (`skew:0.5`), 3.8 (`rounds`) | 44.0 (`skew:0.9`) |
  | Antigens new | 3.2 (`aglen`), 3.4 (`target`) | 22.1 (`agrate`) |

- **When only the binder rate shifts, the forgetting is a shifted output bias** (H2). A
  class-balanced loss with no memory recovers 99% to 295% of replay's gain on the six
  `skew` and `drift` streams where replay has one, and beats ER on five of them
  (`skew:0.9`: balanced +9.2 ± 0.6, ER +5.5 ± 0.4, PDCL +4.2 ± 0.3 over fine-tuning).
  When the shift comes with new antigens (`agrate`) balancing recovers 6% and ER is
  9.6 ± 0.5 points above it. `agrate` is the only stream here where a memory does
  something that a change of loss does not.
- **Nothing helps predict the next round** (G2, exploratory): at the end of each round's
  phase, accuracy on the next, unseen round is the same for PDCL and fine-tuning
  (+0.03 ± 0.11) and for ER and fine-tuning (+0.64 ± 0.61).

## 2. Where the benchmark sits on the dial

Swing = highest minus lowest binder rate among the tasks. Gain over fine-tuning in points
(E1, F3; figure `paper/figures/fig_f3_dial`):

| Swing | Alternating: ER | Alternating: PDCL | Gradual: ER | Gradual: PDCL |
|---|---|---|---|---|
| 0.20 (the real rounds) | +0.3 ± 0.4 | +0.1 ± 0.2 | not run | not run |
| 0.40 | +0.1 ± 0.6 | +0.2 ± 0.2 | **+1.5 ± 0.6** | **+0.6 ± 0.2** |
| 0.60 | +1.0 ± 0.6 | 0.0 ± 0.5 | **+2.4 ± 0.6** | **+1.6 ± 0.3** |
| 0.70 | **+3.3 ± 0.7** | **+2.5 ± 0.4** | not run | not run |
| 0.80 | **+5.4 ± 0.4** | **+4.1 ± 0.3** | **+3.9 ± 0.6** | **+3.0 ± 0.5** |
| 0.90 | **+8.9 ± 0.5** | **+7.4 ± 0.4** | not run | not run |

- The rounds have a swing of 0.20. On the alternating dial a memory of 200 starts to
  matter at 0.70, for ER and PDCL alike.
- Gradual drift is harder than alternation at equal swing: both methods already help at
  0.40, the smallest drift run. A gradual drift of 0.20 was not run, so the rounds are
  below the alternating threshold and of unknown position on the gradual one.
- With a small memory the benchmark is therefore at the easy end. It is not at the easy
  end for memory in general: replaying 1,000 stored examples gains +2.0 ± 0.5 on the
  rounds and replaying every earlier example +3.9 ± 0.5, which reaches joint training
  (70.4 against 70.1) (R1).

## 3. What PDCL is good for, and where it falls short

**Good for: not doing harm with a small memory, without tuning.**

- Default PDCL was compared with fine-tuning in 101 settings (streams, memory sizes,
  lengths, dual step, epsilon, alpha). It is more than two standard errors below in 2,
  the worst by 0.57 ± 0.28 points (`skew:0.6`, long runs, five seeds), and above in 51.
  Two in 101 is what chance gives; no correction was applied.
- With memory 200 it is 2.1 to 2.7 points above ER on the new-antigen streams `aglen` and
  `target` (E1, R1), because ER loses that much against fine-tuning and PDCL switches
  replay off (mean dual variable 0.003 on `aglen`).
- No fixed replay weight is that safe. Each of six weights from 0.01 to 3 is 2.3 to 5.6
  points below PDCL on some stream: small weights under strong skew, weights of 0.1 and
  more on `aglen` (E2).
- Its two settings hardly matter: dual step and epsilon move final accuracy by about
  half a point on four of five streams (E2).

**Falls short: it never beats replay that is tuned, larger or longer.**

- Replay with the best weight per stream is at or above PDCL on all five streams of E2,
  by +1.1 ± 0.4 on `skew:0.8` and +2.2 ± 0.4 on `skew:0.95` (best weight picked on the
  same seeds, so biased towards replay). On the curve of new-task against earlier-task
  accuracy traced by the fixed weights, PDCL is on the curve in 8 of 11 panels, below in
  2 and above in none (G1, exploratory).
- Where forgetting is large it gains less than ER: `skew:0.9` +4.1 against +5.4,
  `skew:0.95` +7.4 against +8.9, `agrate` +5.7 against +10.3. At `skew:0.8` fine-tuning
  forgets 22 points and PDCL gains nothing (0.0 ± 0.5).
- With a large memory its caution costs: at full memory PDCL is 1.1 ± 0.3 below ER on
  `aglen` and 1.9 ± 0.4 below on `rounds` (R1).
- At 5,000 steps per task PDCL equals ER w=1 on all four natural streams and is below
  the best replay weight on `aglen` (−0.9 ± 0.5) and `target` (−0.7 ± 0.3) (H1). Longer
  training does not improve any method: fine-tuning loses 5 points on `aglen` and
  `target`, and memory only repairs that loss.
- Memory allocation by the dual variables is a null in 8 of 8 cells (E3). The dual
  variables rank streams by retention cost (rank correlation 0.84 on stream means) but
  not tasks within a stream (0.23); these two correlations are the E3 thread's and were
  not recomputed by me.
- Choosing stored examples by sample-level dual variables is worse than random selection
  in 16 of 16 cells, by 1.6 to 15.7 points (H4). The rule follows Section 4 of the paper;
  two details are our choices and the paper's appendices were not read, so this is a
  result about our implementation on this data.
- On held-out tasks the paper's pattern does not carry over: ER is 0.8 and 1.2 points
  above PDCL on `aglen` and `skew:0.9` (E2, secondary).

## 4. One mechanism behind PDCL's timidity and replay's harm

A memory of 200 examples (about 66 per earlier task) is fitted completely: its loss
goes to about 0.01 and its accuracy to 100%. After that, the loss on the memory says
nothing about the task it came from.

**Replay's harm.**
- On `aglen` with memory 200, ER's validation loss on earlier tasks is 1.13 against 0.60
  for fine-tuning, while the stored examples are at 0.012 (R1, F2).
- The harm shrinks as the memory grows and turns into a gain: ER − fine-tuning on
  `aglen` is −2.4 ± 0.5, −0.7 ± 0.5, +0.1 ± 0.5, +1.6 ± 0.4 at memory 200, 1,000, 3,000
  and full; on `target` −1.5, +0.2, +1.2, +1.8 (R1, prediction held). So the cause is a
  stored sample too small to stand for its task, not interference from old antigens.
- A smaller part is interference: on the last task ER stays about 2 points below
  fine-tuning on `target` at every memory size (R1).
- Revisiting stored examples less often does not help: replaying 2 instead of 32 per
  task leaves the loss on `aglen` unchanged (−2.7 against −2.8), and the stored examples
  are still fitted to 99–100% (F2, prediction failed). Weight decay does not help either.

**PDCL's timidity.**
- On `skew:0.95` the loss on task 0's stored examples ends below its constraint level
  while validation accuracy on that task is 29%; the dual variable decays to zero (E2).
- An oracle that updates the dual variables from validation loss gains +1.3 ± 0.4 on
  `skew:0.9` and +2.4 ± 0.4 on `skew:0.95` over default PDCL, is never more than two
  standard errors below the best fixed weight, and stays at fine-tuning on `aglen`
  (+0.4 ± 0.3) (F1, prediction held in part; `skew:0.8` +1.0 ± 0.5 is a null).
- With memory 1,000 default PDCL already matches the best fixed weight on `skew:0.9` and
  `0.95`, and the oracle adds nothing (F1). The shortfall is a small-memory problem.
- No deployable fix was found. Holding back half of the memory for the dual update
  beats default PDCL in 1 of 12 cells and loses the safety on `aglen` (−2.0 ± 0.6)
  (F1). Starting the dual variables at 1 closes −13% to 25% of the gap and is 1.4 ± 0.4
  below fine-tuning on `aglen` (H3).
- On validation data no method satisfies PDCL's constraint more often than fine-tuning:
  the share of earlier tasks within 1.1 × their validation loss is 41% for fine-tuning,
  43% for PDCL and 20% for ER (G3, exploratory, F1 runs only).

**PDCL's safety claim, restated:** PDCL is safe when the memory is too small to trust.
With 200 examples its refusal to replay is right. With a memory large enough to stand
for the earlier tasks, the same refusal costs up to 2 points.

## 5. What rests on what

| Claim | Evidence | Status |
|---|---|---|
| Forgetting needs conflicting labels, not new antigens | E1 two-by-two | pre-registered streams; the two-by-two reading was made after |
| On pure label shift the forgetting is an output bias | H2 | pre-registered, 8 of 9 testable predictions held |
| Memory of 200 matters from swing 0.70 (alternating) | F3 | pre-registered; my predicted threshold of 0.60 for ER failed |
| Gradual drift is harder than alternating | F3 | pre-registered; I predicted the opposite |
| PDCL is never meaningfully below fine-tuning | E1, E2, E3, F1, H3, R1 | counted over all runs after the fact |
| No fixed weight is safe everywhere | E2 | pre-registered |
| Tuned replay ≥ PDCL | E2, F1, H1 | pre-registered comparison; best weight picked on the same seeds |
| Allocation is a null | E3 | pre-registered |
| Replay's harm is a too-small memory | R1 | pre-registered, held |
| PDCL under-reacts because of memorised constraints | F1 (oracle) | pre-registered, held in part |
| Sample-level selection is harmful | H4 | pre-registered null failed; fidelity to the paper unverified |
| Trade-off curve, next-round accuracy, validation constraints | G1–G3 | exploratory |

## 6. Every pre-registered prediction and its outcome

| Exp. | Prediction | Outcome |
|---|---|---|
| E1 | PDCL − fine-tuning is zero on `skew:0.5` and `rounds` | `rounds` held (+0.08 ± 0.24); `skew:0.5` failed by the rule (−0.17 ± 0.08) |
| E1 | PDCL's gain grows with fine-tuning's forgetting | Held as a rank trend (Spearman 0.72), but a threshold, not a dial |
| E1 | PDCL never more than 0.5 below the better of fine-tuning and ER | **Failed** on 4 of 11 streams (−0.9 to −4.7) |
| E2 | No fixed weight is best at both ends of the dial | Held |
| E2 | A middle weight may still be close to PDCL everywhere | **Failed** (`aglen`, −2.3 ± 0.4) |
| E2 | Replay hurts on held-out tasks and PDCL avoids it | **Not found**; ER above PDCL on both streams |
| E3 | Allocation weak or null at memory 200 | Held |
| E3 | Allocation possibly positive at memory 40 | **Null** (−0.10 ± 0.14, +0.06 ± 0.23) |
| E3 | Dual variables correlate with fine-tuning's forgetting | Held pooled (+0.48); weak within a stream |
| F1 | Oracle beats default PDCL on `skew:0.8`–`0.95` and closes half the gap | Held on `0.9`, `0.95`; null on `0.8` at memory 200 |
| F1 | Split variant positive but smaller | **Failed**: 1 of 12 cells, worse at memory 1,000 |
| F1 | Neither variant below fine-tuning on `aglen`, `rounds` | Oracle held; split **failed** on `aglen` |
| F1 | Shortfall smaller at memory 1,000 | Held |
| F2 | Replaying 2 per task removes half of ER's loss on `aglen` | **Failed** (loss unchanged) |
| F2 | ...while keeping half of the gain on `skew:0.9` | Held (nothing changes) |
| F3 | ER's alternating threshold is swing 0.60 | **Failed** (0.70) |
| F3 | PDCL's alternating threshold is 0.70 or 0.80 | Held (0.70) |
| F3 | Gradual drift costs fine-tuning less than alternation | **Failed** (2 to 4 points more at swing 0.40, 0.60) |
| F3 | Thresholds are higher under gradual drift | **Failed, reversed** (at most 0.40) |
| F3 | The rounds' swing is below every threshold | Held for alternating; open for gradual |
| H1 | ER w=1 beats fine-tuning on `aglen`, `target` at 5,000 steps | Held (+2.8 ± 0.5, +3.2 ± 0.5) |
| H1 | PDCL gains on at least one stream, less than ER | Held; smaller than ER w=1 in sign only |
| H1 | PDCL does not beat the best replay weight | Held |
| H2 | Balancing recovers at least half of ER's gain on `skew`, `drift` | Held on all six testable streams |
| H2 | Less than half on `agrate` | Held (6%) |
| H2 | No effect on `rounds`, `aglen` | `rounds` held; `aglen` **failed** (−1.0 ± 0.4) |
| H3 | Init 1 within two SE of fine-tuning on `aglen`, `rounds` | **Failed** on `aglen` at memory 200 (−1.4 ± 0.4) |
| H3 | Init 1 closes less than half the gap to ER | Held (−13% to 34%) |
| H4 | Null in at least 12 of 16 cells | **Failed**: worse than random in 16 of 16 |
| R1 | ER's cost gone by memory 1,000, ER above fine-tuning at full memory | Held on `aglen` and `target` |

## 7. Proposed story and figures for the paper

Keep the paper's three findings and add one section, "How far does the idea go?":

1. **Figure A, the dial** (`fig_f3_dial`): gain of ER and PDCL over fine-tuning against
   binder-rate swing, alternating and gradual, with the real rounds marked at 0.20.
2. **Figure B, safety against tuning** (`fig_e2_weight`): final accuracy against the
   fixed replay weight on five streams, with PDCL and fine-tuning as bands.
3. **Figure C, memory size** (`fig_r1_memory`): ER and PDCL minus fine-tuning against
   memory size on `aglen`, `target`, `rounds`.
4. **Figure D, honest constraints** (`fig_f1_differences`, `fig_f1_duals`): default PDCL,
   oracle and split against the best fixed weight.
5. **Table, bias against memory** (H2): ER, PDCL and balanced fine-tuning on the dial
   and on `agrate`.

Supporting: `fig_e1_gain`, `fig_e3_allocation`, `fig_h1_long`, `fig_h3_init`,
`fig_h4_selection`, `fig_g1_tradeoff`.

## 8. Limits

- The labels are the sign of a computed binding estimate, one dataset, one network.
- The `skew`, `drift` and `hetero` streams are constructed; `agrate` uses the labels to
  group antigens. Only `rounds`, `aglen` and `target` are natural.
- "Best fixed weight" is always picked on the seeds it is scored on.
- Two repeats of the same run on these shared GPUs differ slightly, so comparisons across
  experiments are within seed noise, not exact. Within an experiment all methods were
  rerun together.
- The long E1 sweep has five seeds; H1 repeats its natural streams with ten.
- G3 could be computed only where per-task validation loss was logged (F1, F2).
- The oracle in F1 uses validation data during training. It shows what honest
  measurement would buy; it is not a method.

---

# Designs as pre-registered (unchanged)

## 0. Designs, fixed before any PDCL number was seen

Written on 2026-11-21 before any run of this study. What I had seen by then: the paper,
and one sentence from a pilot of another session ("PDCL forgets 2 points less than
fine-tuning on the antigen grouping"). I had seen no other number on any stream below.

**Why the rounds are easy** (data profile, no training): 97–100% of a later round's
examples have an antigen that also occurs in round 1 or 3, and 68–97% share a seed
antibody with another round, while the designed chains themselves are disjoint. The
binder rate of an antigen ranges from 0 to 1. The input-to-label structure is carried by
antigen and seed, and every round has the same ones. So fine-tuning has little to forget.

**PDCL's idea, as three claims that can each fail:**

1. *Replay weight set by need.* The dual variable of an earlier task is zero while its
   constraint holds and grows when it is violated, so PDCL should act like fine-tuning
   when retention is cheap and like replay when it is expensive.
2. *The dual variable is a readout* of how costly a task is to retain.
3. *Memory follows the dual variables*, which should matter when tasks differ in cost and
   the memory is small.

**Streams** (`domainbed/streams.py`). Same 10,606 examples, regrouped into tasks:

| Stream | What changes between tasks | Natural? |
|---|---|---|
| `rounds`, `rounds:3210` | little (the paper's stream, and reversed) | yes |
| `aglen` | antigens (none shared), binder rate flat | yes: a campaign that moves to new antigens |
| `target` | HIV-1, then three groups of SARS-CoV-2 antigens (none shared) | yes |
| `skew:Q`, Q in 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 | binder rate only, alternating; Q=0.5 is a random split | controlled dial |
| `agrate` | antigens (none shared) and binder rate rising 0.22 to 0.86 | stress test, uses labels |
| `hetero` | binder rates 0.72, 0.52, 0.72, 0.11 | controlled, tasks of unequal cost |
| `skew:0.9:8` | eight tasks, binder rate alternating | controlled, longer stream |

**Common protocol.** Whole stream (`--envs -1`), default hyperparameters of the paper
(learning rate 5e-5, dual step 0.01, memory 200, epsilon 1.1 x training loss), 1,000
steps per task as the main length (the paper's non-overfit regime), ten seeds (0–9),
methods compared seed by seed. Outcome: **final accuracy** = validation accuracy at the
last step averaged over the tasks; also forgetting as defined in the paper. PDCL is
`PDCL_RS`, the published method. Joint ERM is the reference. A difference counts as real
when it exceeds two standard errors; everything else is reported as a null.

**E1. Retention dial.** Joint ERM, fine-tuning, ER, PDCL on `rounds`, `rounds:3210`,
`aglen`, `target`, `agrate` and the six `skew:Q`. Second length: 5,000 steps per task,
five seeds. Predictions: (a) at `skew:0.5` and `rounds` PDCL − fine-tuning is zero;
(b) the gain of PDCL over fine-tuning grows with how much fine-tuning forgets on the
stream; (c) PDCL is never more than 0.5 points below the better of fine-tuning and ER.
Open, no prediction: whether the antigen-disjoint natural streams (`aglen`, `target`)
forget enough to matter, and whether PDCL beats ER anywhere.

**E2. Is the adaptivity needed?** ER with a fixed replay weight w in 0.01, 0.03, 0.1,
0.3, 1, 3 (`replay_weight`) against PDCL on `rounds`, `aglen`, `skew:0.6`, `skew:0.8`,
`skew:0.95`. Question: is there one fixed w that is within 0.5 points of PDCL on every
stream? If yes, PDCL's adaptivity saves a search and nothing else. Also PDCL with dual
step 0.001 and 0.1 and epsilon scale 1.0, 1.5, 2.0 on the same streams, and memory 50
and 1,000 for ER and PDCL. Secondary, on held-out tasks (leave one task out, 3,000 steps,
checkpoint by validation loss as in the paper) for `aglen` and `skew:0.9`: fine-tuning,
ER, PDCL, joint ERM. Prediction: no fixed w is best at both ends of the dial; a middle w
may still be close everywhere.

**E3. Allocation and the dual readout.** PDCL with alpha in 0 (uniform memory), 0.5
(default), 1 on `hetero`, `skew:0.9:8`, `agrate`, `rounds`, with memory 40 and 200.
Outcome: final accuracy of alpha 0.5 minus alpha 0. Prediction: weak or null at memory
200, possibly positive at 40 on `hetero` and the eight-task stream. Readout: across
tasks and streams (all E1 and E3 streams), rank correlation between PDCL's dual variable
for a task (mean after the task's phase) and fine-tuning's forgetting of that task in
the same seed. Prediction: positive.

Anything run beyond these designs is marked *exploratory* below.

## 0b. Second round: designs F1–F3, fixed before any of their PDCL numbers were seen

Written on 2026-11-21 after the E1 main sweep and E2 were finished and read (E3 and the
long E1 sweep still training). What I had seen: PDCL equals fine-tuning up to `skew:0.8`
and gains 4 to 7 points at `skew:0.9` and `0.95`; ER w=1 is 1 to 2 points above PDCL
there and 2.4 below it on `aglen`; PDCL's dual variables decay while the loss on the
stored examples sits under the constraint level. No run of F1–F3 existed.

Protocol as in section 0 (whole stream, 1,000 steps per task, seeds 0–9, final accuracy,
paired by seed, two standard errors). Binder-rate swing = highest minus lowest binder
rate among the tasks of a stream.

**F1. Honest constraints.** Hypothesis: PDCL under-reacts because its constraints are
measured on stored examples that the model also trains on. Two variants behind a trainer
option that is off by default:

- *oracle*: the dual variable of an earlier task is updated from that task's validation
  loss; its constraint level is 1.1 x the task's validation loss at the end of its phase.
  Uses data a deployed method would not have; an upper bound on what honest measurement
  can do.
- *split*: half of each task's stored examples is replayed, the other half is never
  trained on after the task's phase and is used only for the dual update; the constraint
  level is 1.1 x the loss on that half at the end of the task's phase.

Streams `skew:0.7`, `0.8`, `0.9`, `0.95`, `aglen`, `rounds`; memory 200 and 1,000.
Methods: fine-tuning, ER with fixed weight 0.03, 0.3, 1, 3, default PDCL, the two
variants. "Best fixed weight" = the best of the four weights on that stream and memory,
picked on the same seeds (biased upward; disclosed). Logged: loss on the stored examples
and validation loss of every earlier task.
Questions and predictions: (a) variant − default PDCL on the four skewed streams.
Prediction: oracle positive on `skew:0.8`, `0.9`, `0.95` and closing at least half of
the gap between default PDCL and the best fixed weight; split positive but smaller.
(b) Safety: variant − fine-tuning on `aglen` and `rounds`. Prediction: neither variant
is more than two standard errors below fine-tuning. (c) The gap between memory loss and
validation loss is larger at memory 200 than at 1,000, and default PDCL's shortfall
against the best fixed weight is smaller at 1,000. The hypothesis fails if the oracle
does not beat default PDCL on the skewed streams.

**F2. Is replay's harm memorization?** `aglen`, `rounds`, `skew:0.9`; memory 50 and 200.
Fine-tuning; ER (w=1) with 32 (default), 8 and 2 replayed examples per earlier task per
step, the replay loss still a mean so that the weight of replay is unchanged and only
the rate of revisiting changes; ER (32) with weight decay 0.01 and 0.1; fine-tuning with
the same weight decay as a control. Logged: loss on the stored examples and validation
loss of earlier tasks. Prediction: on `aglen`, replaying 2 per task removes at least
half of ER's loss against fine-tuning at both memory sizes, while on `skew:0.9` it keeps
at least half of ER's gain. Weight decay: no prediction. The claim fails if ER's loss
on `aglen` does not shrink as the replay batch shrinks.

**F3. Tolerable drift.** Fine-tuning, ER, PDCL, joint ERM on `skew:0.75`, `skew:0.85`
(added to the E1 dial: swings 0, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90) and on
`drift:0.4`, `drift:0.6`, `drift:0.8` (binder rate rising in equal steps over the four
tasks; total swing 0.4, 0.6, 0.8, equal to the swings of `skew:0.7`, `0.8`, `0.9`).
Threshold of a method = the smallest swing at which it is more than two standard errors
above fine-tuning in final accuracy, and stays so at all larger swings. Predictions: on
the alternating dial ER's threshold is at swing 0.6 and PDCL's at 0.7 or 0.8; at equal
swing gradual drift costs fine-tuning less final accuracy than alternation and the
thresholds are higher. The real rounds have a swing of 0.20 (rates 0.49, 0.59, 0.39,
0.53), which I expect to be below every threshold.

**Free analyses** (existing runs only): A, a two-by-two of antigens shared or new
against binder rate stable or shifting; B, the place of the real rounds on the dial;
C, whether PDCL is anywhere more than two standard errors below fine-tuning.

## 0c. Third round: designs H1–H4, fixed before any of their numbers were seen

Written on 2026-11-21 after E1 (main), E2, E3, F2 and F3 were finished and read; F1 and
the long E1 sweep were still training and I had not read F1's numbers. What I had seen
that bears on these designs: PDCL is never below fine-tuning by more than 0.2 points and
never above replay tuned per stream; replay's cost on `aglen` does not depend on the
replay batch; and, second-hand from the user, that the long E1 runs (five seeds) hint
that at 5,000 steps per task replay's harm on `aglen` disappears and both memory methods
beat fine-tuning by 1 to 2 points. No run of H1–H4 existed. Protocol as in section 0
unless stated; every new trainer option is off by default.

**H1. Long training.** 5,000 steps per task, seeds 0–9, `rounds`, `rounds:3210`,
`aglen`, `target`: joint ERM, fine-tuning, PDCL, ER with weight 0.03, 0.3, 1. Outcome:
final accuracy (last step). Predictions, made knowing the hint above: ER w=1 −
fine-tuning is positive beyond two standard errors on `aglen` and `target`; PDCL −
fine-tuning is positive on at least one of the four streams but smaller than ER's gain;
PDCL does not beat the best of the three replay weights on any stream. Secondary,
no prediction: each method at 5,000 against the same method at 1,000 steps per task
(first-round runs), since the paper found that long training lowers held-out accuracy.

**H2. Is the forgetting a shifted output bias?** A class-balanced loss for fine-tuning
(each class gets half of the loss weight in every batch; no memory), on `skew:0.7`,
`0.8`, `0.9`, `0.95`, `drift:0.4`, `0.6`, `0.8`, `agrate`, `rounds`, `aglen`, against
fine-tuning, ER and PDCL. Outcome: share of ER's gain recovered = (balanced − plain
fine-tuning) / (ER − fine-tuning), on the streams where ER's gain exceeds two standard
errors. Predictions: at least half is recovered on every `skew` and `drift` stream with
a real ER gain; less than half on `agrate`, where the shift comes with new antigens; on
`rounds` and `aglen` balanced fine-tuning is within two standard errors of plain
fine-tuning.

**H3. Dual variables that start at 1.** PDCL with every new dual variable initialised at
1 instead of 0, on `skew:0.7`, `0.8`, `0.9`, `0.95`, `aglen`, `rounds`, memory 200 and
1,000, with default PDCL, fine-tuning and ER w=1 rerun alongside. Run only if F1's
oracle variant beat default PDCL. Predictions: on `aglen` and `rounds` it is within two
standard errors of fine-tuning (the dual variables fall quickly when the constraint
holds); on `skew:0.9` and `0.95` it closes less than half of the gap between default
PDCL and ER, because the constraint is still measured on memorised examples and the
dual variables decay.

**H4. Sample-level dual variables.** The per-sample dual variables of Elenter et al.
used to choose which examples to store, checked against the paper before any run.
Against random selection, for PDCL and for ER, on `skew:0.9`, `agrate`, `aglen`,
`rounds`, memory 50 and 200. Outcome: final accuracy, selection by dual variables minus
random. Prediction: a null (within two standard errors) in at least 12 of the 16 cells;
no prediction of sign for the rest. A long shot; a null is an acceptable outcome.

**Free analyses G1–G3** are *exploratory*: they use existing runs and metrics chosen
after the results were seen. G1, accuracy on the last task against mean accuracy on the
earlier tasks at the end of training, with fixed-weight replay tracing a curve. G2,
accuracy on the next, unseen round at the end of each round's phase. G3, share of
earlier tasks whose final validation loss is within 1.1 x its value at the end of the
task's phase.

## 0d. R1: why does replay hurt on the new-antigen stream? Fixed before any R1 run

Written on 2026-11-21 while H1–H4 were training (none of their numbers read). F2 ruled
out the revisiting rate. Two candidate causes: (a) the stored sample is too small to
represent the earlier task; (b) replaying earlier antigens interferes with learning new
ones however much is stored.

**What the existing runs already show** (ER w=1 minus fine-tuning, final accuracy):

| Stream | Memory 50 | Memory 200 | Memory 1,000 | Source |
|---|---|---|---|---|
| `aglen` | −3.96 ± 0.69 | −2.10 ± 0.52 | −0.64 ± 0.42 | E2 |
| `aglen` | −4.03 ± 0.59 | −2.69 ± 0.46 | | F2 |
| `aglen` | | −2.47 ± 0.47 | −0.65 ± 0.46 | F1 |
| `target` | | −1.62 ± 0.60 | | E1 |

On `aglen` the cost shrinks with memory and is a null at 1,000 in two independent
sweeps, which already points to (a). What they do not show: whether the cost keeps
falling past zero (both estimates at 1,000 are still negative), whether ER ever rises
above fine-tuning, and anything about `target` beyond memory 200.

**Design.** `aglen`, `target`, `rounds`; whole stream, 1,000 steps per task, seeds 0–9:
fine-tuning; ER w=1 with memory 200, 1,000, 3,000 and *full* (replay batches drawn from
the whole training split of each earlier task; a trainer option, off by default); PDCL
at the same four sizes. Per-task validation loss is logged. Outcome: final accuracy,
method minus fine-tuning, paired by seed.

**Prediction** (the user's, adopted): the cost is gone by 1,000 and ER is above
fine-tuning at full memory, on `aglen` and `target` (cause a). Cause (b) is supported if
ER is still more than two standard errors below fine-tuning at full memory. Open, no
prediction: whether PDCL stays at fine-tuning or follows ER upward.

## Five-line summary

1. The design rounds are easy because their labels never conflict: forgetting on this data needs a shifting binder rate, and the rounds' swing of 0.20 is far below the 0.70 at which a 200-example memory starts to matter.
2. PDCL's merit is restraint: across 101 settings it is never meaningfully below fine-tuning, while replay with a small memory loses 2 to 2.5 points on new-antigen streams and no fixed replay weight is safe everywhere.
3. PDCL's limit is the same restraint: it never beats replay that is tuned, given a larger memory or trained longer, and it ends 1 to 2 points below replay where forgetting is real; allocation is a null and sample-level selection is harmful.
4. One mechanism explains both: a small memory is memorised, so replaying it hurts the tasks it came from and PDCL's constraints on it look satisfied; an oracle that measures constraints on unseen data closes PDCL's gap, and no deployable variant we tried does.
5. Where only the label balance shifts, a class-balanced loss with no memory beats both methods, so memory is needed only when the shift is tied to the inputs (`agrate`) or when the memory is large enough to stand for the past.
