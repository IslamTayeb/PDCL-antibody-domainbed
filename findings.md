# Where does PDCL's idea help on Antibody DomainBed?

Branch `pdcl-scope`. The results of the paper are untouched; everything here is new runs
under `outputs/e*` made with `domainbed/scripts/train_batched.py`.

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
