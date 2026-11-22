# F2. Is replay's harm memorization?

Branch `claude/pdcl-f2-replay-memorization-3869d5`. Design: `findings.md`, section 0b, F2,
followed without change. 390 runs under `outputs/f2` (spec `results/f2_sweeps.json`), ten
seeds per setting, 4,000 steps (1,000 per task). Every number below is written by
`python -m domainbed.scripts.f2_analysis` from the raw runs into `results/f2_tables.md`
(tables T1–T13) and `results/f2_runs.csv`. "a ± b" is mean ± standard error over seeds,
differences are paired by seed, and a difference is called real only above two standard
errors.

## Answer

**No. Revisiting stored examples less often does not remove replay's cost.** On `aglen`
ER is as far below fine-tuning when it replays 2 stored examples per earlier task per
step as when it replays 32, at both memory sizes. The benefit on `skew:0.9` is also
unchanged. Within the tested range (16 times fewer replays), the replay batch has no
measurable effect on final accuracy on any of the three streams.

| ER − fine-tuning, final accuracy | replay batch 32 | 8 | 2 | ER(2) − ER(32) |
|---|---|---|---|---|
| `aglen`, memory 50 | −4.03 ± 0.59 | −3.99 ± 0.71 | −4.05 ± 0.64 | −0.02 ± 0.22 |
| `aglen`, memory 200 | −2.69 ± 0.46 | −2.73 ± 0.54 | −2.75 ± 0.52 | −0.07 ± 0.22 |
| `rounds`, memory 50 | −0.18 ± 0.46 | −0.15 ± 0.41 | +0.36 ± 0.37 | +0.53 ± 0.31 |
| `rounds`, memory 200 | −0.25 ± 0.35 | −0.02 ± 0.37 | +0.41 ± 0.46 | +0.66 ± 0.35 |
| `skew:0.9`, memory 50 | +4.03 ± 0.32 | +4.00 ± 0.25 | +3.92 ± 0.33 | −0.11 ± 0.36 |
| `skew:0.9`, memory 200 | +5.62 ± 0.55 | +5.91 ± 0.53 | +5.99 ± 0.46 | +0.37 ± 0.38 |

(T4, T7; figure `paper/figures/fig_f2_batch`.) Every entry of the last column is a null.
On `rounds` ER equals fine-tuning at every replay batch.

## Pre-registered prediction and its outcome

Prediction: "on `aglen`, replaying 2 per task removes at least half of ER's loss against
fine-tuning at both memory sizes, while on `skew:0.9` it keeps at least half of ER's
gain." Failure criterion: "the claim fails if ER's loss on `aglen` does not shrink as the
replay batch shrinks."

| Part | Share of the replay-batch-32 effect that remains at replay batch 2 | Outcome |
|---|---|---|
| `aglen`, memory 50: at least half of the loss removed | 1.01 | **failed** |
| `aglen`, memory 200: at least half of the loss removed | 1.03 | **failed** |
| `skew:0.9`, memory 50: at least half of the gain kept | 0.97 | held |
| `skew:0.9`, memory 200: at least half of the gain kept | 1.07 | held |

**The prediction failed, and the failure criterion is met at both memory sizes**: the
loss on `aglen` does not shrink (−4.03, −3.99, −4.05 at memory 50 and −2.69, −2.73,
−2.75 at memory 200 for replay batch 32, 8, 2). The two `skew:0.9` parts held, but only
because the replay batch changes nothing there either.

## Why the manipulation did not bite

Replaying less often did not stop the model from fitting the stored examples (T9–T11,
figure `fig_f2_loss`).

| Memory | Replay batch | Replays of a stored example per 1,000 steps | Memory accuracy at the end, `aglen` (%) | Memory loss, `aglen` | Validation loss of earlier tasks, `aglen` |
|---|---|---|---|---|---|
| 50 | 32 | 880 | 100.0 | 0.003 | 1.256 ± 0.051 |
| 50 | 8 | 320 | 100.0 | 0.003 | 1.256 ± 0.052 |
| 50 | 2 | 80 | 100.0 | 0.005 | 1.229 ± 0.047 |
| 200 | 32 | 320 | 100.0 | 0.012 | 1.120 ± 0.023 |
| 200 | 8 | 80 | 100.0 | 0.016 | 1.111 ± 0.024 |
| 200 | 2 | 20 | 99.4 | 0.042 | 0.999 ± 0.024 |
| fine-tuning | – | – | – | – | 0.598 ± 0.004 |

- A training example of the current task is drawn about 15 times during its own phase
  (10 to 31 on `rounds`). A stored example is replayed 20 to 880 times per 1,000 steps,
  and 60 to 2,640 times over the stream if it is a task-0 example kept to the end (T11).
  Even the lowest setting revisits a stored example more often than a fresh one.
- In every ER setting without weight decay the stored examples end at 99.4–100%
  accuracy, while the validation loss of the same tasks is about twice that of
  fine-tuning on `aglen`. ER's loss on *all* training examples of the earlier tasks is
  also far above fine-tuning's (0.91–1.15 against 0.55 on `aglen`): the model fits the
  stored examples and gets worse on the rest of the tasks they came from.
- A smaller replay batch does narrow the gap between validation and memory loss
  (`aglen`, memory 200, averaged after each task's phase: 0.80, 0.76, 0.61 for replay
  batch 32, 8, 2; T10). That is a change in loss, not in accuracy.

So the result rules out the revisit rate, in the range 2 to 32, as the lever. It does
not rule out memorization as the cause, because memorization was still complete at the
smallest replay batch. The memory-size effect remains: ER's loss on `aglen` is larger at
memory 50 than at 200.

## Weight decay (no prediction)

The batched optimizer applies weight decay **coupled**: `weight_decay x parameter` is
added to the gradient before Adam's moment estimates (L2 in the style of
`torch.optim.Adam`, not AdamW).

| ER − fine-tuning, same weight decay | decay 0 | 0.01 | 0.1 |
|---|---|---|---|
| `aglen`, memory 50 | −4.03 ± 0.59 | −3.03 ± 0.59 | +8.77 ± 0.85 |
| `aglen`, memory 200 | −2.69 ± 0.46 | −1.29 ± 0.42 | +12.12 ± 0.49 |
| `rounds`, memory 50 | −0.18 ± 0.46 | +0.45 ± 0.26 | +4.93 ± 2.24 |
| `rounds`, memory 200 | −0.25 ± 0.35 | +0.49 ± 0.42 | +1.39 ± 1.45 |
| `skew:0.9`, memory 50 | +4.03 ± 0.32 | +9.09 ± 0.45 | +9.84 ± 0.88 |
| `skew:0.9`, memory 200 | +5.62 ± 0.55 | +11.44 ± 0.51 | +13.81 ± 0.63 |

(T8, figure `fig_f2_decay`.)

- **Decay 0.01 does not remove ER's cost on `aglen`**: ER stays below fine-tuning with
  the same decay at both memory sizes. The gap narrows by 1.00 ± 0.23 and 1.39 ± 0.30
  points, but that comes from fine-tuning getting worse (−1.12 ± 0.23), not from ER
  getting better (−0.11 ± 0.25 and +0.28 ± 0.18, both nulls). The stored examples are
  still fitted to 100%.
- **Decay 0.1 is not a usable control.** Fine-tuning collapses to chance on `aglen` and
  `rounds` (51.7% and 50.2%, validation loss 0.693), so the large positive differences
  there mean that ER collapses less, not that replay became safe. ER itself loses 1.5 to
  3.5 points on `aglen` and 11 to 15 on `rounds` against ER without decay.
- On `skew:0.9` decay hurts fine-tuning (−5.6 and −8.4 points) and leaves ER at memory
  200 unchanged (+0.26 ± 0.35, −0.19 ± 0.68), so ER's margin grows.

## Exploratory (beyond the design)

- **Where ER's cost on `aglen` sits.** It is extra forgetting of the earlier tasks, not
  worse learning of the current one: forgetting is 4.3 ± 0.8 points higher than
  fine-tuning's at memory 50 and 1.5 ± 0.6 higher at 200, while accuracy at the end of
  each task's own phase is only 0.65 and 1.41 points lower (T5, T6). The last task ends
  equal (T12).
- **On `skew:0.9` at memory 200 a smaller replay batch trades one thing for another.**
  Forgetting falls (26.7 to 23.3 points, T2) and accuracy at the end of each task's own
  phase falls with it (83.1 to 80.8, T3); final accuracy does not move.
- **Dose-response on the computed replay count** (T13, `fig_f2_revisits`): six points
  per stream. The correlations with final accuracy (−0.48 on `aglen`, −0.56 on
  `skew:0.9`, −0.86 on `rounds`) come from memory size, not from the replay batch:
  within a memory size the lines are flat on `aglen` and `skew:0.9`. Memory 50 with replay
  batch 2 and memory 200 with replay batch 8 replay a stored example equally often (80
  per 1,000 steps) and differ by 1.3 points on `aglen`, so memory size matters beyond
  the revisit rate.
- The bootstrap intervals for the shares in T7 are exploratory.

## Trainer change

`domainbed/scripts/train_batched.py`, hparam `replay_batch`: stored examples replayed
per earlier task per step. Absent or 0 keeps the previous behaviour, which is
**`batch_size` (32) stored examples per earlier task, drawn without replacement, or all
of the task's stored examples when it has fewer than 32**. The replay loss is still the
mean over the replayed examples of each task, so the weight of replay is unchanged.
Values above `batch_size` are clamped to it. New logged keys at every checkpoint:
`mem_loss_k`, `mem_acc_k` (all stored examples of task k), `env{e}_in_loss`,
`env{e}_out_loss`.

Check that existing runs are unchanged: default ER and fine-tuning (800 steps, two
seeds; `aglen` with memory 50, `skew:0.9` with memory 200) give identical logged values
before and after the change (maximum absolute difference 0).

## Limits

- **"Replay batch 32" at memory 50 is not 32 throughout.** Task 0 has 50, 25 and 16
  stored examples in phases 2 to 4, so the default replays 32, 25 and 16 of them, i.e.
  every stored example at every step in the last two phases. Replay batches 8 and 2 are
  exact at both memory sizes.
- A smaller replay batch also makes the replay gradient noisier. The design does not
  separate that from the revisit rate; since nothing moved, it does not change the
  conclusion.
- The smallest replay batch tested is 2. Rates low enough to leave the stored examples
  unfitted were not reached, so "memorization is the cause" is neither confirmed nor
  refuted here.
- Weight decay is coupled L2 at one learning rate; 0.1 is too strong for this network
  and 0.01 too weak to prevent fitting the memory. No value in between was run.
- Replay counts are computed from the logged memory sizes, not counted during training.
- ER − fine-tuning on `aglen` at memory 200 is −2.69 ± 0.46 here against −2.1 in the
  first round: same seeds, different random replay draws (the draw depends on which
  runs are batched together).
- One network, one learning rate, three streams, 1,000 steps per task, ER with fixed
  weight 1 only. PDCL was not run.
