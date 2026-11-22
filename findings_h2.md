# H2. Is the forgetting a shifted output bias?

Branch `claude/pdcl-h2-output-bias-754dce`. Design: `findings.md`, section 0c, H2, followed
without change. 400 runs under `outputs/h2` (spec `results/h2_sweeps.json`): ten streams,
four methods (fine-tuning, fine-tuning with a class-balanced loss, ER, PDCL as published,
all at default hyperparameters), seeds 0–9, 4,000 steps (1,000 per task). Every number
below is written by `python -m domainbed.scripts.h2_analysis` from the raw runs into
`results/h2_tables.md` (T1–T5) and `results/h2_runs.csv`. "a ± b" is mean ± standard
error over seeds, differences are paired by seed, and a difference is called real only
above two standard errors. Figure: `paper/figures/fig_h2_balance`.

## Answer

**When only the binder rate shifts (`skew`, `drift`), rebalancing the loss recovers all of
replay's gain, with no memory: the share recovered is 0.99 to 2.95 on the six streams
where replay has a real gain. When the binder rate shifts together with new antigens
(`agrate`), it recovers 6%.** So on `skew` and `drift` the forgetting behaves as a shifted
output bias, and on `agrate` it does not.

| Stream | ER − fine-tuning | Balanced − fine-tuning | PDCL − fine-tuning | Share of ER's gain recovered [95%] |
|---|---|---|---|---|
| `skew:0.7` | +0.20 ± 0.62 | **+1.79 ± 0.36** | +0.30 ± 0.19 | not defined (no real ER gain) |
| `skew:0.8` | **+1.18 ± 0.55** | **+2.88 ± 0.48** | +0.17 ± 0.42 | 2.44 [1.58, 9.17] |
| `skew:0.9` | **+5.52 ± 0.35** | **+9.20 ± 0.58** | **+4.19 ± 0.27** | 1.67 [1.52, 1.84] |
| `skew:0.95` | **+9.21 ± 0.50** | **+9.09 ± 0.55** | **+7.57 ± 0.41** | 0.99 [0.88, 1.09] |
| `drift:0.4` | **+1.24 ± 0.45** | **+3.66 ± 0.58** | **+0.58 ± 0.27** | 2.95 [1.97, 7.21] |
| `drift:0.6` | **+2.65 ± 0.52** | **+6.31 ± 0.63** | **+1.51 ± 0.33** | 2.38 [1.80, 3.46] |
| `drift:0.8` | **+4.13 ± 0.57** | **+7.57 ± 0.44** | **+2.69 ± 0.57** | 1.83 [1.51, 2.38] |
| `agrate` | **+10.20 ± 0.47** | +0.60 ± 0.30 | **+5.72 ± 0.30** | 0.06 [0.00, 0.11] |
| `rounds` | −0.12 ± 0.48 | +0.27 ± 0.45 | +0.19 ± 0.22 | not defined |
| `aglen` | **−2.52 ± 0.52** | **−0.96 ± 0.37** | +0.05 ± 0.18 | not defined |

Final accuracy, points (T2, T5). Share = mean(balanced − fine-tuning) / mean(ER −
fine-tuning); interval from a paired bootstrap over seeds (20,000 draws). The intervals
are wide where ER's gain is small (`skew:0.8`, `drift:0.4`).

Other things the table shows:

- Balanced fine-tuning beats ER by a real margin on five of the six `skew`/`drift` streams
  with an ER gain (+1.70 to +3.68 points) and on `skew:0.7` (+1.58 ± 0.53); it ties ER on
  `skew:0.95` (−0.12 ± 0.50). It beats PDCL on all seven.
- Forgetting follows the same pattern (T3, T4): on `skew:0.9` it is 43.97 for fine-tuning,
  26.90 for ER, 31.91 for PDCL and 14.04 for balanced fine-tuning. On `skew:0.95`
  balancing still leaves 31.03 points of forgetting, about the same as ER (33.48).
- On `agrate` balancing lowers forgetting (−9.27 ± 0.68) but final accuracy does not move
  (+0.60 ± 0.30, a null); ER is 9.61 ± 0.52 points above balanced fine-tuning.
- On `rounds` all four methods are equal. On `aglen` balancing costs about a point.

## Pre-registered predictions

| Prediction | Observed | Outcome |
|---|---|---|
| `skew:0.7`: at least half recovered | ER gain +0.20 ± 0.62, not real | not tested (condition not met) |
| `skew:0.8`: at least half recovered | 2.44 [1.58, 9.17] | held |
| `skew:0.9`: at least half recovered | 1.67 [1.52, 1.84] | held |
| `skew:0.95`: at least half recovered | 0.99 [0.88, 1.09] | held |
| `drift:0.4`: at least half recovered | 2.95 [1.97, 7.21] | held |
| `drift:0.6`: at least half recovered | 2.38 [1.80, 3.46] | held |
| `drift:0.8`: at least half recovered | 1.83 [1.51, 2.38] | held |
| `agrate`: less than half recovered | 0.06 [0.00, 0.11] | held |
| `rounds`: balanced within two standard errors of fine-tuning | +0.27 ± 0.45 | held |
| `aglen`: balanced within two standard errors of fine-tuning | −0.96 ± 0.37 (2.6 standard errors) | **failed** |

Eight held, one failed, one could not be tested. Every interval of a held prediction
excludes 0.5.

## What was implemented

Hparam `balanced_loss` in `domainbed/scripts/train_batched.py` (11 lines, off by default).
For a run with it on, the loss on the current task's batch of 32 is
`sum_i w_i * CE_i` with `w_i = 0.5 / n_c(i)`, where `n_c(i)` is the number of examples in
the batch with the same label as example i; each class therefore gets total weight 0.5.
If the batch contains a single class, the plain mean is used. It acts only on the
current-task term, uses no memory, and changes nothing at evaluation. Here it was used
with `Finetune` only. Because the trainer's constraint level and `current_loss` log are
computed from this term, `current_loss` of these runs is the balanced loss.

Check of the default path: a 400-step default `Finetune` run on `skew:0.9`, seed 0, gave a
`results.jsonl` identical in every logged metric before and after the change. In a batch
that mixes default and balanced runs, the default run differs from the stand-alone run by
at most 0.002 in any logged value, which is the trainer's usual dependence of floating
point on batch composition.

## Limits

- **Balanced accuracy is not reported.** The trainer logs accuracy and loss per task and
  split, not per class, and saves no weights, so class recalls cannot be computed from
  these runs. This matters for reading the result: the outcome is plain accuracy on
  validation sets whose binder rate matches each task. Plain fine-tuning can gain on the
  last task by predicting its majority class, and loses that on tasks with the opposite
  rate; balancing removes that incentive. The data do not show whether balancing also
  preserves more of the learned features.
- The balanced loss is not the same as correcting a bias after training. It changes
  every gradient step. "Shifted output bias" is supported by the fact that a memory-free
  reweighting matches or beats replay, not by a direct measurement of the output bias.
- Shares above 1 mean balancing beats ER at its default setting (weight 1, default
  memory), not that it beats replay tuned per stream.
- One training length (1,000 steps per task), one batch size (32), synthetic streams for
  all the positive results. On the real `rounds` stream there is nothing to recover.
- At `skew:0.95` a batch of 32 has on average 1.6 examples of the minority class and is
  single-class (plain mean) in roughly a fifth of the steps, so the balanced gradient is
  noisy there; this may be why its advantage over ER disappears at that level. Not tested.
- The `aglen` cost of balancing (−0.96 ± 0.37) was not predicted and is unexplained.
