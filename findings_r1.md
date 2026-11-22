# R1. Why does replay hurt on the new-antigen stream: too little memory, or interference?

Branch `claude/pdcl-r1-replay-memory-size-f4b013`, cut from 1f8ffb8. Design: `findings.md` section 0d,
followed without change. 270 runs (all finished), made with `train_batched.py`.

- Streams `aglen`, `target`, `rounds`; fine-tuning, ER (weight 1) and PDCL (PDCL_RS as published) at
  memory 200, 1,000, 3,000 and full; 4,000 steps (1,000 per task), seeds 0–9. Numbers are mean ±
  standard error over ten seeds; differences are paired by seed; **bold** = larger than two standard
  errors, anything else is a null.
- Spec `results/r1_sweeps.json`; runs `outputs/r1/<stream>/<method>/env-1_seed<s>`; analysis
  `python -m domainbed.scripts.r1_analysis` → `results/r1_runs.csv`, `results/r1_tables.md` (all
  tables), `paper/figures/fig_r1_memory.{pdf,png}`.

## Answers

1. **Does ER's cost fall to a null as memory grows, and at what size?** Yes, at 1,000 on both streams.
   `aglen`: **−2.44 ± 0.45** at 200, −0.65 ± 0.52 at 1,000, +0.05 ± 0.48 at 3,000. `target`:
   **−1.53 ± 0.55** at 200, +0.21 ± 0.44 at 1,000, **+1.17 ± 0.48** at 3,000.
2. **Is ER above fine-tuning at full memory?** Yes on all three streams: `aglen` **+1.56 ± 0.37**,
   `target` **+1.80 ± 0.43**, `rounds` **+3.90 ± 0.53**.
3. **Does PDCL stay at fine-tuning or follow ER upward?** It moves up, but less than ER. `aglen`: a null
   up to 3,000 and **+0.49 ± 0.18** at full, where it is **−1.07 ± 0.33** below ER. `target`:
   **+1.05 ± 0.33** from 1,000 and **+1.56 ± 0.39** at full, level with ER (−0.24 ± 0.28). `rounds`:
   above fine-tuning at every size (**+0.56** to **+1.98**) and below ER from 1,000 (**−0.96 ± 0.47** to
   **−1.91 ± 0.41**). At memory 200 PDCL is above ER on `aglen` and `target` (**+2.28 ± 0.28**,
   **+2.11 ± 0.30**). Its dual variables stay small (0.003 to 0.016 on `aglen` and `target`, 0.016 to
   0.106 on `rounds`, against ER's weight of 1), so it replays little at any size.

**Pre-registered prediction** (cost gone by 1,000, ER above fine-tuning at full memory, on `aglen` and
`target`): **held** on both streams. The test for cause (b), ER more than two standard errors below
fine-tuning at full memory, **failed** on both.

**Cause.** The data support (a), a stored sample that is too small. With a small memory ER fits the
stored examples and the earlier tasks' validation loss rises far above fine-tuning's: on `aglen` at
memory 200 the stored loss is 0.012 and the validation loss 1.132 against 0.598 for fine-tuning. The
excess shrinks with memory (**+0.534**, **+0.242**, **+0.058**) and reverses at full memory
(**−0.024 ± 0.004**); `target` is the same (**+0.430** to **−0.047**).

**One part of (b) is real but smaller.** ER is below fine-tuning on the *last* task at every memory
size on `target` (**−1.90** to **−2.38**, still **−1.92 ± 0.44** at full) and on `aglen` up to 3,000
(**−1.24**, **−2.18**, **−2.27**; −0.62 ± 0.58 at full). So replay does slow learning of the new task,
and more memory does not remove that. The total turns positive because the gain on the earlier tasks
grows with memory (`aglen` **−2.84 ± 0.65** at 200 to **+2.29 ± 0.42** at full; `target` −1.41 ± 0.78
to **+3.04 ± 0.54**). PDCL shows no cost on the last task at any size (all nulls).

## Final accuracy (%)

| Stream | Memory | Fine-tuning | ER | PDCL | ER − FT | PDCL − FT | PDCL − ER |
|---|---|---|---|---|---|---|---|
| aglen | 200 | 68.00 ± 0.44 | 65.56 ± 0.41 | 67.84 ± 0.36 | **-2.44 ± 0.45** | -0.16 ± 0.22 | **+2.28 ± 0.28** |
| aglen | 1,000 | 68.00 ± 0.44 | 67.35 ± 0.40 | 68.12 ± 0.39 | -0.65 ± 0.52 | +0.12 ± 0.22 | +0.76 ± 0.39 |
| aglen | 3,000 | 68.00 ± 0.44 | 68.05 ± 0.49 | 68.25 ± 0.39 | +0.05 ± 0.48 | +0.25 ± 0.18 | +0.20 ± 0.41 |
| aglen | full | 68.00 ± 0.44 | 69.56 ± 0.23 | 68.49 ± 0.43 | **+1.56 ± 0.37** | **+0.49 ± 0.18** | **-1.07 ± 0.33** |
| target | 200 | 67.48 ± 0.60 | 65.94 ± 0.32 | 68.05 ± 0.36 | **-1.53 ± 0.55** | +0.58 ± 0.45 | **+2.11 ± 0.30** |
| target | 1,000 | 67.48 ± 0.60 | 67.69 ± 0.30 | 68.53 ± 0.44 | +0.21 ± 0.44 | **+1.05 ± 0.33** | **+0.84 ± 0.34** |
| target | 3,000 | 67.48 ± 0.60 | 68.65 ± 0.29 | 68.70 ± 0.58 | **+1.17 ± 0.48** | **+1.22 ± 0.44** | +0.05 ± 0.52 |
| target | full | 67.48 ± 0.60 | 69.28 ± 0.37 | 69.04 ± 0.52 | **+1.80 ± 0.43** | **+1.56 ± 0.39** | -0.24 ± 0.28 |
| rounds | 200 | 66.54 ± 0.56 | 66.69 ± 0.47 | 67.10 ± 0.44 | +0.15 ± 0.42 | **+0.56 ± 0.26** | +0.41 ± 0.28 |
| rounds | 1,000 | 66.54 ± 0.56 | 68.57 ± 0.43 | 67.61 ± 0.37 | **+2.02 ± 0.49** | **+1.07 ± 0.32** | **-0.96 ± 0.47** |
| rounds | 3,000 | 66.54 ± 0.56 | 69.34 ± 0.38 | 67.76 ± 0.47 | **+2.79 ± 0.48** | **+1.21 ± 0.37** | **-1.58 ± 0.37** |
| rounds | full | 66.54 ± 0.56 | 70.44 ± 0.14 | 68.53 ± 0.40 | **+3.90 ± 0.53** | **+1.98 ± 0.37** | **-1.91 ± 0.41** |

Forgetting, accuracy on the last and on the earlier tasks, losses, dual variables and stored counts:
`results/r1_tables.md`, T2–T7. Forgetting of ER at full memory is 0.50 ± 0.28 (`aglen`), 0.71 ± 0.21
(`target`) and −0.25 ± 0.14 (`rounds`), against 3.2 to 3.8 for fine-tuning.

## How the memory is filled

- `buffer_size` is the total over all earlier tasks. At the end of each phase it is re-divided: equally
  for ER (floor of total / tasks stored, the newest task taking the remainder), by the dual variables
  for PDCL (half equal, half in proportion to the duals). The last task is never stored.
- `pool_size` (default 1,000) is the number of candidates kept from a phase, so it caps what a task
  can store. It does not bind at memory 200 or 1,000 (left at the default, as in E2 and F1) and would
  at 3,000, so those runs use `pool_size` 3,000.
- *Full* is the new hparam `full_memory` (default off): the memory of an earlier task is its whole
  training split; replay batches, and PDCL's constraint and dual update, are drawn from it. Trainer
  diff: 17 lines. Default runs are unchanged: a short fine-tuning, ER and PDCL sweep written to the
  same paths before and after the change agrees on all 2,772 logged values (largest difference 0).
- Stored per earlier task during the last phase, ER (from `buf_k`): 66/66/68 at 200, 333/333/334 at
  1,000, 1,000 each at 3,000; full is 2,149/2,076/2,130 on `aglen`, 1,428/2,375/2,345 on `target`,
  1,041/3,099/1,396 on `rounds`. PDCL's division varies by seed (for example 500 to 1,500 for a task
  at 3,000).
- **Fewer distinct examples than slots.** The candidates are sampled from the minibatch stream, which
  draws with replacement, so an example can be stored more than once (new log key `buf_unique_k`).
  Distinct examples per task: about 65 of 66 at 200, about 310 of 333 at 1,000, about 800 of 1,000 at
  3,000 on `aglen` (640 to 850 on `rounds`). Memory 3,000 is therefore closer to 2,400 distinct
  examples, a third or more of each task's training split.

## Limits

- One replay weight (1), one training length, ten seeds. The cost at 1,000 on `aglen` is a null, not a
  shown zero: its estimate is negative here (−0.65 ± 0.52) as in E2 and F1.
- Memory 3,000 and full are not comparable budgets on `rounds` and `target`, whose tasks differ in
  size (1,041 to 3,099 training examples).
- The cost on the last task was not the pre-registered outcome; it is reported as a secondary reading.
- Full memory stores the training data of every earlier task, so it is joint replay, not a
  memory-limited method.
