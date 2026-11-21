# Does primal-dual continual learning help across antibody design rounds?

A test of primal-dual continual learning (PDCL, [Elenter et al., 2023](https://arxiv.org/abs/2310.00154)) on [Antibody DomainBed](https://arxiv.org/abs/2407.21028), where a surrogate model sees antibody design rounds one at a time. Supervised by Prof. Navid NaderiAlizadeh (Duke University).

Paper: [`paper/main.pdf`](paper/main.pdf).

## Findings

1. **The benchmark's model selection hides over-training.** Held-out accuracy peaks after about 1,500 steps and then falls by three points, while validation accuracy stays flat. Selecting checkpoints by validation loss fixes this and removes apparent gains of V-REx, GroupDRO and CORAL over ERM.
2. **PDCL equals fine-tuning.** At six training lengths, PDCL and sequential fine-tuning with no memory differ by at most 0.2 points on held-out rounds, and tuning does not separate them.
3. **The dual variables explain why.** Plain replay over-fits its 200 stored examples and loses 1 to 2.5 points. PDCL's constraints are nearly satisfied without intervention, so its dual variables stay near zero and the memory goes almost unused.

Leave-one-round-out test accuracy (%), selected by validation loss, mean ± SE over ten seeds:

| Total steps | Joint ERM | Fine-tuning | Replay (ER) | PDCL |
|---|---|---|---|---|
| 1,500 | 66.5 ± 0.1 | 65.9 ± 0.2 | 63.5 ± 0.3 | 66.0 ± 0.2 |
| 3,000 | 66.6 ± 0.2 | 65.8 ± 0.2 | 64.1 ± 0.4 | 66.0 ± 0.2 |
| 15,000 | 66.4 ± 0.2 | 62.9 ± 0.2 | 62.6 ± 0.2 | 62.6 ± 0.2 |

## Layout

| Path | Contents |
|---|---|
| `domainbed/algorithms.py` | ERM, IRM, V-REx, GroupDRO, CORAL, SD, and the sequential methods (`Finetune`, `ER`, `ER_FF`, `PDCL_RS`, `PDCL`) |
| `domainbed/scripts/train.py` | reference trainer, one run per process |
| `domainbed/scripts/train_batched.py` | trains many runs as one batched model on one GPU; produced every reported number |
| `domainbed/scripts/paper_assets.py` | numbers, tables and figures of the paper |
| `results/sweeps.json` | every run of the study, as input to `train_batched.py` |
| `results/runs.tar.xz` | the logged results of all runs |

`PDCL_RS` is PDCL as published (random selection of stored examples); `PDCL` and `ER_FF` add farthest-first selection.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data && curl -L -o data/abdb.csv https://zenodo.org/api/records/11446107/files/abdb.csv/content
```

Tables and figures from the stored results:

```bash
mkdir -p outputs && tar xJf results/runs.tar.xz -C outputs
.venv/bin/python -m domainbed.scripts.paper_assets
```

All runs from scratch (several GPU-hours on one GPU; finished runs are skipped):

```bash
.venv/bin/python -m domainbed.scripts.train_batched --spec results/sweeps.json
```

One run with the reference trainer (test on round 0, train on the rest):

```bash
.venv/bin/python -m domainbed.scripts.train --algorithm PDCL_RS --test_envs 0 --seed 0 --output_dir outputs/example
```

## Notes

- `--envs -1` (batched) or `--test_envs` with no value (reference) trains on all four rounds in sequence.
- The batched trainer follows the reference trainer step for step, with the same initial weights, data splits and minibatch stream. Results agree within seed noise but are not bit-identical.
- For sequential methods a checkpoint is selected only inside the final phase, so a reported model has trained on every round.
- Labels are the sign of a pyRosetta ΔΔG estimate, a computed proxy for binding.
- An earlier version of this repository reported 66.27% for PDCL and a win over the baselines. That number came from selecting checkpoints on the test set with one seed.

The code started as a fork of [DomainBed](https://github.com/facebookresearch/DomainBed) and the Antibody DomainBed code.
