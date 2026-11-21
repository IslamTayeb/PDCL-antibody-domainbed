"""Train one model. This is the reference implementation: one run per process.

    python -m domainbed.scripts.train --algorithm PDCL_RS --test_envs 0 --seed 0 \
        --data_dir data --output_dir outputs/example

train_batched.py trains many runs at once and follows this file step for step.
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.utils.data

from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib import misc

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data")
    parser.add_argument('--algorithm', default="ERM", choices=algorithms.ALGORITHMS)
    parser.add_argument('--hparams', help='JSON dict of hyperparameters that replace the defaults')
    parser.add_argument('--seed', type=int, default=0, help='data split, initialization and sampling order')
    parser.add_argument('--steps', type=int, default=datasets.AbRosetta.N_STEPS)
    parser.add_argument('--checkpoint_freq', type=int, default=datasets.AbRosetta.CHECKPOINT_FREQ)
    parser.add_argument('--test_envs', type=int, nargs='*', default=[0],
                        help='Held-out rounds. Pass the flag with no value to train on all four.')
    parser.add_argument('--output_dir', default="outputs/example")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(sys.stdout, os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(sys.stderr, os.path.join(args.output_dir, 'err.txt'))

    hparams = hparams_registry.default_hparams(args.algorithm)
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    hparams['n_steps'] = args.steps
    print('Args:', vars(args))
    print('HParams:', hparams)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = datasets.AbRosetta(args.data_dir)

    # Each round is split into a training part ('in') and a validation part ('out').
    in_splits, out_splits = [], []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.seed, env_i))
        in_splits.append(in_)
        out_splits.append(out)

    train_envs = [i for i in range(len(dataset)) if i not in args.test_envs]
    algorithm = algorithms.get_algorithm_class(args.algorithm)(
        dataset.input_shape, dataset.num_classes, len(train_envs), hparams)
    algorithm.to(device)

    # The dataset is small: every split is built once and kept on the device as
    # uint8, and minibatches are sampled there with replacement.
    def materialise(split):
        loader = torch.utils.data.DataLoader(split, batch_size=256, num_workers=2, collate_fn=misc.collate)
        xs, ys = zip(*loader)
        x, y = torch.cat(xs), torch.cat(ys)
        assert torch.equal(x, x.to(torch.uint8).float())
        return x.to(torch.uint8).to(device), y.to(device)

    class EvalBatches:
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __iter__(self):
            for i in range(0, self.x.size(0), 1024):
                yield self.x[i:i + 1024].float(), self.y[i:i + 1024]

    def train_batches(tensors):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        while True:
            batch = []
            for x, y in tensors:
                idx = torch.randint(x.size(0), (hparams['batch_size'],), device=device, generator=generator)
                batch.append((x[idx].float(), y[idx]))
            yield batch

    in_tensors = [materialise(env) for env in in_splits]
    out_tensors = [materialise(env) for env in out_splits]
    minibatches = train_batches([in_tensors[i] for i in train_envs])
    eval_loaders = {f'env{i}_in': EvalBatches(*t) for i, t in enumerate(in_tensors)}
    eval_loaders.update({f'env{i}_out': EvalBatches(*t) for i, t in enumerate(out_tensors)})

    steps_per_epoch = min(len(env) / hparams['batch_size'] for env in in_splits)
    checkpoint_vals = collections.defaultdict(list)
    last_results_keys = None
    for step in range(args.steps):
        step_start_time = time.time()
        step_vals = algorithm.update(next(minibatches))
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if step % args.checkpoint_freq == 0 or step == args.steps - 1:
            results = {'step': step, 'epoch': step / steps_per_epoch}
            for key, val in checkpoint_vals.items():
                # State variables are logged at the checkpoint, not averaged.
                is_state = key.startswith(('dual_', 'buf_', 'eps_'))
                results[key] = val[-1] if is_state else np.mean(val)
            for name, loader in eval_loaders.items():
                results[name + '_acc'] = misc.accuracy(algorithm, loader)

            # Validation accuracy and loss: means over the validation parts of the training rounds.
            results['agg_val_acc'] = float(np.mean([results[f'env{i}_out_acc'] for i in train_envs]))
            results['agg_val_loss'] = float(np.mean(
                [misc.mean_loss(algorithm, eval_loaders[f'env{i}_out']) for i in train_envs]))

            # Test accuracy: over every example of the held-out rounds.
            n_correct, n_total = 0, 0
            for i in args.test_envs:
                for split, parts in (('in', in_splits), ('out', out_splits)):
                    n_correct += results[f'env{i}_{split}_acc'] * len(parts[i])
                    n_total += len(parts[i])
            results['agg_test_acc'] = n_correct / n_total if n_total else float('nan')
            results['mem_gb'] = torch.cuda.max_memory_allocated() / 1024 ** 3

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({'hparams': hparams, 'args': vars(args)})
            with open(os.path.join(args.output_dir, 'results.jsonl'), 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")
            checkpoint_vals = collections.defaultdict(list)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
