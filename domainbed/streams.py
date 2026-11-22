"""Task streams: ways to group the examples of the dataset into tasks.

The benchmark's stream is 'rounds' (the four design rounds). Every other stream
regroups the same 10,606 examples, so that the cost of retaining an earlier
task can be varied while data, model and methods stay fixed.

    rounds            the four design rounds, in order
    rounds:3210       the design rounds in another order
    skew:Q[:T]        T (default 4) equal tasks; a fraction Q of the binders goes
                      to the even tasks and 1-Q to the odd ones (non-binders the
                      other way round). Q=0.5 is a random split; Q -> 1 makes the
                      binder rate alternate between high and low.
    drift:D           four equal tasks; the binder rate rises in equal steps from
                      task to task, by D in total (gradual drift)
    aglen             antigens sorted by length, cut into four groups (no antigen
                      shared between tasks; binder rates stay close)
    target            HIV-1, then the SARS-CoV-2 antigens in three groups by length
    agrate            antigens sorted by their binder rate, cut into four groups
                      (no antigen shared, and the binder rate rises task by task;
                      uses the labels, so a stress test rather than a natural stream)
    hetero            four equal tasks with binder shares that make tasks 0 and 2
                      costly to retain and task 1 free (see task_ids)

The grouping never depends on the run's seed; a seed still fixes the
train/validation split inside each task, the initialization and the batches.
"""

import contextlib
import io
from functools import lru_cache

import numpy as np
import pandas as pd

from domainbed import datasets


@lru_cache(None)
def frame(data_dir):
    """All examples in the row order of the token cache (round by round)."""
    with contextlib.redirect_stdout(io.StringIO()):
        ds = datasets.AbRosetta(data_dir)
    return pd.concat([env.df for env in ds], ignore_index=True)


def split_by_shares(labels, pos_shares, neg_shares, seed=0):
    """Deal binders and non-binders to tasks in the given shares."""
    rng = np.random.RandomState(seed)
    task = np.zeros(len(labels), dtype=int)
    for value, shares in ((True, pos_shares), (False, neg_shares)):
        rows = rng.permutation(np.flatnonzero(labels == value))
        cuts = (np.cumsum(shares)[:-1] * len(rows)).round().astype(int)
        for k, part in enumerate(np.split(rows, cuts)):
            task[part] = k
    return task


def task_ids(data_dir, stream):
    df = frame(data_dir)
    labels = df['is_binder'].values.astype(bool)
    name, *args = stream.split(':')
    if name == 'rounds':
        rounds = np.unique(df['env'].values, return_inverse=True)[1]
        order = [int(c) for c in args[0]] if args else sorted(set(rounds))
        return np.argsort(order)[rounds]
    if name == 'skew':
        q, T = float(args[0]), int(args[1]) if len(args) > 1 else 4
        pos = np.array([q if k % 2 == 0 else 1 - q for k in range(T)])
        return split_by_shares(labels, pos / pos.sum(), pos[::-1] / pos.sum() if T % 2 == 0
                               else (1 - pos) / (1 - pos).sum())
    if name == 'drift':
        # Four equal tasks whose binder rate rises in equal steps; D is the
        # rate of the last task minus that of the first.
        rate = labels.mean() + float(args[0]) * (np.arange(4) - 1.5) / 3
        return split_by_shares(labels, rate / rate.sum(), (1 - rate) / (1 - rate).sum())
    if name == 'hetero':
        # Binder shares of the four tasks: task 1 is a random quarter of the
        # data; tasks 0 and 2 are rich in binders and task 3 is poor in them.
        return split_by_shares(labels, (.35, .25, .35, .05), (.15, .25, .15, .45))
    if name in ('target', 'aglen', 'agrate'):
        # Whole antigens are kept together, so no antigen (and no seed antibody)
        # is shared between tasks. Antigens are ordered by a property and cut
        # into groups of equal numbers of examples.
        ag = df['ag']
        if name == 'agrate':
            key = df.groupby('ag')['is_binder'].mean()       # the antigen's binder rate
        else:
            key = pd.Series({a: len(a) for a in ag.unique()})  # the antigen's length
        pool = df if name != 'target' else df[df['target'] != 'HIV1']
        groups = 4 if name != 'target' else 3
        counts = pool['ag'].value_counts()
        order = sorted(counts.index, key=lambda a: (key[a], a))
        mid = counts[order].cumsum() - counts[order] / 2
        group = np.minimum(groups - 1, (groups * mid / len(pool)).astype(int))
        task = np.zeros(len(df), dtype=int)                  # HIV-1 is task 0 of 'target'
        task[pool.index] = pool['ag'].map(group).values + (name == 'target')
        return task
    raise ValueError(f"unknown stream {stream!r}")


@lru_cache(None)
def task_order(data_dir, stream):
    """(row order that sorts the examples by task, task sizes)."""
    task = task_ids(data_dir, stream)
    return np.argsort(task, kind='stable'), np.bincount(task).tolist()
