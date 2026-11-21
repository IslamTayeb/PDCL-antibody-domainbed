import hashlib
import sys

import numpy as np
import torch


def seed_hash(*args):
    """An integer hash of the arguments, for use as a random seed."""
    return int(hashlib.md5(str(args).encode("utf-8")).hexdigest(), 16) % (2**31)


class _SplitDataset(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, keys):
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """A random split of the dataset into n examples and the rest."""
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    return _SplitDataset(dataset, keys[:n]), _SplitDataset(dataset, keys[n:])


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def accuracy(network, loader):
    correct, total = 0, 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            correct += network.predict(x).argmax(1).eq(y.argmax(1)).sum().item()
            total += len(x)
    network.train()
    return correct / total


def mean_loss(network, loader):
    """Mean cross-entropy of a classifier over a loader."""
    total, count = 0.0, 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            total += torch.nn.functional.cross_entropy(network.predict(x), y, reduction='sum').item()
            count += len(x)
    network.train()
    return total / count


def print_row(row, colwidth=10):
    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print("  ".join(format_val(x) for x in row))


class Tee:
    """Write to a file and to the stream it replaces."""

    def __init__(self, stream, fname):
        self.stream = stream
        self.file = open(fname, "a")

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()
